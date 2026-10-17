"""Replay identical particle counts / clumping draws through unmodified Python.
This isolates numerical grain processing from different random number streams.
"""
import json
from pathlib import Path
import numpy as np
import spektrafilm.model.grain as grain
from spektrafilm.model.density_curves import interp_density_cmy_layers
root=Path(__file__).resolve().parents[2]
w,h=37,29
x,y=np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h));density=np.stack([1.8*x,1.5*y,1.2*(x+y)/2],axis=-1).astype('float32').astype(float)
fractions=np.array([.2,.3,.5]);channel=np.array([1.,.8,1.2]);area=.2;particle_scale=np.array([1.6,1.6,3.2]);layer_scale=np.array([2.,1.,.5]);dmin=np.array([.03,.04,.02]);uniformity=np.array([.97,.99,.96]);pixel=5.
rng=np.random.default_rng(734);counts=rng.integers(0,45,size=(3,h,w,3)).astype('float32');var=np.log1p(.2**2);micro=rng.lognormal(-var/2,var**.5,size=(h,w,3)).astype('float32').astype(float)
draws=np.concatenate([counts.reshape(-1),micro.reshape(-1)]).tolist()
original_poisson=grain.scipy.stats.poisson.rvs;original_binom=grain.scipy.stats.binom.rvs;original_lognormal=grain.fast_lognormal_from_mean_std
cases=[]
try:
 for positive,cloud,micro_blur,post in [(False,16.,25.,.65),(True,40.,0.,4.),(False,0.,0.,0.)]:
  t=np.linspace(0,1,65)
  if positive:t=t[::-1]
  curves=2*t[:,None]*channel;layers=curves[:,None,:]*fractions[None,:,None]
  parts=interp_density_cmy_layers(density,curves,layers,positive_film=positive)
  maxima=np.max(layers,axis=0);fraction=maxima/maxima.sum(axis=0);min_layers=fraction*dmin;max_layers=maxima+min_layers
  npp=pixel**2*fraction/(area*particle_scale[None,:]*layer_scale[:,None])
  counter=[0]
  def binomial(*args,**kwargs):
   call=counter[0];counter[0]+=1;ch,sl=divmod(call,3);return counts[sl,:,:,ch].astype('int64')
  grain.scipy.stats.poisson.rvs=lambda lam:np.zeros_like(lam,dtype='int64')
  grain.scipy.stats.binom.rvs=binomial
  grain.fast_lognormal_from_mean_std=lambda mean,std:micro.copy()
  expected=grain.apply_grain_to_density_layers(parts.copy(),maxima.copy(),pixel_size_um=pixel,particle_area_um2=area,particle_scale=particle_scale,particle_scale_layers=layer_scale,density_min=dmin,grain_uniformity=uniformity,grain_blur=post,grain_blur_dye_clouds_um=cloud,grain_micro_structure=(micro_blur,1000.),use_fast_stats=False)
  table=[]
  for i in range(65):
   for c in range(3):table.append([(-1 if positive else 1)*curves[i,c],*layers[i,:,c]])
  # This is the f32 transport boundary used by the GPU backend.
  mins=min_layers.astype('float32');maxs=(maxima+mins.astype(float)).astype('float32');particles=npp.astype('float32')
  packed=[{'density_min':mins[l].tolist(),'density_max':maxs[l].tolist(),'particles':particles[l].tolist(),'cloud_sigma':(cloud*np.sqrt(maxs[l]/particles[l])).astype('float32').tolist()} for l in range(3)]
  cases.append({'positive':positive,'curves':np.array(table,dtype='float32').tolist(),'layers':packed,'uniformity':uniformity.tolist(),'density_min':dmin.tolist(),'blur':post,'micro_sigma':.2,'micro_blur':micro_blur/pixel,'expected':expected.reshape(-1).tolist()})
finally:
 grain.scipy.stats.poisson.rvs=original_poisson;grain.scipy.stats.binom.rvs=original_binom;grain.fast_lognormal_from_mean_std=original_lognormal
out=root/'crates/spektrafilm-gpu/tests/fixtures/grain_replay.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'size':[w,h],'density':density.reshape(-1).tolist(),'draws':draws,'cases':cases}));print(out)

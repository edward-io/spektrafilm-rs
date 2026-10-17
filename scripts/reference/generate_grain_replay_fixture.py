"""Replay coupled counts through experimental grain, including RMS sizing and USM.
Run with PYTHONPATH=<experimental-28bf883>/src; stochastic draws are supplied only
so this checks processing error separately from different CPU/GPU RNG streams.
"""
import json
from pathlib import Path
import numpy as np
import spektrafilm.model.grain as grain
from spektrafilm.model.density_curves import interp_density_cmy_layers
root=Path(__file__).resolve().parents[2]
w,h=37,29
x,y=np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h))
density=np.stack([1.8*x,1.5*y,1.2*(x+y)/2],axis=-1).astype('float32').astype(float)
fractions=np.array([.2,.3,.5]);channel=np.array([1.,.8,1.2]);scales=np.array([1.,.4,.25])
dmin=np.array([.03,.04,.02]);uniformity=np.array([.97,.99,.96]);pixel=5.;rms=np.array([6.,8.,12.])
rng=np.random.default_rng(734);counts=rng.integers(0,45,size=(3,h,w,3)).astype('float32')
z=rng.standard_normal((h,w,3));var=np.log1p(.2**2);micro=np.exp(z*var**.5-var/2)
draws=np.concatenate([counts.reshape(-1),micro.astype('float32').reshape(-1)]).tolist()
originals=(grain.scipy.stats.poisson.rvs,grain.scipy.stats.binom.rvs,np.random.standard_normal)
cases=[]
try:
 for positive,cloud,micro_blur,post,usm in [(False,16.,25.,.65,0.),(True,40.,0.,4.,0.),(False,0.,0.,0.,0.),(False,2.,0.,.89,1.5),(True,2.,25.,.89,1.5)]:
  t=np.linspace(0,1,65)
  if positive:t=t[::-1]
  curves=2*t[:,None]*channel;layers=curves[:,None,:]*fractions[None,:,None]
  parts=interp_density_cmy_layers(density,curves,layers,positive_film=positive)
  maxima=np.max(layers,axis=0)
  _,mins,maxs,npp=grain._layer_grain_params(maxima,dmin,rms,scales,uniformity,3,pixel,density_curves_layers=layers)
  counter=[0]
  def binomial(*args,**kwargs):
   call=counter[0];counter[0]+=1;ch,sl=divmod(call,3);return counts[sl,:,:,ch].astype('int64')
  grain.scipy.stats.poisson.rvs=lambda lam:np.zeros_like(lam,dtype='int64')
  grain.scipy.stats.binom.rvs=binomial
  np.random.standard_normal=lambda shape:z.copy()
  expected=grain.apply_grain_to_density_layers(parts.copy(),maxima.copy(),pixel_size_um=pixel,rms_granularity=rms,particle_scale_sublayers=scales,density_min=dmin,grain_uniformity=uniformity,grain_blur=post,grain_blur_dye_clouds_um=cloud,grain_micro_structure=(micro_blur,1000.),use_fast_stats=False,usm_sigma=.7,usm_amount=usm,density_curves_layers=layers)
  table=[]
  for i in range(65):
   for c in range(3):table.append([(-1 if positive else 1)*curves[i,c],*layers[i,:,c]])
  mins=mins.astype('float32');maxs=maxs.astype('float32');particles=npp.astype('float32')
  packed=[dict(density_min=mins[l].tolist(),density_max=maxs[l].tolist(),particles=particles[l].tolist(),cloud_sigma=(cloud*np.sqrt(maxs[l]/particles[l])).astype('float32').tolist()) for l in range(3)]
  cases.append(dict(positive=positive,curves=np.array(table,dtype='float32').tolist(),layers=packed,uniformity=uniformity.tolist(),density_min=dmin.tolist(),blur=post,micro_sigma=.2,micro_blur=micro_blur/pixel,usm_sigma=.7,usm_amount=usm,expected=expected.reshape(-1).tolist()))
finally:
 grain.scipy.stats.poisson.rvs,grain.scipy.stats.binom.rvs,np.random.standard_normal=originals
out=root/'crates/spektrafilm-gpu/tests/fixtures/grain_replay.json'
out.write_text(json.dumps(dict(reference_commit='28bf883e1672e884307edc75852549376e13644e',size=[w,h],density=density.reshape(-1).tolist(),draws=draws,cases=cases)))
print(out)

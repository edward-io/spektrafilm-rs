"""Run with PYTHONPATH=<experimental-28bf883>/src and the reference dependencies.

Captures actual runtime behavior, not a reimplementation of its formulas.
"""
import copy,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from spektrafilm.runtime.api import init_params,simulate
from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.params_builder import digest_params
from spektrafilm.model.convert import FilmConverter
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.model.grain import _layer_grain_params
from spektrafilm.model.color_filters import COLOR_FILTER_LIBRARY,color_filter_transmittance

root=Path(__file__).resolve().parents[2]
(root/'data/filters/taking_filters.json').write_text(json.dumps({s.key:color_filter_transmittance(s.key).tolist() for s in COLOR_FILTER_LIBRARY},indent=2,allow_nan=False)+'\n')
base=digest_params(init_params('kodak_portra_400','kodak_portra_endura'))
base.camera.auto_exposure=False
base.io.input_color_space='sRGB';base.io.input_cctf_decoding=False;base.io.output_cctf_encoding=False
base.film_render.grain.active=False;base.film_render.halation.active=False
base.film_render.dir_couplers.active=False;base.print_render.glare.active=False
base.scanner.unsharp_mask=(0.,0.)
x=(.015+.65*((np.arange(8*6*3)*37)%256)/255).reshape(6,8,3)
cases=[]
dir_details=[]
def add(name,q,image=x):
    q=digest_params(copy.deepcopy(q),apply_stocks_specifics=False)
    data=asdict(q)
    data.pop('film');data.pop('print')
    data["settings"]["apply_stock_specifics"]=False
    y=simulate(image.copy(),q,digest_params_first=False)
    cases.append(dict(name=name,film=q.film.info.stock,print=q.print.info.stock,params=data,width=image.shape[1],height=image.shape[0],input=image.ravel().tolist(),output_width=y.shape[1],output_height=y.shape[0],expected=y.ravel().tolist()))
    print(name,flush=True)
add('default',base)
for method in ['hanatos2025','arctic2026beta04','mallett2019','jakob2019','otsu2018','gauss-lasers']:
    q=copy.deepcopy(base);q.settings.rgb_to_raw_method=method;add(method,q)
for film in ['fujifilm_provia_100f','kodak_doublex','kodak_trix','kodak_vision3_500t']:
    q=digest_params(init_params(film,'kodak_2302' if film in ['kodak_doublex','kodak_trix'] else 'kodak_portra_endura'))
    q.camera.auto_exposure=False;q.io.input_cctf_decoding=False;q.io.output_cctf_encoding=False
    q.film_render.grain.active=False;q.film_render.halation.active=False;q.film_render.dir_couplers.active=False;q.print_render.glare.active=False;q.scanner.unsharp_mask=(0.,0.)
    add(film,q)
    if film in ['kodak_doublex','kodak_trix']:
        q.settings.use_enlarger_lut=True;q.settings.use_scanner_lut=True;add(film+'_luts',q)
for channel in ['film','print']:
    for name in ['gamma','exhaustion','base']:
        q=copy.deepcopy(base);p=getattr(q,channel+'_render')
        if name=='gamma':
            p.chemistry=replace(p.chemistry,active=True,gamma_factor=1.2,gamma_factor_fast=.85,gamma_factor_slow=1.1,gamma_factor_red=1.04,gamma_factor_blue=.94)
        elif name=='exhaustion':p.chemistry=replace(p.chemistry,active=True,developer_exhaustion=.35)
        else:
            p.base.active=True;p.base.scale=.85;p.base.cyan=1.15;p.base.magenta=.9;p.base.yellow=1.08
            if channel=='film':p.base.tilt=.2
        add(channel+'_'+name,q)
for spec in COLOR_FILTER_LIBRARY:
    q=copy.deepcopy(base);q.camera.color_filter=spec.key;add(spec.key,q)
for method in ['average','median','center_weighted','spot','partial','matrix','multi_zone','highlight_weighted']:
    q=copy.deepcopy(base);q.camera.auto_exposure=True;q.camera.auto_exposure_method=method;add('auto_'+method,q)
q=copy.deepcopy(base);q.camera.auto_exposure=True;q.io.crop=True;q.io.crop_size=(.5,.5);q.io.input_cctf_decoding=True;add('auto_encoded_crop',q)
for film in ['kodak_portra_400','fujifilm_provia_100f']:
    q=copy.deepcopy(base);q.film=init_params(film,'kodak_portra_endura').film
    q.film_render.dir_couplers.active=True
    q.film_render.dir_couplers.langmuir_donor_k_rgb=(.4,.8,1.2)
    q.film_render.dir_couplers.langmuir_receiver_k_rgb=(.5,.7,1.1)
    q.film_render.dir_couplers.amount=1.4;q.camera.film_format_mm=.5
    from spektrafilm.model.couplers import compute_dir_couplers_matrix,langmuir_donor_params,receiver_langmuir_params,compute_density_curves_before_dir_couplers
    prepared=SimulationPipeline(copy.deepcopy(q)).film
    curves=prepared.data.density_curves-np.nanmin(prepared.data.density_curves,axis=0)
    matrix=compute_dir_couplers_matrix(q.film_render.dir_couplers)
    k,ref=langmuir_donor_params(curves,q.film_render.dir_couplers.langmuir_donor_k_rgb,3)
    positive=prepared.info.type=='positive'
    kr,cr=receiver_langmuir_params(ref,matrix,q.film_render.dir_couplers.langmuir_receiver_k_rgb,3)
    before=compute_density_curves_before_dir_couplers(curves,prepared.data.log_exposure,matrix*q.film_render.dir_couplers.amount,None if positive else k,ref,receiver_kr=kr if positive else None,receiver_c_ref=cr if positive else None,positive=positive)
    dir_details.append(dict(film=film,curves=curves.tolist(),axis=prepared.data.log_exposure.tolist(),before=before.tolist()))
    add('dir_'+film,q)
    for tap in ['log_e_film','cmy_film']:
        t=copy.deepcopy(q);t.taps.collect=tap;add('dir_'+film+'_'+tap,t)
for route in ['input','input > film > scan','input > film > print > scan','input > convert-film > print > scan','input > convert-film > scan-minus-base','input > convert-film > scan']:
    q=copy.deepcopy(base);q.workflow.route=route
    add(route,q)
    if 'convert-film' in route:
        q.settings.use_convert_lut=True;add(route+'_lut',q)
        q.film_render.convert.exposure_compensation_ev=.3;q.film_render.convert.calibration='1 .01 0 0 1 .01 .01 0 1';add(route+'_lut_calibrated',q)
# Converter oracle is generated from the prepared characteristic model.
pipeline=SimulationPipeline(copy.deepcopy(base))
f=pipeline.film
maximum=np.nanmax(f.data.density_curves-np.nanmin(f.data.density_curves,axis=0),axis=0)
c=FilmConverter(f.data.channel_density,f.data.base_density,base.film_render.base,standard_illuminant('D55')[:],'sRGB',maximum)
densities=np.array([[0,0,0],[.2,.4,.7],[.9,1.4,.3],[2,1,2],[.3,.3,.3]],np.float32)
scans=c._forward_jac(densities,need_jac=False)[0]
converter=dict(densities=densities.tolist(),scans=scans.tolist(),inverted=c.invert(scans).tolist(),bounds=[a.tolist() for a in c.gamut_bounds()])
from spektrafilm.utils.morph_curves import apply_print_curves_morph
morph_params=replace(base.print_render.chemistry,active=True,gamma_factor=1.1,gamma_factor_fast=.9,gamma_factor_slow=1.2,gamma_factor_red=1.05,gamma_factor_green=.95,gamma_factor_blue=1.,developer_exhaustion=.3)
morph=apply_print_curves_morph(base.print.data.log_exposure,base.print.data.density_curves_model,morph_params,profile_type=base.print.info.type)
q=copy.deepcopy(base);q.enlarger.c_filter_neutral=0.;q.enlarger.m_filter_neutral=65.;q.enlarger.y_filter_neutral=55.;q.enlarger.preflash_exposure=.0625;q.settings.neutral_print_filters_from_database=False
pipe=SimulationPipeline(q)
sens=np.nan_to_num(10**pipe.print.data.log_sensitivity)
preflash=pipe._printing_stage._compute_raw_preflash(standard_illuminant('TH-KG3')[:],sens).reshape(3)
q=copy.deepcopy(base);q.scanner.white_correction=True;q.scanner.black_correction=True
pipe=SimulationPipeline(q);pipe.process(np.full((2,2,3),.184))
cref=pipe._color_reference_service
m=(cref._white_level-cref._black_level)/(cref._y_white-cref._y_black+1e-10);offset=cref._black_level-m*cref._y_black
color_reference=dict(m=float(np.ravel(m)[0]),q=float(np.ravel(offset)[0]),printing=float(np.ravel(cref.black_white_printing_exposure_correction())[0]))
gray=np.full((1,1,3),.184);ramp=(.05+.5*((np.arange(4*4*3)*37)%256)/255).reshape(4,4,3)
taps={}
for tap in ['log_e_film','cmy_film','log_e_print','cmy_print']:
    q=copy.deepcopy(base);q.taps.collect=tap;taps[tap]=simulate(gray.copy(),q,digest_params_first=False).reshape(3).tolist()
defaults=asdict(init_params('kodak_portra_400','kodak_portra_endura'));defaults.pop('film');defaults.pop('print')
extra=dict(defaults=defaults,morph=morph.tolist(),preflash=preflash.tolist(),color_reference=color_reference,gray=simulate(gray.copy(),copy.deepcopy(base),digest_params_first=False).reshape(3).tolist(),ramp=simulate(ramp.copy(),copy.deepcopy(base),digest_params_first=False).reshape(-1,3).tolist(),taps=taps)
from spektrafilm.utils.morph_curves import apply_print_curves_morph_with_layers
model=replace(base.print.data.density_curves_model,model_type='sept_norm_cdfs',alphas=np.array([[.4,-.3,.2]]*3))
chemistry=replace(base.print_render.chemistry,active=True,gamma_factor=1.1,gamma_factor_fast=.9,gamma_factor_slow=1.2,gamma_factor_red=1.05,gamma_factor_green=.95,developer_exhaustion=.3)
sept_cases=[]
for positive in [False,True]:
 total,layers=apply_print_curves_morph_with_layers(base.print.data.log_exposure,model,chemistry,profile_type='positive' if positive else 'negative')
 sept_cases.append(dict(model={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in asdict(model).items()},params=asdict(chemistry),positive=positive,axis=base.print.data.log_exposure.tolist(),total=total.tolist(),layers=layers.tolist()))
extra['sept']=sept_cases
out=dict(reference_commit='28bf883e1672e884307edc75852549376e13644e',cases=cases,converter=converter,dir_details=dir_details,extra=extra)
(root/'crates/spektrafilm-core/tests/fixtures/experimental.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

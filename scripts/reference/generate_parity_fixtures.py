"""Generate small, reproducible golden outputs from the adjacent Python repository.

Run with PYTHONPATH=../src using the reference's pinned dependencies.
Stochastic texture is checked separately; these fixtures isolate numerical transforms.
"""
import copy,json,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
from spektrafilm.runtime.api import init_params,simulate
p=init_params('kodak_portra_400','kodak_portra_endura')
p.camera.auto_exposure=False;p.io.input_color_space='sRGB';p.io.input_cctf_decoding=False;p.io.output_cctf_encoding=False
p.film_render.grain.active=False;p.film_render.halation.active=False;p.film_render.dir_couplers.active=False;p.print_render.glare.active=False;p.scanner.unsharp_mask=(0.,0.)
cases=[]
for name in ['base','crop_encoded_auto','rgb_in_passthrough','rgb_pre_encoded','inject_rgb_pre_encoded','encoded_input','rec2020','display_p3','prophoto','acescg','crop','spectral_blur','adaptation_surface','spectral_luts','preview','lut_mode','unbounded','rescale_down','rescale_up','jzazbz','oklch_p3','oklrab_prophoto','input_acescg']:
 q=copy.deepcopy(p);x=(.05+.5*((np.arange(4*4*3)*37)%256)/255).reshape(4,4,3)
 if name=='rescale_down':q.io.upscale_factor=.5
 if name=='rescale_up':q.io.upscale_factor=1.5
 if name=='input_acescg':q.io.input_color_space='ACEScg'
 if name=='input_oklch':q.io.input_gamut_compress=type(q.io.input_gamut_compress)(algorithm='oklch')
 if name=='jzazbz':q.io.output_gamut_compress=type(q.io.output_gamut_compress)(algorithm='jzazbz')
 if name=='oklch_p3':q.io.output_color_space='Display P3';q.io.output_gamut_compress=type(q.io.output_gamut_compress)(algorithm='oklch')
 if name=='oklrab_prophoto':q.io.output_color_space='ProPhoto RGB';q.io.output_gamut_compress=type(q.io.output_gamut_compress)(algorithm='oklrab')
 if name=='encoded_input':q.io.input_cctf_decoding=True
 if name in ['rec2020','display_p3','prophoto','acescg']:
  q.io.output_color_space={'rec2020':'ITU-R BT.2020','display_p3':'Display P3','prophoto':'ProPhoto RGB','acescg':'ACEScg'}[name];q.io.output_cctf_encoding=True
 if name=='crop':q.io.crop=True;q.io.crop_size=(.5,.5)
 if name=='uv':q.camera.filter_uv=(1.,450.,8.)
 if name=='ir':q.camera.filter_ir=(.7,620.,15.)
 if name=='spectral_blur':q.settings.spectral_gaussian_blur=3.
 if name=='adaptation_surface':q.settings.apply_hanatos2025_adaptation_surface=True
 if name=='spectral_luts':q.settings.use_enlarger_lut=True;q.settings.use_scanner_lut=True
 if name=='preview':q.settings.preview_mode=True;q.film_render.grain.active=True;q.scanner.unsharp_mask=(.7,.7)
 if name=='lut_mode':q.debug.lut_mode=True;q.camera.auto_exposure=True;q.film_render.grain.active=True
 if name=='unbounded':q.io.output_gamut_compress=type(q.io.output_gamut_compress)(algorithm='off');x=x*20
 if name=='sublayer_grain':q.film_render.grain.active=True
 if name=='crop_encoded_auto':q.io.crop=True;q.io.crop_size=(.5,.5);q.io.upscale_factor=1.5;q.io.input_cctf_decoding=True;q.camera.auto_exposure=True
 if name=='rgb_in_passthrough':q.taps.collect='rgb_in';q.io.crop=True;q.io.input_cctf_decoding=True;q.camera.auto_exposure=True
 if name=='rgb_pre_encoded':q.taps.collect='rgb_pre';q.io.crop=True;q.io.crop_size=(.5,.5);q.io.input_cctf_decoding=True;q.camera.auto_exposure=True
 if name=='inject_rgb_pre_encoded':q.taps.inject='rgb_pre';q.io.crop=True;q.io.crop_size=(.5,.5);q.io.upscale_factor=1.5;q.io.input_cctf_decoding=True;q.camera.auto_exposure=False
 data=asdict(q)
 for k in ['film','print']:data.pop(k)
 y=simulate(x.copy(),q)
 cases.append(dict(name=name,params=data,width=4,height=4,input=x.ravel().tolist(),output_width=y.shape[1],output_height=y.shape[0],expected=y.ravel().tolist(),tolerance=0.005 if name=='sublayer_grain' else 2e-6))
Path('crates/spektrafilm-core/tests/fixtures/python_parity.json').write_text(json.dumps(dict(reference_commit='28bf883e1672e884307edc75852549376e13644e',cases=cases),indent=2)+'\n')
print('wrote',len(cases),'fixtures')

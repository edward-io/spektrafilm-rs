"""Reference simulator strips with independently owned condition settings.

The old CalibrationTarget class shallow-copies nested dataclasses, so use
explicit deep copies to preserve its intended per-condition ramp semantics.
"""
import copy,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from spektrafilm.runtime.api import init_params,simulate
p=init_params('kodak_portra_400','kodak_portra_endura')
p.camera.auto_exposure=False;p.io.input_color_space='sRGB';p.io.input_cctf_decoding=False;p.io.output_cctf_encoding=False
p.film_render.grain.active=False;p.film_render.halation.active=False;p.film_render.dir_couplers.active=False;p.print_render.glare.active=False;p.scanner.unsharp_mask=(0.,0.)
x=(.05+.5*((np.arange(12*8*3)*37)%256)/255).reshape(8,12,3)
cases=[]
for name,values in [('negative_exposure',[-2.,0.,2.]),('print_exposure',[.5,1.,2.])]:
 for vertical in [False,True]:
  sections=[]
  for value in values:
   q=copy.deepcopy(p);q.io.crop=True;q.io.crop_center=(.5,.5);q.io.crop_size=(.5,.5)
   if name=='negative_exposure':q.camera.exposure_compensation_ev=value
   else:q.enlarger.print_exposure=value
   sections.append(simulate(x.copy(),q))
  y=np.concatenate(sections,axis=0 if vertical else 1)
  cases.append(dict(ramp=name,values=values,crop_center=[.5,.5],crop_size=[.5,.5],vertical=vertical,shape=list(y.shape),expected=y.ravel().tolist()))
params=asdict(p);params.pop('film');params.pop('print')
out=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/calibration.json'
out.write_text(json.dumps(dict(input=x.ravel().tolist(),shape=list(x.shape),params=params,cases=cases),indent=2)+'\n')

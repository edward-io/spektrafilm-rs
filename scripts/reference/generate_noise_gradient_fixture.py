"""Actual upstream OkLab noise-gradient inputs (independent Cmax and encoding)."""
import json
from pathlib import Path
from types import SimpleNamespace
from spektrafilm_lut_creator.qa.tests.picture_style import _build_oklab_noise_gradient_input
from spektrafilm_lut_creator.color_spaces import BakeFrame
cases=[]
for name in ['sRGB','ProPhoto RGB','Panasonic V-Log','Rec.2100 HLG']:
 frame=BakeFrame.from_spec(SimpleNamespace(input_color_space=name,output_color_space='sRGB',exposure_ev=0.))
 x,stats=_build_oklab_noise_gradient_input(frame,width=32,height=24,rim_scale=.96)
 cases.append(dict(space=name,input=x.reshape(-1,3).tolist(),width=32,height=24,stats=stats))
p=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/noise_gradient.json'
p.write_text(json.dumps(cases,separators=(',',':'))+'\n')

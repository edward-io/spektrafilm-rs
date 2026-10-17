"""Capture the unmodified source stress stimulus before its runtime call."""
import json
from pathlib import Path
from spektrafilm_lut_creator.qa.tests.picture_style import _build_gamut_edge_stress_panel
cases=[]
for input_space in ['sRGB','Panasonic V-Log']:
 for target in ['Rec.709','DCI-P3','Rec.2020']:
  class Echo:
   image=None
   def process(self,image):self.image=image.copy();return image
  echo=Echo()
  _,stats=_build_gamut_edge_stress_panel(target,input_space,'sRGB',echo,width=24,height=16)
  cases.append(dict(input_space=input_space,target=target,input=echo.image.reshape(-1,3).tolist(),stats=stats))
p=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/edge_stress.json'
p.write_text(json.dumps(cases,separators=(',',':'))+'\n')

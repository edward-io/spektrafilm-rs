"""Actual upstream daylight / Planckian QA stimuli across transport families."""
import json,math
from pathlib import Path
from spektrafilm_lut_creator.qa import patterns
from spektrafilm_lut_creator.color_spaces import input_gain,get
cases=[]
for space in ['sRGB','Display P3','ProPhoto RGB','Panasonic V-Log','Rec.2100 PQ','Rec.2100 HLG','ACEScc']:
 for legacy in [False,True]:
  exposure=math.log2(1/input_gain(space)) if legacy else 0.
  samples,cct=patterns.planckian_sweep(space,n=16,exposure_ev=exposure)
  ev,ramp,clip=patterns.dynamic_range_neutral_ramp(space,n=257,middle_gray_linear=get(space).midgray_linear)
  cases.append(dict(space=space,gain=1.0 if legacy else None,exposure_ev=exposure,samples=samples.tolist(),cct=cct.tolist(),dynamic=dict(stops=ev.tolist(),encoded=ramp.tolist(),clipped=clip.tolist())))
p=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/qa_patterns.json'
p.write_text(json.dumps(cases,indent=2)+'\n')

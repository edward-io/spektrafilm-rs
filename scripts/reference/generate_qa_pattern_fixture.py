"""Actual upstream daylight / Planckian QA stimuli across transport families."""
import json
from pathlib import Path
from spektrafilm_lut_creator.qa import patterns
from spektrafilm_lut_creator.color_spaces import native_stops_above_midgray,get
cases=[]
for space in ['sRGB','Display P3','ProPhoto RGB','Panasonic V-Log','Rec.2100 PQ','Rec.2100 HLG','ACEScc']:
 for legacy in [False,True]:
  stops=None if legacy else native_stops_above_midgray(space)
  samples,cct=patterns.planckian_sweep(space,n=16,stops_above_midgray=stops)
  ev,ramp,clip=patterns.dynamic_range_neutral_ramp(space,n=257,middle_gray_linear=get(space).midgray_linear)
  cases.append(dict(space=space,gain=1.0 if legacy else None,stops_above_midgray=stops,samples=samples.tolist(),cct=cct.tolist(),dynamic=dict(stops=ev.tolist(),encoded=ramp.tolist(),clipped=clip.tolist())))
p=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/qa_patterns.json'
p.write_text(json.dumps(cases,indent=2)+'\n')

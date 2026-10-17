"""Real resizing-service metering order, including encoded-space anti-aliasing."""
import json
from pathlib import Path
import numpy as np
from spektrafilm.runtime.services.resize import ResizingService
from spektrafilm.utils.autoexposure import measure_autoexposure_ev
root=Path(__file__).resolve().parents[2]
cases=[]
for w,h,pattern in [(512,101,'ramp'),(511,129,'checker')]:
 x,y=np.meshgrid(np.arange(w),np.arange(h))
 if pattern=='ramp': image=np.stack([((x*7+y*13+c*29)%100)/100 for c in range(3)],axis=-1)
 else:
  v=np.where((x+y)%2==0,.95,.02);image=np.stack([v,.8*v+.04,.7*v+.1],axis=-1)
 image=image.astype('float32').astype(float)
 thumbnail=ResizingService(None,35.).small_preview(image)
 for space,encoded in [('sRGB',False),('sRGB',True),('Display P3',True),('ACEScg',False)]:
  expected={method:float(measure_autoexposure_ev(thumbnail,space,encoded,method)) for method in ['average','median','center_weighted','partial','matrix','multi_zone','highlight_weighted']}
  cases.append({'size':[w,h],'pattern':pattern,'color_space':space,'encoded':encoded,'expected':expected,'thumbnail_size':[thumbnail.shape[1],thumbnail.shape[0]]})
out=root/'crates/spektrafilm-core/tests/fixtures/metering.json';out.write_text(json.dumps(cases));print(out)

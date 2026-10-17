"""Independent skimage preview-resize ground truth, including edge boundaries."""
import json,numpy as np
from pathlib import Path
from spektrafilm.utils.preview import resize_for_preview
cases=[]
for w,h,limit in [(17,11,7),(16,9,3),(7,7,1),(31,7,5)]:
 x=(((np.arange(w*h*3)*37)%701)/699.-.1).reshape(h,w,3).astype(np.float32)
 y=resize_for_preview(x,limit)
 cases.append(dict(width=w,height=h,limit=limit,input=x.ravel().tolist(),output_width=y.shape[1],output_height=y.shape[0],expected=y.ravel().tolist()))
Path('crates/spektrafilm-core/tests/fixtures/preview.json').write_text(json.dumps(cases,indent=2)+'\n')

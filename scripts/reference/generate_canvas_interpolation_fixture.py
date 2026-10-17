"""Sampling oracle from Vispy's actual filter weight classes.
Pass its vispy/glsl/build_spatial_filters.py path (BSD-2-Clause).
"""
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import numpy as np
root=pathlib.Path(__file__).resolve().parents[2]
source=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('vispy_spatial_filters',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fixture=json.loads((root/'crates/spektrafilm-gui/tests/fixtures/development.json').read_text())
image=np.array(fixture['rgba'],dtype=np.float64).reshape(8,16,4)[:,:,:3]/255.
uv=np.array([.013,.025,.927,.973],dtype=np.float32).astype(float)
frames=[]
filters=[None,module.Linear(),module.Cubic(),module.Spline16(),module.Spline36(),module.Lanczos(),module.Blackman()]
for mode,filter in enumerate(filters):
 out=np.zeros_like(image)
 for y in range(8):
  for x in range(16):
   u=uv[0]+(uv[2]-uv[0])*(x+.5)/16
   v=uv[1]+(uv[3]-uv[1])*(y+.5)/8
   px=u*16-.5;py=v*8-.5
   if mode==0:
    out[y,x]=image[max(0,min(7,math.floor(py+.5))),max(0,min(15,math.floor(px+.5)))];continue
   radius=filter.radius;weights=[];values=[]
   for dy in range(1-radius,radius+1):
    yy=math.floor(py)+dy
    wy=filter.weight(abs(py-yy)) if abs(py-yy)<radius else 0
    for dx in range(1-radius,radius+1):
     xx=math.floor(px)+dx
     wx=filter.weight(abs(px-xx)) if abs(px-xx)<radius else 0
     weights.append(wx*wy);values.append(image[max(0,min(7,yy)),max(0,min(15,xx))])
   out[y,x]=np.einsum('i,ij->j',weights,values)/sum(weights)
 frames.append({'mode':mode,'uv':uv.tolist(),'rgb':np.clip(out,0,1).reshape(-1).tolist()})
out=root/'crates/spektrafilm-gui/tests/fixtures/interpolation.json'
out.write_text(json.dumps({'vispy_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':frames}))
print(out)

"""Independent small / broad Gaussian edge oracle from the Python runtime."""
import json
from pathlib import Path
import numpy as np
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
root=Path(__file__).resolve().parents[2]
w,h=257,193;x,y=np.meshgrid(np.arange(w),np.arange(h));v=np.where((x//5+y//7)%2==0,.9,.01)+.0002*x+.0001*y
v[(x==0)&(y==h-1)]=25.
image=(v[:,:,None]*np.array([1.,.4,.1])[None,None,:]).astype('float32').astype(float)
points=[[x,y] for y in [0,1,2,11,96,191,192] for x in [0,1,2,17,128,254,256]]
cases=[]
for sigma in [.1,.7,float(np.float32(5/6)),float(np.float32(7/6)),1.7,2.99,3.,5.99,10.,32.,100.]:
 out=fast_gaussian_filter(image,sigma)
 cases.append({'sigma':sigma,'expected':[out[y,x].tolist() for x,y in points]})
out=root/'crates/spektrafilm-math/tests/fixtures/gaussian.json';out.write_text(json.dumps({'size':[w,h],'points':points,'cases':cases}));print(out)

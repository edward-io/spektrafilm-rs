"""Independent colour-science saving-transport conversions."""
from pathlib import Path
import colour,json,numpy as np
x=np.array([[-.05,.18,1.5],[.5,.3,.02]],dtype=np.float64)
rows=[]
for source,source_encoded,destination,destination_encoded in [('Display P3',True,'sRGB',True),('ProPhoto RGB',False,'ACEScg',False),('sRGB',True,'ProPhoto RGB',True),('sRGB',True,'sRGB',False)]:
 y=colour.RGB_to_RGB(x,source,destination,chromatic_adaptation_transform='CAT02',apply_cctf_decoding=source_encoded,apply_cctf_encoding=destination_encoded)
 rows.append(dict(source=source,source_encoded=source_encoded,destination=destination,destination_encoded=destination_encoded,input=x.ravel().tolist(),expected=y.ravel().tolist()))
Path('crates/spektrafilm-io/tests/fixtures/saving.json').write_text(json.dumps(rows,indent=2)+'\n')

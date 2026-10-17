"""Small synthetic DNG plus LibRaw/rawpy reference outputs; contains no personal image data."""
from pathlib import Path
import json,numpy as np,tifffile,rawpy
from spektrafilm.utils.raw_file_processor import load_and_process_raw_file
root=Path('crates/spektrafilm-io/tests/fixtures')
y,x=np.mgrid[:32,:32];raw=(128+(x*37+y*59)%3000).astype(np.uint16)
tags=[(50706,'B',4,(1,4,0,0),False),(50707,'B',4,(1,1,0,0),False),(50708,'s',0,'Spektrafilm synthetic DNG',False),(33421,'H',2,(2,2),False),(33422,'B',4,(0,1,1,2),False),(50717,'I',1,4095,False),(50714,'I',1,64,False),(50721,'2i',9,(1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1),False),(50778,'H',1,21,False),(50728,'2I',3,(1,2,1,1,1,3),False)]
path=root/'synthetic.dng';tifffile.imwrite(path,raw,photometric=32803,metadata=None,extratags=tags)
cases=[]
for mode in ['as_shot','daylight','tungsten','custom']:
 image=load_and_process_raw_file(path,white_balance=mode,temperature=4500.,tint=1.1,output_colorspace='ACES2065-1')
 cases.append(dict(mode=mode,width=image.shape[1],height=image.shape[0],expected=image.ravel().tolist()))
(root/'libraw.json').write_text(json.dumps(cases,indent=2))
print('wrote RAW reference fixtures')

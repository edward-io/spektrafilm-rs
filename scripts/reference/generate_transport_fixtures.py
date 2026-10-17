"""Generate colour-science transfer and Python bundle golden outputs."""
import json
from pathlib import Path
import colour,numpy as np
from spektrafilm_lut_creator import color_spaces as spaces
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.builders import BundleBuilder
samples=[0.,.001,.01,.18,.5,1.]
def record(name,encode,decode):
 return dict(name=name,input=samples,encode=[float(encode(x)) for x in samples],decode=[float(decode(x)) for x in samples])
core=Path('crates/spektrafilm-core/tests/fixtures')
rows=[record(e.name,lambda x:spaces.encode_cctf(x,e.name),lambda x:spaces.decode_cctf(x,e.name)) for e in spaces._REGISTRY.values()]
(core/'transfers.json').write_text(json.dumps(rows,indent=2)+'\n')
rows=[record(name,cs.cctf_encoding,cs.cctf_decoding) for name,cs in colour.RGB_COLOURSPACES.items()]
Path('crates/spektrafilm-math/tests/fixtures/color_curves.json').write_text(json.dumps(rows,indent=2)+'\n')
rows=[]
for source,destination in [('sRGB','sRGB'),('Panasonic V-Log','Display P3'),('Rec.2100 PQ','Rec.2100 PQ'),('Rec.2100 HLG','sRGB')]:
 spec=BundleSpec(film_profile='kodak_portra_400',print_profiles=('kodak_portra_endura',),input_color_space=source,output_color_space=destination,resolution=5)
 bundle=BundleBuilder(spec).build()
 rows.append(dict(input=source,output=destination,expected=np.asarray(bundle.luts[0][1].table).ravel().tolist()))
(core/'luts.json').write_text(json.dumps(rows,indent=2)+'\n')

"""Generate colour-science transfer and Python bundle golden outputs."""
import json
from dataclasses import asdict,replace
from pathlib import Path
import colour,numpy as np
from spektrafilm_lut_creator import color_spaces as spaces
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm.runtime.api import init_params
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
 bundle=BundleBuilder(spec,base_params=init_params('kodak_portra_400','kodak_portra_endura')).build()
 rows.append(dict(input=source,output=destination,expected=np.asarray(bundle.luts[0][1].table).ravel().tolist()))
params=init_params('kodak_portra_400','kodak_portra_endura')
params.film_render.chemistry=replace(params.film_render.chemistry,gamma_factor=1.2,developer_exhaustion=.2)
params.film_render.base.cyan=1.12
params.camera.color_filter='hoya_x1'
params.settings.rgb_to_raw_method='arctic2026beta04'
spec=BundleSpec(film_profile='kodak_portra_400',print_profiles=('kodak_portra_endura',),input_color_space='sRGB',output_color_space='Display P3',resolution=5,exposure_ev=.5)
bundle=BundleBuilder(spec,base_params=params).build()
settings=asdict(params);settings.pop('film');settings.pop('print')
rows.append(dict(input='sRGB',output='Display P3',exposure_ev=.5,params=settings,expected=np.asarray(bundle.luts[0][1].table).ravel().tolist()))
(core/'luts.json').write_text(json.dumps(rows,indent=2)+'\n')

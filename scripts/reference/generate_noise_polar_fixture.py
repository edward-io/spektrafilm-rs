"""Actual source polar samples and complete noise-sensitivity summaries."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from spektrafilm_lut_creator.qa.tests.picture_style import _polar_oklch_input_samples,noise_sensitivity
from spektrafilm_lut_creator.color_spaces import BakeFrame
root=Path(__file__).resolve().parents[2]
f=json.loads((root/'crates/spektrafilm-core/tests/fixtures/qa.json').read_text());n=f['n']
table=np.array(f['table']).reshape(n,n,n,3)
figures=Path('/tmp/spektrafilm-noise-oracle');figures.mkdir(exist_ok=True)
cases=[]
for name in ['sRGB','ProPhoto RGB','Panasonic V-Log','Rec.2100 HLG']:
 frame=BakeFrame.from_spec(NS(input_color_space=name,output_color_space='sRGB',exposure_ev=0.))
 samples=_polar_oklch_input_samples(frame,L=.5646225971435698,chroma_rings=(.07,.14,.21),n_hues=16)
 ctx=NS(frame=frame,spec=NS(input_color_space=name,output_color_space='sRGB'),lut=NS(table=table,resolution=n),figures_dir=figures,bundle=NS(meta=NS(provenance=NS(spektrafilm_version='0.3.4'))))
 result=noise_sensitivity(ctx)
 cases.append(dict(space=name,samples={k:v.tolist() for k,v in samples.items()},summary=result.summary))
(root/'crates/spektrafilm-core/tests/fixtures/noise_polar.json').write_text(json.dumps(dict(n=n,table=f['table'],cases=cases),indent=2)+'\n')

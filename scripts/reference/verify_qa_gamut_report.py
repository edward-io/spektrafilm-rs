"""Run original gamut QA on a native one-LUT bundle and compare all reported measurements."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import matplotlib.pyplot as plt
from spektrafilm_lut_creator.formats import get_format
from spektrafilm_lut_creator.grid import cube_grid
from spektrafilm_lut_creator.qa.tests import lut_fidelity as fidelity,model_diagnostic as model,gamut_compression as gamut
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec,OutputGamutCompressSpec
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);args=p.parse_args()
bundle=args.bundle;meta=json.loads((bundle/'bundle.json').read_text());q=json.loads((bundle/'qa.json').read_text())
lut=get_format('cube').read(bundle/meta['luts'][0]['file'])
cs=meta['color_spaces']
ctx=NS(spec=NS(input_color_space=cs['input']['name'],output_color_space=cs['output']['name'],film_profile=meta['film'],print_profiles=[meta['paper']],input_gamut_compress=InputGamutCompressSpec(),output_gamut_compress=OutputGamutCompressSpec()),
 bundle=NS(meta=NS(stocks=NS(prints=[meta['paper']]))),print_index=0,print_name=meta['paper'],lut=lut,
 grid_input=cube_grid(meta['resolution']),grid_output=lut.table.reshape(-1,3))
results=[]
for module,fn in [(fidelity,fidelity.output_gamut_compression),(model,model.spectral_locus_envelope),(gamut,gamut.input_gamut_compression_preview),(gamut,gamut.input_gamut_compression_smoothness)]:
 module._save=lambda ctx,fig,name:(plt.close(fig) or Path('unused'))
 expected=fn(ctx);actual=next(c for c in q['scientific']['cases'] if c['name']==expected.name)['summary']
 errors={}
 for key,value in expected.summary.items():
  got=actual[key]
  if isinstance(value,(int,float)):
   delta=abs(value-got);errors[key]=delta
   # The runtime's f32 input capture and f64 integration differ slightly; use
   # a strict relative/absolute gate for values, exact integer populations.
   tolerance=0 if isinstance(value,int) else 2e-5*(1+abs(value))
   assert delta<=tolerance,(expected.name,key,got,value,delta,tolerance)
  else:assert got==value,(expected.name,key,got,value)
 results.append(dict(name=expected.name,passed=True,absolute_differences=errors))
 print(expected.name,'matches original Python probe',flush=True)
(bundle/'reference-gamut-check.json').write_text(json.dumps(results,indent=2)+'\n')

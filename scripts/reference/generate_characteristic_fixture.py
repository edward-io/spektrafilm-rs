"""Capture outputs of the unmodified upstream multi-pin characteristic probe."""
import json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from spektrafilm_lut_creator.qa.tests.model_diagnostic import characteristic_curve
root=Path(__file__).resolve().parents[2]
f=json.loads((root/'crates/spektrafilm-core/tests/fixtures/qa.json').read_text());n=f['n']
table=np.array(f['table']).reshape(n,n,n,3)
figures=Path('/tmp/spektrafilm-characteristic-oracle');figures.mkdir(exist_ok=True)
ctx=NS(lut=NS(table=table,resolution=n),figures_dir=figures,bundle=NS(meta=NS(provenance=NS(spektrafilm_version='0.3.4'))))
capture={}
def trace(frame,event,arg):
 if event=='return' and frame.f_code.co_name=='characteristic_curve':
  v=frame.f_locals
  capture.update(pin_codes=list(v['pin_codes']),axis=v['sweep_x'].tolist(),neutral=v['neutral_samples'].tolist(),sweeps=[[x[1].tolist() for x in v[key]] for key in ['r_sweep_data','g_sweep_data','b_sweep_data']])
sys.setprofile(trace)
try:result=characteristic_curve(ctx)
finally:sys.setprofile(None)
capture.update(n=n,table=f['table'],summary=result.summary)
(root/'crates/spektrafilm-core/tests/fixtures/characteristic.json').write_text(json.dumps(capture,indent=2)+'\n')

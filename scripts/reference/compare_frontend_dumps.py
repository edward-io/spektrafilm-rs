"""Check the 0.1%-of-full-scale gate across independent frontend builds."""
import json
import pathlib
import sys
import numpy as np
left,right=map(pathlib.Path,sys.argv[1:3])
a=json.loads((left/'manifest.json').read_text());b=json.loads((right/'manifest.json').read_text())
assert a['precision']=='f64' and a['backend']=='cpu'
assert b['precision']=='f32' and b['backend']=='gpu'
assert len(a['cases'])==len(b['cases'])
results=[]
for x,y in zip(a['cases'],b['cases']):
 assert (x['case'],x['width'],x['height'])==(y['case'],y['width'],y['height'])
 reference=np.fromfile(left/x['file'],dtype='<f8');actual=np.fromfile(right/y['file'],dtype='<f8')
 assert reference.shape==actual.shape and np.isfinite(reference).all() and np.isfinite(actual).all()
 error=np.abs(reference-actual)
 row={'case':x['case'],'max_absolute':float(error.max()),'p99':float(np.quantile(error,.99)),'rms':float(np.sqrt(np.mean(error**2))),'over_0_1_percent':int((error>.001).sum())}
 # Relative error is informative above black, not the full-scale pass gate.
 mask=np.abs(reference)>.01
 if mask.any(): row['relative_p99_above_0_01']=float(np.quantile(error[mask]/np.abs(reference[mask]),.99))
 results.append(row)
 print(f"{row['case']}: max {row['max_absolute']:.7f}, p99 {row['p99']:.7f}, failures {row['over_0_1_percent']}")
if len(sys.argv)>3: pathlib.Path(sys.argv[3]).write_text(json.dumps(results,indent=2)+'\n')
assert all(r['max_absolute']<=.001 for r in results),'frontend difference exceeds 0.1% full scale'

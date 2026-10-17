"""Exact NumPy hull subsamples, covering Floyd and tail-shuffle choice paths."""
import json,numpy as np
from pathlib import Path
cases=[]
for population in [5**3,17**3,33**3,65**3,129**3]:
 indices=np.random.default_rng(0).choice(population,size=min(population,8000),replace=False)
 cases.append(dict(population=population,indices=indices.tolist()))
p=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/hull_sampling.json'
p.write_text(json.dumps(cases,separators=(',',':'))+'\n')

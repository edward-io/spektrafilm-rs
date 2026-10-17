"""Independent SciPy oracle for the profile-analysis helpers."""
import json
from pathlib import Path
import numpy as np
from scipy.special import erf
from spektrafilm.utils.measure import measure_gamma,measure_slopes_at_exposure,measure_density_min
cases=[]
x=np.linspace(-3,3,65)
for positive in [False,True]:
 for offset in [0.,.3]:
  c=np.array([1.1,1.2,1.]);e0=np.array([-.1,.2,0.])+offset
  y=np.array([2.4,2.2,2.7])*(1+erf(((-1 if positive else 1)*c)*(x[:,None]-e0)))/2+np.array([.03,.05,.04])
  cases.append(dict(log_exposure=x.tolist(),density_curves=y.tolist(),positive=positive,
    gamma=measure_gamma(x,y).tolist(),slopes=measure_slopes_at_exposure(x,y).tolist(),
    density_min=measure_density_min(x,y,'positive' if positive else 'negative').tolist()))
out=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/measurement.json'
out.write_text(json.dumps({'source':'unmodified spektrafilm.utils.measure; SciPy 1.17.1','cases':cases},indent=2)+'\n')

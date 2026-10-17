import json
from pathlib import Path
import numpy as np
from spektrafilm.model.parametric import parametric_density_curves_model
root=Path(__file__).resolve().parents[2]
p={'log_exposure':np.linspace(-5,7,121).tolist(),'gamma':[0.62,0.73,0.81],'log_exposure_0':[-.9,-.7,-.65],'density_max':[2.4,2.8,3.1],'toe_size':[.2,.4,.3],'shoulder_size':[.5,.3,.6]}
p['density']=parametric_density_curves_model(np.array(p['log_exposure']),p['gamma'],p['log_exposure_0'],p['density_max'],p['toe_size'],p['shoulder_size']).tolist()
out=root/'crates/spektrafilm-model/tests/fixtures/parametric.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(p));print(out)

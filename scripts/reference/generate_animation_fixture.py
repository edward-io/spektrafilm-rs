"""Regenerate native canvas animation oracle from the adjacent Python GUI."""
import importlib.util
import json
import pathlib
import sys
import numpy as np
root=pathlib.Path(__file__).resolve().parents[3]
path=root/'src/spektrafilm_gui/polaroid_animation.py'
spec=importlib.util.spec_from_file_location('polaroid_animation',path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
rng=np.random.default_rng(581)
rgba=rng.integers(0,256,size=(8,16,4),dtype=np.uint8);rgba[:,:,3]=255
rgba[0,:4,:3]=np.array([[0,0,0],[255,255,255],[127,32,250],[255,0,10]],dtype=np.uint8)
rgb=rgba[:,:,:3].astype(np.float32)/255.
state=module.prepare_polaroid_state(rgb)
frames=[{'t':t,'coefficients':module.layer_coefficients(t).tolist(),'rgb':module.render_polaroid_frame(state,t).reshape(-1).tolist()} for t in [0.,0.15,0.4,0.75,1.]]
out=root/'spektrafilm-rs/crates/spektrafilm-gui/tests/fixtures/development.json'
out.write_text(json.dumps({'size':[16,8],'rgba':rgba.reshape(-1).tolist(),'frames':frames}))
print(out)

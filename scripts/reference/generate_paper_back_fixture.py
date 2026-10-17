import importlib.util
import json
import pathlib
root=pathlib.Path(__file__).resolve().parents[3]
path=root/'src/spektrafilm_gui/virtual_photo_paper_back.py'
spec=importlib.util.spec_from_file_location('paper_back',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
image=module.virtual_photo_paper_back(canvas_size=(192,128))
rgba=__import__('numpy').full((128,192,4),255,dtype='uint8');rgba[:,:,:3]=image
out=root/'spektrafilm-rs/crates/spektrafilm-gui/tests/fixtures/paper_back.json'
out.write_text(json.dumps({'size':[192,128],'rgba':rgba.reshape(-1).tolist()}))
print(out)

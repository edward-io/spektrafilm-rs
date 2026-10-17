"""Copy the reference's verified OCIO builtin catalog (development only)."""
import json
from pathlib import Path
from spektrafilm_lut_creator.ocio_emit import _COLORSPACE_BUILTIN
Path('crates/spektrafilm-core/src/ocio_builtins.json').write_text(json.dumps(_COLORSPACE_BUILTIN,indent=2)+'\n')

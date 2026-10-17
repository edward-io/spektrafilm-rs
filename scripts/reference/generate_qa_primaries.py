"""Exact colour-science primary chromaticities for the native LUT transports."""
import re,colour
from pathlib import Path
root=Path(__file__).resolve().parents[2]
names=sorted(set(re.findall(r'primaries: "([^"]+)"',(root/'crates/spektrafilm-core/src/lut_spaces.rs').read_text())))
lines=['//! Generated CIE primary chromaticities from colour-science 0.4.6.','pub const PRIMARIES: &[(&str,[[f64;2];3])] = &[']
for name in names:
 p=colour.RGB_COLOURSPACES[name].primaries
 lines.append('('+repr(name).replace("'",'"')+',['+','.join('['+','.join(repr(float(x)) for x in row)+']' for row in p)+']),')
lines.append('];')
(root/'crates/spektrafilm-core/src/qa_primaries.rs').write_text('\n'.join(lines)+'\n')

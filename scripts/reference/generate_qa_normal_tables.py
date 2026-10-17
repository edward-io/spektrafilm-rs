"""Import NumPy's numerical normal-Ziggurat tables and record its license."""
import re,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[2]
url='https://raw.githubusercontent.com/numpy/numpy/v2.4.6/numpy/random/src/distributions/ziggurat_constants.h'
source=urllib.request.urlopen(url).read().decode()
result=['//! Numerical Ziggurat tables from NumPy 2.4.6 (BSD-3-Clause).',
 '//! See data/licenses/numpy.txt and scripts/reference/generate_qa_normal_tables.py.']
for name,kind in [('ki_double','u64'),('wi_double','f64'),('fi_double','f64')]:
 match=re.search(r'const\s+\w+\s+'+name+r'\s*\[\s*\]\s*=\s*\{(.*?)\};',source,re.S)
 assert match,name
 tokens=[v.strip() for v in match[1].split(',') if v.strip()]
 assert len(tokens)==256
 values=[str(int(v.rstrip('uUlL'),0)) if kind=='u64' else repr(float(v)) for v in tokens]
 result.append(f'pub const {name.upper()}: [{kind};256] = [\n'+',\n'.join(values)+'\n];')
for name in ['ziggurat_nor_r','ziggurat_nor_inv_r']:
 match=re.search(r'const\s+double\s+'+name+r'\s*=\s*([^;]+);',source)
 assert match,name
 result.append(f'pub const {name.upper()}: f64 = {float(match[1])!r};')
(root/'crates/spektrafilm-core/src/qa_normal_tables.rs').write_text('\n'.join(result)+'\n')
license=urllib.request.urlopen('https://raw.githubusercontent.com/numpy/numpy/v2.4.6/LICENSE.txt').read()
(root/'data/licenses/numpy.txt').write_bytes(license)

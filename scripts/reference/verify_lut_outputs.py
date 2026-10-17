"""Read native LUT files through the Python reference and execute OCIO configs."""
import json,subprocess,tempfile,math,zipfile
from pathlib import Path
import numpy as np
import PyOpenColorIO as ocio
from spektrafilm_lut_creator.formats import get_format as get
import argparse
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--cli',type=Path,default=root/'target/debug/spektrafilm-f64');cli=parser.parse_args().cli.resolve()
def run(*args):
 p=subprocess.run([str(cli),*map(str,args)],text=True,capture_output=True)
 if p.returncode: raise RuntimeError(p.stderr+'\n'+p.stdout)
with tempfile.TemporaryDirectory(prefix='spektrafilm-lut-check-') as d:
 d=Path(d)
 base=['export-lut','--film','kodak_portra_400','--paper','kodak_portra_endura','--data-dir',root/'data']
 for fmt,ext,n in [('cube','cube',5),('3dl','3dl',5),('hald_png','png',4),('lumix','cube',17)]:
  path=d/(fmt+'.'+ext)
  run(*base,'--size',n,'--format',fmt,'--input-space','vlog' if fmt=='lumix' else 'sRGB','-o',path)
  lut=get(fmt).read(path)
  assert lut.table.shape==(n,n,n,3),(fmt,lut.table.shape)
  assert np.isfinite(lut.table).all()
  if fmt=='lumix':assert '#LUMIXPHOTOSTYLE VLOG'==path.read_text().splitlines()[1]
  print(fmt, n, 'read by Python format plugin')
 a=get('cube').read(d/'cube.cube').table;b=get('3dl').read(d/'3dl.3dl').table
 assert np.max(np.abs(a-b))<=.5001/1023
 for topology in ['1lut','2lut','3lut','4lut']:
  path=d/topology
  run(*base,'--size',5,'--topology',topology,'--include-combinations','-o',path)
  config=ocio.Config.CreateFromFile(str(path/'config.ocio'));config.validate()
  film='spektrafilm_kodak_portra_400_kodak_portra_endura'
  processor=config.getProcessor('sRGB',film).getDefaultCPUProcessor()
  assert all(math.isfinite(v) for v in processor.applyRGB([.18,.3,.6]))
  print(topology,'OCIO config validated and executed')
 spec=d/'spec.json'
 spec.write_text(json.dumps(dict(film_profile='kodak_portra_400',print_profiles=['kodak_portra_endura','fujifilm_crystal_archive_typeii'],input_color_space='vlog',output_color_space='srgb',topology='4lut',resolution=5,container='zip',qa=True,ocio_config=True,include_combinations=True)))
 output=d/'multi';run('build-luts','--from',spec,'-o',output,'--data-dir',root/'data')
 with zipfile.ZipFile(output.with_suffix('.zip')) as z:
  assert z.testzip() is None
  assert sum(n.endswith('qa.json') for n in z.namelist())==2
  assert 'SPEKTRAFILM_LICENSE.txt' in z.namelist()
 for paper in ['kodak_portra_endura','fujifilm_crystal_archive_typeii']:
  ocio.Config.CreateFromFile(str(output/paper/'config.ocio')).validate()
  qa=json.loads((output/paper/'qa.json').read_text());assert qa['samples']==50_000;assert all(math.isfinite(v) for v in qa.values() if isinstance(v,float)); scientific=qa['scientific'];assert len(scientific['cases'])>=15
  for case in scientific['cases']: assert (output/paper/case['figure']).is_file()
  assert math.isfinite(scientific['trilinear_de2000']['max'])
  print(paper,'native QA',len(scientific['cases']),'cases, max encoded error',qa['max_absolute_error'])
 print('PASS: native formats, all topologies, combinations, OCIO, multi-print ZIP and fidelity reports')

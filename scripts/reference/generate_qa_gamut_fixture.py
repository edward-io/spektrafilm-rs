"""Capture original gamut QA summaries/projections; no native implementation calls."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as Polygon
from spektrafilm_lut_creator.qa.tests import gamut_compression as g, model_diagnostic as md
from spektrafilm_lut_creator.grid import cube_grid
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec,spectral_locus_xy
from spektrafilm_lut_creator.color_spaces import get
import colour
cases=[]
with tempfile.TemporaryDirectory() as tmp:
 for name,algorithm,active in [('sRGB','xy',True),('Panasonic V-Log','xy',True),('ACEScg','xy',False),('sRGB','oklch',True)]:
  grid=cube_grid(5).reshape(-1,3)
  spec=NS(input_color_space=name,output_color_space='sRGB',film_profile='kodak_portra_400',input_gamut_compress=InputGamutCompressSpec(active=active,algorithm=algorithm))
  ctx=NS(spec=spec,grid_input=grid,figures_dir=Path(tmp),print_name='fixture',print_index=0)
  # Saving is unrelated to the numerical probes.
  old=g._save;g._save=lambda ctx,fig,name:(plt.close(fig) or Path(tmp)/f'{name}.png')
  preview=g.input_gamut_compression_preview(ctx).summary
  smooth=g.input_gamut_compression_smoothness(ctx).summary
  xy,b=g._cube_xy_in_film_frame(ctx,'D55')
  g._save=old
  locus=spectral_locus_xy();inside=Polygon(locus).contains_points(xy)
  cases.append(dict(input=name,algorithm=algorithm,active=active,grid=grid.tolist(),xy=xy.tolist(),b=b.tolist(),inside=inside.tolist(),preview=preview,smooth=smooth))
out=Path(__file__).resolve().parents[2]/'crates/spektrafilm-core/tests/fixtures/qa_gamut.json'
out.write_text(json.dumps(cases,separators=(',',':'))+'\n')

# Full output-cube summaries, evaluated by the actual source QA function.
envelopes=[]
for name in ['sRGB','DCI-P3','Rec.2020']:
 grid=cube_grid(5).reshape(5,5,5,3)
 table=(0.03+0.8*grid**1.3+0.04*np.roll(grid,1,axis=-1)).astype(np.float32)
 ctx=NS(spec=NS(input_color_space="sRGB",output_color_space=name,film_profile="fixture"),print_name="fixture",lut=NS(table=table))
 old=md._save;md._save=lambda ctx,fig,name:(plt.close(fig) or Path('unused'))
 summary=md.spectral_locus_envelope(ctx).summary
 md._save=old
 envelopes.append(dict(output=name,n=5,table=table.reshape(-1,3).tolist(),summary=summary))
(out.parent/'qa_envelope.json').write_text(json.dumps(envelopes,separators=(',',':'))+'\n')

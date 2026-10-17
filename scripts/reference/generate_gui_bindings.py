"""Import paths for Python v0.3.4 GUI state files (development only)."""
from pathlib import Path
import json
from spektrafilm_gui import param_manifest as m
bindings={}
for section,group in [('input_image',m.INPUT_IMAGE_FIELDS),('special',m.SPECIAL_FIELDS),('simulation',m.SIMULATION_FIELDS)]:
 for spec in group:
  path=spec.path
  if path=='selection.film_stock':dst='film'
  elif path=='selection.print_paper':dst='paper'
  elif path=='workflow.saving_color_space':dst='output.color_space'
  elif path=='workflow.saving_cctf_encoding':dst='output.encoded'
  elif path=='film_channel_swap':dst='params.film_render.channel_order'
  elif path=='print_channel_swap':dst='params.print_render.channel_order'
  elif path=='workflow.auto_preview':dst='auto_preview'
  elif path.startswith('workflow.'):continue
  else:dst='params.'+path
  bindings[section+'.'+spec.leaf]=dst
  bindings[section+'.'+path]=dst
for spec in m.DISPLAY_PANEL_FIELDS:
 if spec.path=='settings.preview_max_size':dst='params.settings.preview_max_size'
 elif spec.path in ['gray_18_canvas','white_padding','output_interpolation']:dst='display.'+spec.path
 else:continue
 for section in ['display','gui_only.display']:
  bindings[section+'.'+spec.leaf]=dst
  bindings[section+'.'+spec.path]=dst
Path('crates/spektrafilm-gui/src/python_gui_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')

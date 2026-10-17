"""Scientific metric ground truth from the original Python QA functions."""
import json,numpy as np,colour
from pathlib import Path
from spektrafilm_lut_creator.qa import metrics,evaluators
from spektrafilm_lut_creator.color_spaces import to_xyz_qa
rgb=np.array([[.01,.1,.8],[.18,.18,.18],[.7,.2,.05],[.4,.8,.3]])
other=np.clip(rgb+np.array([[.001,-.003,.002],[.005,0.,-.004],[-.01,.002,.003],[.001,-.005,0.]]),0,1)
colors=[]
for space in ['sRGB','Display P3','ProPhoto RGB','Rec.2100 PQ','Rec.2100 HLG']:
 colors.append(dict(space=space,a=rgb.tolist(),b=other.tolist(),de2000=metrics.delta_e_2000(rgb,other,output_color_space=space).tolist(),itp=metrics.delta_itp(rgb,other,output_color_space=space).tolist(),oklab=metrics.oklab_delta(rgb,other,output_color_space=space).tolist()))
x=np.linspace(0,1,5);b,g,r=np.meshgrid(x,x,x,indexing='ij');grid=np.stack([r,g,b],axis=-1)
table=np.clip(grid**1.3+np.stack([g*.07,-r*.03,b*.02],axis=-1),0,1)
samples=np.array([[.1,.2,.3],[.7,.2,.5],[.99,.78,.66],[0,0,0],[1,1,1]])
stop=np.linspace(-8,4,65);y=np.clip(.18*2**stop,.005,.95);clipped=stop>2.5
out=dict(colors=colors,table=table.ravel().tolist(),n=5,samples=samples.tolist(),trilinear=evaluators.apply_trilinear(table,samples).tolist(),tetrahedral=evaluators.apply_tetrahedral(table,samples).tolist(),monotonicity=metrics.monotonicity_violations(table),condition=metrics.summary_stats(metrics.local_jacobian_log_cond(table)),tv=metrics.total_variation(table),fft=metrics.axial_fft_highband_ratio(table),surface=metrics.gamut_self_intersection_score(table),hull=metrics.gamut_hull_volume_ratio(grid.reshape(-1,3),table.reshape(-1,3),'sRGB'),hue=metrics.hue_rotation_per_band(colour.XYZ_to_Oklab(to_xyz_qa(grid.reshape(-1,3),'sRGB')),colour.XYZ_to_Oklab(to_xyz_qa(table.reshape(-1,3),'sRGB'))),stops=stop.tolist(),y=y.tolist(),clipped=clipped.tolist(),dynamic=metrics.dynamic_range_stats(stop,y,clipped))
noise=metrics.noise_sensitivity_field(table,samples,in_cs='sRGB',out_cs='sRGB')
out['noise']={key:noise[key].tolist() for key in ['jacobian','cov_oklab','sigma1','anisotropy','sigma_L','sigma_ab']}
Path('crates/spektrafilm-core/tests/fixtures/qa.json').write_text(json.dumps(out,indent=2)+'\n')

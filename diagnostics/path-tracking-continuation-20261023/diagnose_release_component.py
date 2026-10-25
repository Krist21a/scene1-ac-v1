"""Read retained RGB-D groups; inspect geometric center and point colors only."""
import json,sys
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-project';sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
from scene1.retreat_projection import FingerSeparation
scope=dict(__name__='rgbd_component_diagnostic')
source=(policy/'scene1/rgbd.py').read_text()
source=source.replace('"position_base_m": center.tolist(),', '"debug_points": group.tolist(), "debug_pixels": image_points.tolist(), "debug_rgb": rgb[image_points[:,1],image_points[:,0]].tolist(), "debug_hsv": hsv[image_points[:,1],image_points[:,0]].tolist(), "debug_oriented_center": ((np.percentile(projected,5,axis=0)+np.percentile(projected,95,axis=0))/2 @ np.column_stack((major,minor)).T).tolist(), "position_base_m": center.tolist(),')
exec(compile(source,'diagnostic_rgbd','exec'),scope)
f=json.loads((root/'project-initial-block-input.json').read_text());cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
with np.load(root/'project-initial-block-sensors.npz') as raw:
 candidates,plane=scope['metric_candidates'](raw['rgb'],raw['depth_m'],raw['intrinsic'],raw['camera_to_base'],cfg['cartesian_alignment'],stride=2,spatial_components=True)
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text());pose=kin.matrix(np.array(f['failed']['observation_state'])[8:15],1)
c=min(candidates,key=lambda c:np.linalg.norm(np.array(c['position_base_m'])[:2]-np.array(f['failed']['rgbd_candidates'][0]['position_base_m'])[:2]))
alt=dict(c);alt['position_base_m']=c['debug_oriented_center']+[c['position_base_m'][2]]
hsv=np.array(c['debug_hsv']);pts=np.array(c['debug_points'])
out=dict(candidate=c,plane=plane,original_clear=open_fingers_clear(pose,c,plane),original_gaps_m=FingerSeparation(c,plane).gaps(pose).tolist(),oriented_center_clear=open_fingers_clear(pose,alt,plane),oriented_center_gaps_m=FingerSeparation(alt,plane).gaps(pose).tolist(),center_shift_m=(np.array(alt['position_base_m'])-c['position_base_m']).tolist(),color_counts=dict(total=len(hsv),sat_under10=int((hsv[:,1]<10).sum()),sat_under20=int((hsv[:,1]<20).sum())),low_sat_points=pts[hsv[:,1]<10].tolist())
(root/'release-component-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['candidate','low_sat_points']},indent=2))

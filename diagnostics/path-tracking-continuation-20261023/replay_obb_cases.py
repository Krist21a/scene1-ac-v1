"""Same observed points/extents, corrected oriented-box centers on retained cases."""
import copy,json,sys,time
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-obb';sys.path.insert(0,str(policy))
from scene1.rgbd import metric_candidates
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
from scene1.retreat_projection import FingerSeparation
from scene1.release_retreat import RetreatPreflightError
from run_scene1 import make_residual_assist
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text());cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text());out=[]
for item in json.loads((root/'obb-replay-inputs/manifest.json').read_text()):
 label=item['label'];f=json.loads((root/'obb-replay-inputs'/(label+'.json')).read_text())
 with np.load(root/'obb-replay-inputs'/(label+'.npz')) as raw:
  candidates,plane=metric_candidates(raw['rgb'],raw['depth_m'],raw['intrinsic'],raw['camera_to_base'],cfg['cartesian_alignment'],stride=2,spatial_components=True)
 target=min(candidates,key=lambda c:np.linalg.norm(np.array(c['position_base_m'])[:2]-np.array(f['pad'])[:2]));old=copy.deepcopy(target);del old['oriented_center_base_m']
 measured=np.array(f['failed']['observation_state']);pose=kin.matrix(measured[8:15],1)
 record=dict(label=label,raw_sha256=item['raw_sha256'],observed_points=target['points'],dimensions_m=target['oriented_extent_m'],old_center_m=old['position_base_m'],center_m=target['oriented_center_base_m'],center_shift_m=(np.array(target['oriented_center_base_m'])-old['position_base_m']).tolist(),old_clear=open_fingers_clear(pose,old,plane),new_clear=open_fingers_clear(pose,target,plane),old_gap_m=FingerSeparation(old,plane).gaps(pose).tolist(),new_gap_m=FingerSeparation(target,plane).gaps(pose).tolist())
 a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.);a.wrist_config=cfg['wrist_assist'];a.weighing_target=np.array(f['pad']);base=np.array(f['release']['proposed_action']);a.offset=np.array(f['last']['proposed_action'])-base;a.observe_command(f['last']['sent_action']);a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=f['last']['cartesian_alignment']['phase_steps']
 start=time.monotonic()
 try:
  command,info=a._pending_action(measured,dict(candidates=candidates,table_z_base_m=plane,stamp=f['failed']['rgbd_stamp']));record['certificate']=info['retreat_goal_check']
 except Exception as error:
  record['error']=str(error)
  if isinstance(error,RetreatPreflightError):record['certificate']=error.retreat_preflight_audit
 record['elapsed_s']=time.monotonic()-start;out.append(record)
(root/'obb-cases-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(label=x['label'],old_clear=x['old_clear'],new_clear=x['new_clear'],shift_mm=(np.array(x['center_shift_m'])*1000).tolist(),certified=x.get('certificate',{}).get('certified'),error=x.get('error'),seconds=x['elapsed_s']) for x in out],indent=2))

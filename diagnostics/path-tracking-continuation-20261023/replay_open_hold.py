"""Opening-hold sensitivity using retained unload disturbance, not dynamics proof."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
policy=root/'candidate-policy-executor-hold'
sys.path.insert(0,str(policy))
from run_scene1 import make_residual_assist
from scene1.kinematics import UrdfArmKinematics
from scene1.motion import MotionFilter
from scene1.clearance import open_fingers_clear
from scene1.release_retreat import RetreatPreflightError
f=json.loads((root/'hold-drift-input.json').read_text())
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=Path('/tmp/scene1_robot.urdf').read_text())
release=f['release'];base=np.array(release['proposed_action']);q0=np.array(release['observation_state'])
pose=kin.matrix(q0[8:15],1);target=f['failed']['retreat_preflight_audit']['observed_parcel']
plane=f['last']['rgbd_table_z_base_m'];out=[]
for response in [1.,.5,.25,.1]:
 for enabled in [False,True]:
  a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.)
  a.wrist_config=copy.deepcopy(cfg['wrist_assist']);a.wrist_config['release_cartesian_open_hold']=enabled
  a.weighing_target=np.r_[target['position_base_m'][:2],plane]
  a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={},
           release_hold_position=pose[:3,3].copy(),release_hold_rotation=Rotation.from_matrix(pose[:3,:3]))
  motion=MotionFilter(a.low,a.high,1.,.6);motion.reset(release['sent_action'])
  effect=np.zeros(16);records=[]
  for row in f['opening_rows'][1:]:
   measured=np.array(row['observation_state'])+effect
   proposal,info=a._pending_action(measured,None)
   sent,limit=motion.apply(proposal,.1);a.observe_command(sent)
   effect+=response*(sent-np.array(row['sent_action'])-effect)
   records.append(dict(step=row['step'],hold=info.get('release_pose_hold'),motion=limit,
       measured_tcp_m=kin.matrix(measured[8:15],1)[:3,3].tolist()))
  measured=np.array(f['failed']['observation_state'])+effect
  snapshot=dict(candidates=[target],table_z_base_m=plane,stamp=f['failed']['rgbd_stamp'])
  try:
   proposal,info=a._pending_action(measured,snapshot)
   audit=info['retreat_goal_check']
  except RetreatPreflightError as error:
   audit=error.retreat_preflight_audit
  out.append(dict(response=response,hold_enabled=enabled,records=records,retreat_certificate=audit,
      end_drift_m=float(np.linalg.norm(kin.matrix(measured[8:15],1)[:3,3]-pose[:3,3])),
      final_pose_clear=open_fingers_clear(kin.matrix(measured[8:15],1),target,plane),
      limitation='Linear additive retained unload disturbance with assumed command response; no contact/plant identification, no commands sent'))
(root/'open-hold-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(response=x['response'],hold=x['hold_enabled'],drift=x['end_drift_m'],clear=x['final_pose_clear'],certified=x['retreat_certificate']['certified']) for x in out],indent=2))

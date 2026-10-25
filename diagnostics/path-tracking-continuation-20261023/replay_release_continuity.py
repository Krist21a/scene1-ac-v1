"""Reference jump and unchanged geometric certificate, no dynamics claim."""
import copy,json,sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
policy=root/'candidate-policy-executor-continuity'
sys.path.insert(0,str(policy))
from run_scene1 import make_residual_assist
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
from scene1.release_retreat import RetreatPreflightError
from scene1.motion import MotionFilter
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'hold-reference-jump-input.json').read_text())
q=np.array(f['release']['observation_state']);before=np.array(f['before']['sent_action'])
target=f['failed']['retreat_preflight_audit']['observed_parcel'];plane=f['last']['rgbd_table_z_base_m']
snapshot=dict(candidates=[target],table_z_base_m=plane,stamp=f['failed']['rgbd_stamp'])
out=[]
for mode in ['measured_physical','continuous_command']:
 a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.)
 a.wrist_config=copy.deepcopy(cfg['wrist_assist']);a.wrist_config['release_reference_mode']=mode
 a.weighing_target=np.r_[target['position_base_m'][:2],plane]
 a.geometry[1]=dict(extent=np.array([.11,.04,.1]));a._view_ready=lambda *args:True
 a.observe_command(before)
 a._begin('weighing',1,before.copy(),stage='release',reference_base=before.copy(),requested_open=0.,surface={},hold_grip=.98)
 command,release=a._pending_action(q,snapshot)
 motion=MotionFilter(a.low,a.high,1.,.6);motion.reset(before);sent,limit=motion.apply(command,.1);a.observe_command(sent)
 # Kinematic hold counterfactual: measured Cartesian release pose stays fixed.
 for _ in range(12):
  command,info=a._pending_action(q,None)
  sent,limit=motion.apply(command,.1);a.observe_command(sent)
 q[15]=0.
 try:
  command,info=a._pending_action(q,snapshot);audit=info['retreat_goal_check']
 except RetreatPreflightError as error:audit=error.retreat_preflight_audit
 out.append(dict(mode=mode,release_transfer=release['release_reference_transfer'],
   measured_pose_clear=open_fingers_clear(kin.matrix(q[8:15],1),target,plane),
   previous_command_pose_clear=open_fingers_clear(kin.matrix(before[8:15],1),target,plane),
   certificate=audit,limitation='Fixed measured release posture is a kinematic counterfactual, not a load/contact simulation. Original failed measured pose remains blocked.'))
(root/'release-continuity-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(mode=x['mode'],jump=max(abs(v) for v in x['release_transfer']['requested_change_rad']),clear=x['measured_pose_clear'],command_clear=x['previous_command_pose_clear'],certificate=x['certificate']['certified'],reason=x['certificate']['reason'],steps=x['certificate'].get('nominal_steps')) for x in out],indent=2))

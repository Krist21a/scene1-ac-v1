"""Retained pre-retreat states, original guard, hypothetical earlier readiness."""
import copy,json,sys
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-open'
sys.path.insert(0,str(policy))
from run_scene1 import make_residual_assist
from scene1.kinematics import UrdfArmKinematics
from scene1.release_retreat import RetreatPreflightError
from scene1.clearance import open_fingers_clear
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'hold-reference-jump-input.json').read_text())
q0=np.array(f['release']['observation_state']);pose=kin.matrix(q0[8:15],1)
base=np.array(f['release']['proposed_action']);target=f['failed']['retreat_preflight_audit']['observed_parcel']
plane=f['last']['rgbd_table_z_base_m'];snapshot=dict(candidates=[target],table_z_base_m=plane,stamp=f['failed']['rgbd_stamp'])
out=[]
for dwell in [3,4,5,6,8,12]:
 previous=f['opening_rows'][dwell]
 row=f['opening_rows'][dwell+1] if dwell+1<len(f['opening_rows']) else f['failed']
 measured=np.array(row['observation_state'])
 a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.)
 a.wrist_config=copy.deepcopy(cfg['wrist_assist']);a.wrist_config['release_open_dwell_steps']=dwell
 a.weighing_target=np.r_[target['position_base_m'][:2],plane]
 a.offset=np.array(previous['proposed_action'])-base
 a.observe_command(previous['sent_action'])
 a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={},release_hold_position=pose[:3,3],release_hold_rotation=Rotation.from_matrix(pose[:3,:3]))
 a.pending['steps']=dwell
 try:
  command,info=a._pending_action(measured,snapshot);audit=info['retreat_goal_check']
 except RetreatPreflightError as error:audit=error.retreat_preflight_audit
 out.append(dict(dwell=dwell,step=row['step'],gripper=measured[15],drift_m=float(np.linalg.norm(kin.matrix(measured[8:15],1)[:3,3]-pose[:3,3])),clear=open_fingers_clear(kin.matrix(measured[8:15],1),target,plane),certificate=audit,limitation='Uses later observed parcel geometry with actual earlier joint observations. No early RGB-D snapshot, no predicted unload/contact dynamics, no commands sent.'))
(root/'open-readiness-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(dwell=x['dwell'],step=x['step'],gripper=x['gripper'],drift=x['drift_m'],clear=x['clear'],certificate=x['certificate']['certified'],reason=x['certificate']['reason']) for x in out],indent=2))

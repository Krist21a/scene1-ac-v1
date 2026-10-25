"""Geometric screening only: preserve loaded wrist references at release."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'candidate-policy-executor-fast'))
from scene1.kinematics import UrdfArmKinematics
from scene1.command_sweep import sampled_sweep
from scene1.retreat_projection import FingerSeparation
from scipy.spatial.transform import Rotation
k=UrdfArmKinematics(description=(root/'candidate-policy-executor-fast/tests/fixtures/release_robot.urdf').read_text())
x=json.loads((root/'retreat-preflight-input.json').read_text())
r=x['release'];c=r['cartesian_alignment']['release_reference_transfer']
measured=np.array(c['measured_target_rad']);previous=np.array(c['previous_command_rad'])
target=x['row']['cartesian_alignment']['retreat_goal_check']['observed_parcel'];plane=x['row']['rgbd_table_z_base_m']
origin=k.matrix(measured,1);separation=FingerSeparation(target,plane);cases=[]
for name,q in [('measured',measured),('previous',previous),('measured_proximal_previous_wrist',np.r_[measured[:4],previous[4:]]),('previous_proximal_measured_wrist',np.r_[previous[:4],measured[4:]])]:
 p=k.matrix(q,1)
 cases.append(dict(name=name,q=q.tolist(),tcp_delta_m=(p[:3,3]-origin[:3,3]).tolist(),
     rotation_change_degrees=float(np.rad2deg((Rotation.from_matrix(origin[:3,:3]).inv()*Rotation.from_matrix(p[:3,:3])).magnitude())),
     gaps_m=separation.gaps(p).tolist(),sweep=sampled_sweep(k,measured,q,target,plane)))
out=dict(source='executor-obb-c-p4-s0',cases=cases,limitations='Kinematic screening using first post-opening geometry; no plant/contact simulation or dynamic-benefit proof.')
(root/'release-hybrid-replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

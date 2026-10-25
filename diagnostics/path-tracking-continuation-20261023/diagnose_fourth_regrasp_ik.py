import json,sys
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
p=Path(__file__).resolve().parent;policy=p/'candidate-policy-executor-support-prepare';sys.path.insert(0,str(policy));from scene1.kinematics import UrdfArmKinematics
k=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text());f=json.loads((p/'fourth-regrasp-failure-input.json').read_text());row=f['last'];c=row['cartesian_alignment'];target=c['command_sweep']['target'];plane=c['command_sweep']['plane_m'];q=np.array(row['observation_state'][8:15]);pos=k.matrix(q,1)[:3,3];forward=np.array(target['major_axis_xy']);forward/=np.linalg.norm(forward)
if forward@(np.array(target['position_base_m'])[:2]-pos[:2])<0:forward=-forward
pitch=np.deg2rad(yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())['wrist_assist']['regrasp_pitch_degrees']);closing=np.r_[-forward[1],forward[0],0];z=-np.r_[forward*np.cos(pitch),-np.sin(pitch)];rot=Rotation.from_matrix(np.column_stack((closing,np.cross(z,closing),z)));goal=np.r_[target['grasp_goal_xy'],(plane+target['top_z_base_m'])/2]
def fun(q):
 t=k.matrix(q,1);return np.r_[t[:3,3]-goal,.15*(rot*Rotation.from_matrix(t[:3,:3]).inv()).as_rotvec()]
lo,hi=k.limits[1][:,0],k.limits[1][:,1];out=[]
for start in [q,np.array(row['sent_action'][8:15]),(lo+hi)/2]:
 s=least_squares(fun,np.clip(start,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=1500,gtol=1e-10,ftol=1e-10,xtol=1e-10);v=fun(s.x);out.append(dict(success=bool(s.success),position_error_m=float(np.linalg.norm(v[:3])),angle_error_degrees=float(np.rad2deg(np.linalg.norm(v[3:])/.15)),q=s.x.tolist(),max_change_from_measured_rad=float(np.max(abs(s.x-q)))))
report=dict(goal=goal.tolist(),actual=pos.tolist(),original_physical_limits=[lo.tolist(),hi.tolist()],solutions=out,interpretation='Offline static IK feasibility only; not a swept-path or dynamic execution certificate');(p/'fourth-regrasp-ik.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

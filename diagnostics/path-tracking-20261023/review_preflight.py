"""Offline audit of the retained E rejection; no commands or new limits."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
ROOT=Path('/root/scene1_path_tracking_20261023')
sys.path.insert(0,str(ROOT/'candidate-policy-axis-sweep'))
from scene1.kinematics import UrdfArmKinematics
x=json.loads(Path('/tmp/e-preflight-fixture.json').read_text())
k=UrdfArmKinematics(description=Path('/tmp/scene1_robot.urdf').read_text())
q=np.array(x['failed']['observation_state'])[8:15]
transfer=x['release']['cartesian_alignment']['release_reference_transfer'];ref=np.array(transfer['original_reference_rad']);cap=transfer['correction_cap_rad']
low=np.maximum(np.maximum(-3.14,k.limits[1][:,0]),ref-cap)
high=np.minimum(np.minimum(3.14,k.limits[1][:,1]),ref+cap)
pose=k.matrix(q,1);rot=Rotation.from_matrix(pose[:3,:3]);axis=pose[:2,2]/np.linalg.norm(pose[:2,2]);goal=pose[:3,3]+np.r_[.06*axis,.07]
def error(j):
 p=k.matrix(j,1)
 return p[:3,3]-goal,(rot.inv()*Rotation.from_matrix(p[:3,:3])).as_rotvec()
def report(j):
 p,r=error(j)
 return dict(position_error_m=float(np.linalg.norm(p)),orientation_error_deg=float(np.rad2deg(np.linalg.norm(r))),joints=j.tolist(),within_original_bounds=bool(np.all(j>=low-1e-9) and np.all(j<=high+1e-9)),accepted=bool(np.linalg.norm(p)<=.014 and np.linalg.norm(r)<=np.deg2rad(8)))
rows=[]
for weight in [.15,.12,.10,.08]:
 for name,seed in [('measured',q),('original_reference',ref),('bounds_midpoint',(low+high)/2)]:
  def fun(j):
   p,r=error(j);return np.r_[p,weight*r]
  result=least_squares(fun,np.clip(seed,low,high),bounds=(low,high),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
  rows.append(dict(method='weighted least squares',weight=weight,start=name,**report(result.x)))
seed=np.array(rows[0]['joints'])
result=minimize(lambda j:float(error(j)[0]@error(j)[0]),seed,method='SLSQP',bounds=list(zip(low,high)),constraints=[dict(type='ineq',fun=lambda j:np.deg2rad(8)**2-float(error(j)[1]@error(j)[1]))],options=dict(maxiter=180,ftol=1e-12))
rows.append(dict(method='explicit same orientation constraint, minimize position error',solver_success=bool(result.success),solver_message=str(result.message),**report(result.x)))
out=dict(original_error=x['failed']['error'],failed_step=x['failed']['step'],origin=pose[:3,3].tolist(),goal=goal.tolist(),low=low.tolist(),high=high.tolist(),tests=rows,conclusion='Numerical feasibility evidence only; no task execution, no budget or tolerance change')
Path('/tmp/e-preflight-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

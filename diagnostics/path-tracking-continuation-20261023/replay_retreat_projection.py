"""Reproduce an actually rejected proposal and solve within its original bounds."""
import json,sys,time
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-project';sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from scene1.retreat_projection import project_retreat_reference,FingerSeparation
from run_scene1 import make_residual_assist
f=json.loads((policy/'tests/fixtures/retreat_blocked_reference_20261023.json').read_text())
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.)
measured=np.array(f['measured']);previous=np.array(f['previous']);proposed=np.array(f['proposed']);goal=np.array(f['goal'])
start=time.monotonic();chosen,audit=project_retreat_reference(kin,measured,proposed,previous,previous,a.low,a.high,.006,f['target'],f['plane'],a.retreat_motion_config)
out=dict(audit=audit,elapsed_s=time.monotonic()-start,chosen=chosen.tolist(),previous_goal_error_m=float(np.linalg.norm(kin.matrix(previous[8:15],1)[:3,3]-goal)),chosen_goal_error_m=float(np.linalg.norm(kin.matrix(chosen[8:15],1)[:3,3]-goal)),proposed_goal_error_m=float(np.linalg.norm(kin.matrix(proposed[8:15],1)[:3,3]-goal)),separation_m=FingerSeparation(f['target'],f['plane']).gaps(kin.matrix(chosen[8:15],1)).tolist(),limitation='One actual blocked command replanned with unchanged sampled geometry and bounds; not a future dynamics guarantee')
(root/'retreat-projection-replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

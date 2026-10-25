import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'candidate-policy-executor-prepare-adapt'))
from scene1.kinematics import UrdfArmKinematics
from scene1.retreat_projection import FingerSeparation
from scene1.command_sweep import sampled_sweep
k=UrdfArmKinematics(description=(root/'candidate-policy-executor-prepare-adapt/tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'prepare-adapt-clearance-input.json').read_text())
near=[c for c in f['failed']['rgbd_candidates'] if np.linalg.norm(np.array(c['position_base_m'])[:2]-np.array(f['pad'])[:2])<.12]
assert len(near)==1
target=near[0];sep=FingerSeparation(target,f['plane']);results=[]
for r in f['rows']+[f['failed']]:
    q=np.array(r['observation_state'])[8:15];pose=k.matrix(q,1)
    results.append(dict(step=r['step'],state=r.get('cartesian_alignment',{}).get('state','failed'),
        pose=pose.tolist(),q=q.tolist(),gap_m=sep.gaps(pose).tolist(),
        gripper=r['observation_state'][15],
        reference_sweep=sampled_sweep(k,q,np.array(r['sent_action'])[8:15],target,f['plane']) if 'sent_action' in r else None))
out=dict(label=f['label'],target=target,plane=f['plane'],records=results,
    limitation='One post-opening observed parcel box reused for all earlier poses. Most recently logged plane reused; counterfactual geometry comparison, not proof of earlier contact.')
(root/'prepare-adapt-clearance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(target=target,plane=f['plane'],records=[{k:v for k,v in r.items() if k not in ['pose','q']} for r in results]),indent=2))

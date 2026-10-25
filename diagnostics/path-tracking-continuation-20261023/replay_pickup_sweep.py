import sys,json
from pathlib import Path
import numpy as np
policy=Path('/root/scene1_path_tracking_continuation_20261023/candidate-policy-executor-pickup-sweep');sys.path.insert(0,str(policy))
from scene1.command_sweep import command_sweep_audit
from scene1.kinematics import UrdfArmKinematics
k=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text());rows=list(map(json.loads,Path('/tmp/first-select-output/steps.jsonl').read_text().splitlines()));out=[]
for i,r in enumerate(rows):
 a=r['cartesian_alignment']
 if a.get('stage') not in ('clear','hover','descend') or 'selected_target' not in a or i%10:continue
 c=dict(stage='visual_pickup_'+a['stage'],target=a['selected_target'],plane=r['rgbd_table_z_base_m'])
 audit=command_sweep_audit(k,np.array(r['measured_state']),r['proposed_action'],r['sent_action'],rows[max(0,i-1)]['sent_action'],c)
 out.append(dict(step=r['step'],legacy_increment_clear=a.get('open_fingers_clear'),audit=audit))
print(json.dumps(dict(samples=len(out),filtered_blocked_steps=[r['step'] for r in out if not r['audit']['filtered']['clear']],records=out,limitation='Recorded planning feedback, not a fresh runtime command audit; sampled every10 policy frames'),indent=2))

"""Audit actual published progressive opening and preserved open feedback gate."""
import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent;label=sys.argv[1]
rows=[json.loads(s) for s in (p/label/'output/steps.jsonl').read_text().splitlines()]
blocks=[];active=[]
for i,row in enumerate(rows):
 c=row['cartesian_alignment'];ramp=c.get('release_gripper_ramp')
 if ramp and ramp.get('step') is not None:
  assert i>0
  previous=rows[i-1]['sent_action'][15];sent=row['sent_action'][15]
  assert abs(sent-max(0.,previous-ramp['step']))<1e-12
  assert ramp['original_open_timeout_steps']==40
  active.append(dict(step=row['step'],sent=sent,measured=row['observation_state'][15],hold_error_m=c.get('release_pose_hold',{}).get('position_error_m')))
 elif active:
  if c.get('retreat_goal_check'):
   assert rows[i-1]['sent_action'][15]==0.
   assert row['observation_state'][15]<=.1
  blocks.append(dict(rows=active,max_hold_error_m=max(x['hold_error_m'] or 0. for x in active)));active=[]
if active:blocks.append(dict(rows=active,incomplete=True))
out=dict(label=label,blocks=blocks,all_checks_passed=True);target=p/'analysis-full'/label/'soft-open-audit.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(label=label,blocks=len(blocks),all_checks_passed=True)))

"""Audit full visual first pickup and final-send sweeps without implying task success."""
import sys,json
from pathlib import Path
p=Path(__file__).resolve().parent;label=sys.argv[1];d=p/label;rows=[json.loads(l) for l in (d/'output/steps.jsonl').read_text().splitlines()];first=next(i for i,r in enumerate(rows) if r['cartesian_alignment'].get('stage')=='first_clear');contact=next((i for i in range(first,len(rows)) if rows[i]['cartesian_alignment'].get('state')=='grasp_contact_confirmed'),None);end=contact if contact is not None else len(rows)-1
errors=[]
if first==0 or rows[first-1]['cartesian_alignment'].get('state')!='pregrasp_aligned':errors.append('First visual entry not immediately after original pregrasp confirmation')
for r in rows[first:end+1]:
 a=r['cartesian_alignment']
 if a.get('policy_controls_arms') is not False or not r.get('policy_paused_for_alignment'):errors.append('ACT active at '+str(r['step']))
 if a.get('stage')!='local_grasp' and max(r['sent_action'][7],r['sent_action'][15])>.15:errors.append('Premature closure at '+str(r['step']))
sweeps=[]
for r in rows:
 a=r['cartesian_alignment']
 if a.get('strategy')=='visual_pickup' and a.get('stage') in ['first_clear','first_back','clear','hover','descend']:
  audit=a.get('command_sweep');ok=bool(audit and audit.get('feedback',{}).get('source')=='fresh_sdk_get_obs_after_planning' and audit['filtered']['clear'])
  sweeps.append(dict(step=r['step'],valid=ok,stage=a['stage']))
  if not ok:errors.append('Missing/failed final-send sweep '+str(r['step']))
transitions=[]
for r in rows[first:end+1]:
 a=r['cartesian_alignment'];stage=a.get('stage')
 if not transitions or transitions[-1]['stage']!=stage:transitions.append(dict(step=r['step'],stage=stage))
selected=next((r['cartesian_alignment']['selected_target'] for r in rows[first:] if r['cartesian_alignment'].get('stage')=='clear' and 'selected_target' in r['cartesian_alignment']),None)
result=dict(label=label,status='passed' if not errors else 'failed',errors=errors,entry_step=rows[first]['step'],contact_step=rows[contact]['step'] if contact is not None else None,transitions=transitions,final_send_sweep_frames=len(sweeps),sweep_failures=[r for r in sweeps if not r['valid']],selected_target=selected,task_success=None,full_task_acceptance_claimed=False)
out=p/'analysis-full'/label/'first-pickup-audit.json';out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(errors))

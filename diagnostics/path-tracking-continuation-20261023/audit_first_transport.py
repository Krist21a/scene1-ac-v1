"""Audit the first confirmed-grasp transition without accepting failed tasks."""
import argparse,json
from pathlib import Path
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args();d=p/a.label
rows=[json.loads(line) for line in (d/'output/steps.jsonl').open()]
first=next((i for i,r in enumerate(rows) if r['cartesian_alignment'].get('strategy')=='visual_pickup'),None)
out=dict(label=a.label,first_transport_exercised=first is not None,errors=[])
if first is not None:
 r=rows[first];prev=rows[first-1];info=r['cartesian_alignment']
 assert prev['cartesian_alignment'].get('state')=='grasp_contact_confirmed'
 assert info['stage']=='lift'
 assert r['sent_action'][15]>=.9
 assert all(x['cartesian_alignment'].get('policy_controls_arms') is False for x in rows[first:] if x['cartesian_alignment'].get('strategy')=='visual_pickup')
 out.update(first_transport_step=r['step'],preceding_contact_step=prev['step'],
            first_stage=info['stage'],first_goal=next((x['cartesian_alignment']['goal_tcp'][1] for x in rows[first:] if x['cartesian_alignment'].get('goal_tcp')),None),
            initial_observation_hold=('goal_tcp' not in info),
            first_command_delta=[x-y for x,y in zip(r['sent_action'],prev['sent_action'])],
            stages=list(dict.fromkeys(x['cartesian_alignment'].get('stage') for x in rows[first:] if x['cartesian_alignment'].get('strategy')=='visual_pickup')))
state=json.loads((d/'experiment.json').read_text());out.update(task_success=state.get('task_success'),policy_summary=state.get('policy_summary'))
(d/'first-transport-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

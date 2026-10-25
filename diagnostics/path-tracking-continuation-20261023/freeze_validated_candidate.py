"""Freeze a candidate after development seed0 and used-seed1 regression full4 pass."""
import argparse,datetime,hashlib,json
from pathlib import Path
from controller_variants import variant
p=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--development',required=True);ap.add_argument('--regression',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
name=a.policy.replace('_','-');manifest=p/(name+'-manifest.json');expected=read(manifest)
for rel,digest in expected.items():assert sha(p/('candidate-policy-'+name)/rel)==digest
trials=[]
for label,seed in [(a.development,0),(a.regression,1)]:
 d=p/label;state=read(d/'experiment.json');pipeline=read(p/(label+'-pipeline.json'));analysis=p/'analysis-full'/label
 assert state['seed']==seed and state['version']=='origin_pd' and state['policy_manifest_sha256']==sha(manifest)
 assert state.get('ended_at') and state.get('independent_four_parcel_visual_completion') is True
 assert state['policy_summary']['visual_confirmed_parcels']==4
 assert pipeline['current_stage']=='done' and all(v['returncode']==0 for v in pipeline['stages'].values())
 audits=['audit.json','arm-preparation-audit.json','soft-open-audit.json','support-recovery-audit.json','first-pickup-audit.json']
 for f in audits:
  r=read(analysis/f)
  if f in ['arm-preparation-audit.json','soft-open-audit.json','support-recovery-audit.json']:assert r['all_checks_passed'],f
  if f=='first-pickup-audit.json':assert r['status']=='passed',f
 trials.append(dict(label=label,seed=seed,steps=state['policy_summary']['steps'],runtime_sha256=sha(d/'runtime.json'),audits={f:sha(analysis/f) for f in audits},task_success=state.get('task_success')))
for trial in p.glob('*/experiment.json'):assert read(trial).get('seed') not in [2,3],'Fresh seed already used: '+str(trial)
planned=[]
for seed,version in [(0,'baseline'),(1,'baseline'),(2,'baseline'),(2,'origin_pd'),(3,'baseline'),(3,'origin_pd')]:
 v='base' if version=='baseline' else 'ac';planned.append(dict(seed=seed,controller=version,parcels=4,label=f'frozen-{name}-{v}-p4-s{seed}-1025'))
receipt=dict(at=datetime.datetime.now().astimezone().isoformat(),policy=a.policy,policy_manifest_sha256=sha(manifest),selected_controller='origin_pd',controller_identities={v:variant(v) for v in ['baseline','origin_pd']},development_trials=trials,used_development_seeds=[0,1],fresh_heldout_seeds=[2,3],planned_trials=planned,official_success_claimed=False,production_promoted=False,rules=['No path/control tuning on fresh2,3','Retain every failed attempt','Timing only for successful samecondition full4pairs','Separate proximal/wrist static/dynamic metrics','Null official verdict is not official success','User resumed20261025 to complete acceptance; previous quota stop retained; restore original runtime after study'])
out=p/a.output
with out.open('x') as f:f.write(json.dumps(receipt,indent=2,default=str)+'\n')
print(json.dumps(dict(freeze=str(out),sha256=sha(out),trials=planned)))

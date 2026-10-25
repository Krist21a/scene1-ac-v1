"""Freeze only after the independent development full-four trial and audits pass."""
import datetime,hashlib,json
from pathlib import Path
from controller_variants import variant
p=Path(__file__).resolve().parent
label='executor-first-transport-ac-p4-s0-1024';d=p/label;analysis=p/'analysis-full'/label
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
state=read(d/'experiment.json');pipeline=read(p/(label+'-pipeline.json'))
assert state.get('ended_at') and state.get('independent_four_parcel_visual_completion') is True
assert pipeline['current_stage']=='done' and all(v['returncode']==0 for v in pipeline['stages'].values())
assert state['policy_summary']['visual_confirmed_parcels']==4
for trial in p.glob('*/experiment.json'):
 assert read(trial).get('seed') not in (1,2),'Heldout seed already used: '+str(trial)
manifest=p/'executor-first-transport-manifest.json';expected=read(manifest)
for relative,digest in expected.items():assert sha(p/'candidate-policy-executor-first-transport'/relative)==digest
assert state['policy_manifest_sha256']==sha(manifest)
auditnames=['audit.json','arm-preparation-audit.json','soft-open-audit.json','support-recovery-audit.json']
assert read(analysis/'arm-preparation-audit.json')['all_checks_passed']
assert read(d/'first-transport-audit.json')['first_transport_exercised']
planned=[]
for seed,version in [(0,'baseline'),(1,'baseline'),(1,'origin_pd'),(2,'baseline'),(2,'origin_pd')]:
 name='base' if version=='baseline' else 'ac'
 planned.append(dict(seed=seed,controller=version,parcels=4,label=f'frozen-first-transport-{name}-p4-s{seed}-1024'))
r=dict(at=datetime.datetime.now().astimezone().isoformat(),policy='executor_first_transport',
 policy_manifest_sha256=sha(manifest),selected_controller='origin_pd',
 controller_identities={v:variant(v) for v in ['baseline','origin_pd']},
 development_trial=label,development_runtime_sha256=sha(d/'runtime.json'),
 development_audits={f:sha(analysis/f) for f in auditnames},
 first_transport_audit_sha256=sha(d/'first-transport-audit.json'),
 development_steps=state['policy_summary']['steps'],task_success=state.get('task_success'),
 official_success_claimed=False,production_promoted=False,tests_passed=496,patch_chain_count=31,
 heldout_seeds=[1,2],heldout_seeds_used_at_freeze=[],planned_trials=planned,
 rules=['No path/control tuning on heldout seeds','Every attempt retained, no success-selection retries',
        'Null official verdict remains null','Timing gains only for successful matched full-four pairs',
        'Wrist and proximal metrics separately','Restore original runtime on completion or remaining usage<=20%'])
f=p/'path-control-freeze-first-transport-20261024.json'
with f.open('x') as stream:stream.write(json.dumps(r,indent=2,default=str)+'\n')
print(json.dumps(dict(freeze=str(f),sha256=sha(f),trials=planned)))

"""Collect all scheduled attempts without filtering failures or null verdicts."""
import datetime
import hashlib
import json
from pathlib import Path
from compare_task_runs import load_run

P = Path(__file__).resolve().parent
read = lambda f: json.loads(f.read_text())
freeze = read(P / 'path-control-freeze-bin-dense-20261025.json')
schedule = read(P / 'frozen-acceptance-20261025.json')
assert schedule['status'] == 'completed'
rows = []
for seed in range(4):
    for version in ['baseline', 'origin_pd']:
        if version == 'origin_pd' and seed < 2:
            label = freeze['development_trials'][seed]['label']
        else:
            label = next(t['label'] for t in freeze['planned_trials'] if t['seed'] == seed and t['controller'] == version)
        state = read(P / label / 'experiment.json')
        assert state['policy_manifest_sha256'] == freeze['policy_manifest_sha256']
        run = load_run(label)
        analysis = P / 'analysis-full' / label
        audit = read(analysis / 'audit.json')
        preparation = read(analysis / 'arm-preparation-audit.json')
        row = dict(seed=seed, version=version, label=label,
                   steps=state['policy_summary']['steps'], parcels=state['policy_summary']['visual_confirmed_parcels'],
                   visual_full4=run['four_parcel_visual_complete'], task_success=run['official_task_success'],
                   failure=state['policy_summary'].get('error'), duration_s=run['duration_s'],
                   ros_run_id=state['ros_run_id'], independent_official_reset=state['independent_official_full_reset'],
                   proximal_wbc_rmse_rad=run['proximal_wbc_rmse_rad'], wrist_wbc_rmse_rad=run['wrist_wbc_rmse_rad'],
                   target_clips=audit['target_clips'], velocity_limited_steps=audit['velocity_limited_steps'],
                   pd_limits_50hz=audit['pd_limits_50hz'], reference=run['reference'], phases=run['phases'],
                   per_parcel=run['per_parcel'], joint_bounds=run['joint_bounds'],
                   preparation=[dict(steps=b['preparation_steps'], released=b['released'], final=b['last']) for b in preparation['blocks']],
                   audit_files={f:hashlib.sha256((analysis/f).read_bytes()).hexdigest() for f in ['audit.json','first-pickup-audit.json','arm-preparation-audit.json','soft-open-audit.json','support-recovery-audit.json']})
        rows.append(row)
assert len({r['ros_run_id'] for r in rows}) == 8
comparisons = {str(seed):read(P / f'frozen-comparison-seed{seed}-20261025.json') for seed in range(4)}
out = dict(generated_at=datetime.datetime.now().astimezone().isoformat(), freeze=freeze,
           runs=rows, comparisons=comparisons,
           selected_full4_passes=sum(r['visual_full4'] for r in rows if r['version']=='origin_pd'),
           selected_fresh_full4_passes=sum(r['visual_full4'] for r in rows if r['version']=='origin_pd' and r['seed'] in [2,3]),
           baseline_full4_passes=sum(r['visual_full4'] for r in rows if r['version']=='baseline'),
           official_success_claimed=False, production_promoted=False,
           limitations=['Single retained attempt per frozen condition; not a statistical reliability guarantee.',
                        'task_success=null is not official success.',
                        'Failed baseline and full candidate contain different phase distributions; aggregate RMS is descriptive only.',
                        'Use per-parcel common phases; timing requires both full4 successes.',
                        'C is bounded post-WBC compensation, not an internal QP change.',
                        'Velocity-correlation lag is approximate; boundary estimates do not prove zero delay.'])
(P / 'MULTISEED-RESULTS-20261025.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['selected_full4_passes','selected_fresh_full4_passes','baseline_full4_passes','official_success_claimed']}))

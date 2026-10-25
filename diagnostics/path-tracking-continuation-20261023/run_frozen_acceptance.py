"""Execute the immutable acceptance schedule once, preserving failed trials."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('freeze')
ap.add_argument('--receipt', default='frozen-acceptance-20261025.json')
a = ap.parse_args()
freeze_path = HERE / a.freeze
freeze = json.loads(freeze_path.read_text())
receipt_path = HERE / a.receipt
assert not receipt_path.exists(), 'No implicit acceptance retries'
receipt = dict(freeze=str(freeze_path), freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(), trials=[], status='running')

def save():
    receipt['updated_at'] = datetime.datetime.now().astimezone().isoformat()
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')

def run(script, arguments, log):
    with (HERE / log).open('x') as output:
        return subprocess.run([sys.executable, str(HERE / script), *arguments], stdout=output, stderr=subprocess.STDOUT).returncode

save()
try:
    for trial in freeze['planned_trials']:
        label = trial['label']
        receipt['current_trial'] = label
        save()
        rc = run('run_one.py', [trial['controller'], label, '--policy', freeze['policy'], '--seed', str(trial['seed']), '--parcels', '4'], label + '-orchestration.log')
        pipeline = json.loads((HERE / (label + '-pipeline.json')).read_text())
        state = json.loads((HERE / label / 'experiment.json').read_text())
        record = dict(**trial, returncode=rc, visual_complete=state.get('independent_four_parcel_visual_completion'), task_success=state.get('task_success'), summary=state.get('policy_summary'), audits={})
        receipt['trials'].append(record)
        save()
        assert pipeline['current_stage'] == 'done'
        for stage in ['switch', 'analysis', 'audit', 'archive', 'summary']:
            assert pipeline['stages'].get(stage, {}).get('returncode') == 0, (label, stage)
        for script in ['audit_first_pickup.py', 'audit_arm_preparation.py', 'audit_soft_open.py', 'audit_support_recovery.py']:
            record['audits'][script] = run(script, [label], label + '-' + script.removesuffix('.py') + '.log')
        save()
        if record['visual_complete']:
            assert all(rc == 0 for rc in record['audits'].values()), label
        assert run('archive_analysis_staging.py', [], label + '-staging-archive.log') == 0
        print(json.dumps(record), flush=True)
    receipt['status'] = 'completed'
except BaseException as error:
    receipt['status'] = 'stopped'
    receipt['error'] = repr(error)
    raise
finally:
    save()

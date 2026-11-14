"""Run all twenty preregistered final-test rounds after the model freeze.

Pair order alternates by seed. Every attempt, including policy failures, stays
in the final report; the final-test set cannot be used to revise the model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from evidence_storage import EVIDENCE_ROOT, STORE, require_storage
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-freeze', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', args.prefix):
        raise ValueError('Invalid final-test prefix')
    require_storage()
    freeze = json.loads(args.model_freeze.read_text())
    registry = json.loads(args.registry.read_text())
    seeds = registry['seed_splits']['test']
    split_hash = hashlib.sha256(json.dumps(registry['seed_splits'], sort_keys=True).encode()).hexdigest()
    if (freeze['stage'] != 'frozen_final' or freeze['training_updates'] != 20000
            or registry['status'] != 'complete' or freeze['test_seeds'] != seeds
            or len(set(seeds)) != 10 or freeze['seed_split_sha256'] != split_hash
            or registry['seed_split_sha256'] != split_hash
            or freeze['collection_registry_sha256'] != digest(args.registry)
            or freeze['record_video'] is not True):
        raise ValueError('Require the original ten final-test seeds and completed final freeze')
    frozen_at = datetime.fromisoformat(freeze['frozen_at'])
    if frozen_at.tzinfo is None or frozen_at >= datetime.now(timezone.utc):
        raise ValueError('Final model/configuration freeze must precede testing')
    provenance = dict(model_freeze_sha256=digest(args.model_freeze),
        registry_sha256=digest(args.registry), prefix=args.prefix, report=str(args.report.absolute()))
    policy_manifest = STORE/'policies'/(freeze['policy_label']+'-manifest.json')
    if digest(policy_manifest) != freeze['policy_manifest_sha256']:
        raise ValueError('Final policy manifest changed')
    if args.resume:
        batch = json.loads(args.output.read_text())
        if batch['provenance'] != provenance:
            raise ValueError('Cannot revise a preregistered final-test batch')
        if batch['status'] not in ('running', 'quota_stop', 'rounds_complete', 'complete'):
            raise ValueError('Incomplete evidence or restoration needs reconciliation')
        if batch['status'] == 'complete':
            return 0
    else:
        if args.output.exists() or args.report.exists():
            raise ValueError('Final-test batch or report already exists')
        collected = {row['label'] for row in registry['runs']}
        for path in STORE.glob('*-pipeline.json'):
            run = json.loads(path.read_text())
            if run.get('seed') in seeds and run['label'] not in collected:
                raise ValueError('Final-test seeds have already been evaluated; no replacement batch')
        labels = {mode:[f'{args.prefix}-{mode}-s{seed}' for seed in seeds]
                  for mode in ('traditional', 'local-act')}
        schedule = []
        for index, seed in enumerate(seeds):
            order = ('traditional', 'local-act') if index % 2 == 0 else ('local-act', 'traditional')
            schedule.extend(dict(seed=seed, mode=mode, label=labels[mode][index]) for mode in order)
        batch = dict(status='running', registered_at=datetime.now(timezone.utc).isoformat(),
            provenance=provenance, test_seeds=seeds, labels=labels, schedule=schedule, runs=[],
            pair_order='traditional first on even seed indices; local ACT first on odd indices')

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.pending.json')
        temporary.write_text(json.dumps(batch, indent=2)+'\n')
        temporary.replace(args.output)

    save()
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    for attempt in batch['schedule']:
        label = attempt['label']
        if any(row['label'] == label for row in batch['runs']):
            continue
        quota = usage_guard.latest()
        batch['latest_quota'] = quota
        if quota['reached']:
            batch['status'] = 'quota_stop'; save(); return 1
        if digest(args.model_freeze) != provenance['model_freeze_sha256']:
            raise ValueError('Model freeze changed during final testing')
        relative = Path(freeze['model_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe model evidence path')
        for name, expected in freeze['files_sha256'].items():
            name = Path(name)
            if name.is_absolute() or '..' in name.parts or digest(EVIDENCE_ROOT/relative/name) != expected:
                raise ValueError('Final checkpoint or preprocessing changed')
        pipeline_path = STORE/(label+'-pipeline.json')
        if pipeline_path.exists():
            raise ValueError('Existing attempt needs outcome reconciliation; never rerun: '+label)
        batch.update(status='running', current_label=label)
        save()
        key = 'traditional_profile' if attempt['mode'] == 'traditional' else 'local_act_profile'
        rc = subprocess.call([sys.executable, str(ROOT/'tools/speed_trial.py'), label,
            '--seed', str(attempt['seed']), '--parcels', '4', '--frozen-from', freeze['policy_label'],
            '--speed-profile', freeze[key], '--controller', freeze['controller'],
            '--cpu-layout', freeze['cpu_layout'], '--record-video'])
        pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
        outcome = dict(attempt, returncode=rc)
        try:
            outcome['acceptance'] = inspect(label, ROOT, STORE, 650.)
        except (OSError, ValueError, KeyError) as exc:
            outcome['evidence_error'] = str(exc)
        batch['runs'].append(outcome)
        save()
        cpu_path = STORE/(label+'-cpu-allocation.json')
        cpu = json.loads(cpu_path.read_text()) if cpu_path.exists() else {}
        if pipeline.get('stages', {}).get('restore') != 0 or cpu.get('restored') is not True:
            batch['status'] = 'restoration_incomplete'; save(); return 1
        if pipeline.get('quota_stop'):
            batch['status'] = 'quota_stop'; save(); return 1
        if any(pipeline.get('stages', {}).get(stage) != 0 for stage in ('analysis', 'audit', 'archive')):
            batch['status'] = 'evidence_incomplete'; save(); return 1
        from prune_policy_copies import prune, prune_analysis
        from compact_evidence import compact
        prune(label); prune_analysis(label); compact(label)
        print(json.dumps(dict(label=label, mode=attempt['mode'],
            passed=outcome.get('acceptance', {}).get('passed', False))), flush=True)
    batch.update(status='rounds_complete', current_label=None)
    save()
    rc = subprocess.call([sys.executable, str(ROOT/'tools/report_grasp_comparison.py'),
        '--traditional', *batch['labels']['traditional'], '--local-act', *batch['labels']['local-act'],
        '--registry', str(args.registry), '--model-freeze', str(args.model_freeze), '--output', str(args.report)])
    if rc == 0:
        batch.update(status='complete', report_sha256=digest(args.report))
        save()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())

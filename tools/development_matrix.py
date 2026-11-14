"""Complete a fixed development seed list, retaining every policy failure.

This entry point cannot sample holdouts or issue formal acceptance. Restoration,
archival and exact frozen-runtime agreement remain mandatory between resets.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from evidence_storage import STORE, require_storage
from speed_acceptance import inspect, agreement

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-from', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', value) for value in (args.prefix, args.frozen_from)):
        raise ValueError('Invalid development labels')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Development seeds must be distinct')
    require_storage()
    source = inspect(args.frozen_from, ROOT, STORE, 650.)
    if not source['passed']:
        raise ValueError('Source needs one complete guarded round below 650 s')
    output = STORE/(args.prefix+'-development.json')
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.resume:
        batch = json.loads(output.read_text())
        if (batch['frozen_source'] != source or batch['requested_seeds'] != args.seeds
                or batch['runner_sha256'] != runner_hash):
            raise ValueError('Cannot revise a registered development batch')
        if batch['status'] == 'complete':
            return 0
        if batch['status'] not in ('running', 'quota_stop'):
            raise ValueError('Infrastructure failure needs reconciliation before continuing')
    else:
        if output.exists():
            raise ValueError('Development batch already exists')
        batch = dict(status='running', frozen_source=source, requested_seeds=args.seeds,
            registered_at=datetime.now(timezone.utc).isoformat(), runner_sha256=runner_hash,
            runs=[], formal_acceptance=False, sampling='fixed development seeds; no replacements')

    def save():
        temporary = output.with_suffix('.pending.json')
        temporary.write_text(json.dumps(batch, indent=2)+'\n')
        temporary.replace(output)

    save()
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    for seed in args.seeds:
        if any(row['seed'] == seed for row in batch['runs']):
            continue  # Failed policy attempts are retained, never repeated.
        quota = usage_guard.latest()
        batch['latest_quota'] = quota
        if quota['reached']:
            batch['status'] = 'quota_stop'; save(); return 1
        label = f'{args.prefix}-s{seed}'
        pipeline_path = STORE/(label+'-pipeline.json')
        if pipeline_path.exists():
            raise ValueError('Existing attempt needs reconciliation; never overwrite: '+label)
        batch.update(status='running', current_label=label)
        save()
        rc = subprocess.call([sys.executable, str(ROOT/'tools/speed_trial.py'), label,
            '--seed', str(seed), '--parcels', '4', '--speed-profile', source['profile'],
            '--controller', source['controller'], '--cpu-layout', source['cpu_layout'],
            '--frozen-from', source['label'], '--record-video'])
        pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
        row = dict(label=label, seed=seed, returncode=rc)
        try:
            result = inspect(label, ROOT, STORE, 650.)
            row['acceptance'] = result
            row['frozen_match'] = (result['profile'] == source['profile']
                                   and all(agreement([source, result]).values()))
        except (OSError, ValueError, KeyError) as exc:
            row['evidence_error'] = str(exc)
        batch['runs'].append(row)
        save()
        cpu_path = STORE/(label+'-cpu-allocation.json')
        cpu = json.loads(cpu_path.read_text()) if cpu_path.exists() else {}
        if pipeline.get('stages', {}).get('restore') != 0 or cpu.get('restored') is not True:
            batch['status'] = 'restoration_incomplete'; save(); return 1
        if pipeline.get('quota_stop'):
            batch['status'] = 'quota_stop'; save(); return 1
        if (row.get('frozen_match') is not True
                or not row.get('acceptance', {}).get('checks', {}).get('cpu_allocation')
                or any(pipeline.get('stages', {}).get(stage) != 0 for stage in ('analysis', 'audit', 'archive'))):
            batch['status'] = 'evidence_incomplete'; save(); return 1
        from prune_policy_copies import prune, prune_analysis
        from compact_evidence import compact
        prune(label); prune_analysis(label); compact(label)
        print(json.dumps(dict(label=label, passed=result['passed'], wall_s=result['formal_wall_s'])), flush=True)
    batch.update(status='complete', current_label=None,
        all_development_rounds_passed=all(row['acceptance']['passed'] for row in batch['runs']))
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

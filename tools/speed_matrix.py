"""Sequential frozen-policy validation; stop at the first failed acceptance run."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/media/krist/KINGSTON/scene1-speed-20261025')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-from', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    for value in (args.frozen_from, args.prefix):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', value):
            parser.error('Invalid label')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Validation seeds must be distinct')
    source = inspect(args.frozen_from, ROOT, STORE)
    if not source['passed'] or not source['tests_passed']:
        raise RuntimeError('Frozen source must pass full timing, visual, safety and unit-test acceptance')
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    receipt = dict(frozen_source=source, requested_seeds=args.seeds, runs=[], status='running')
    output = STORE/(args.prefix+'-matrix.json')
    if output.exists():
        raise RuntimeError('Matrix receipt already exists; choose a new prefix')
    def save():
        output.write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    for seed in args.seeds:
        quota = usage_guard.latest()
        receipt['latest_quota'] = quota
        if quota['reached']:
            receipt['status'] = 'quota_stop'; save(); return 1
        label = f'{args.prefix}-s{seed}'
        receipt['current_label'] = label; save()
        rc = subprocess.call([sys.executable, str(ROOT/'tools/speed_trial.py'), label,
                              '--seed', str(seed), '--parcels', '4', '--speed-profile', source['profile'],
                              '--frozen-from', args.frozen_from, '--skip-tests'])
        if rc:
            receipt['runs'].append(dict(label=label, seed=seed, returncode=rc))
            receipt['status'] = 'trial_failed'; save(); return 1
        result = inspect(label, ROOT, STORE)
        result['frozen_match'] = result['manifest'] == source['manifest'] and result['profile'] == source['profile']
        receipt['runs'].append(result); save()
        print(json.dumps(dict(label=label, wall_s=result['formal_wall_s'], passed=result['passed'],
                              frozen_match=result['frozen_match'])), flush=True)
        if not result['passed'] or not result['frozen_match']:
            receipt['status'] = 'acceptance_failed'; save(); return 1
    receipt['status'] = 'passed'; save(); return 0


if __name__ == '__main__':
    raise SystemExit(main())

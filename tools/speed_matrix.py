"""Sequential frozen-policy validation; stop at the first failed acceptance run."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from speed_acceptance import inspect, agreement
import secrets
import random

ROOT = Path(__file__).resolve().parents[1]
from evidence_storage import STORE, require_storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-from', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--random-count', type=int, default=10)
    parser.add_argument('--exclude-seeds', type=int, nargs='*', default=list(range(10)))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-wall-s', type=float, choices=(600.,650.), default=600.)
    args = parser.parse_args()
    for value in (args.frozen_from, args.prefix):
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', value):
            parser.error('Invalid label')
    if args.random_count < 10:
        parser.error('At least ten random seeds are required')
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error('Validation seeds must be distinct')
    source = inspect(args.frozen_from, ROOT, STORE, args.max_wall_s)
    if not source['passed']:
        raise RuntimeError('Frozen source must pass full timing, visual and safety acceptance')
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    output = STORE/(args.prefix+'-matrix.json')
    if args.resume:
        receipt = json.loads(output.read_text())
        if receipt['frozen_source'] != source:
            raise RuntimeError('Frozen source changed')
        if receipt['status'] in ('trial_failed', 'acceptance_failed'):
            raise RuntimeError('Failed strategy batch cannot be resumed as passing')
        args.seeds = receipt['requested_seeds']
    else:
        if output.exists():
            raise RuntimeError('Matrix receipt already exists; choose a new prefix')
        entropy = secrets.randbits(128)
        rng = random.Random(entropy)
        excluded = set(args.exclude_seeds) | {source['seed']}
        # Once any seed has been exposed in development or a previous batch,
        # it cannot silently become a fresh holdout for a revised policy.
        for previous in STORE.glob('*-pipeline.json'):
            prior = json.loads(previous.read_text())
            if type(prior.get('seed')) is int:
                excluded.add(prior['seed'])
        if args.seeds is None:
            args.seeds = []
            while len(args.seeds) < args.random_count:
                seed = rng.randrange(0, 2**31)
                if seed not in excluded and seed not in args.seeds:
                    args.seeds.append(seed)
            sampling = dict(method='uniform rejection without replacement', entropy=entropy,
                            range=[0, 2**31-1], excluded=sorted(excluded))
        else:
            sampling = dict(method='explicit development seeds')
        receipt = dict(frozen_source=source, requested_seeds=args.seeds, sampling=sampling,
                       runs=[], status='running')
    def save():
        output.write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    completed = {r['seed'] for r in receipt['runs'] if r.get('passed') and r.get('frozen_match')}
    for seed in args.seeds:
        if seed in completed:
            continue
        quota = usage_guard.latest()
        receipt['latest_quota'] = quota
        if quota['reached']:
            receipt['status'] = 'quota_stop'; save(); return 1
        interrupted = [r for r in receipt.get('interrupted_attempts', []) if r['seed'] == seed]
        label = f'{args.prefix}-s{seed}' + (f'-a{len(interrupted)+1}' if interrupted else '')
        receipt['current_label'] = label; save()
        if (STORE/(label+'-pipeline.json')).exists():
            raise RuntimeError('Existing incomplete run requires analysis before resuming; never overwrite an attempt')
        rc = subprocess.call([sys.executable, str(ROOT/'tools/speed_trial.py'), label,
                              '--seed', str(seed), '--parcels', '4', '--speed-profile', source['profile'],
                              '--controller', source['controller'],
                              '--frozen-from', args.frozen_from,
                              '--cpu-layout', source.get('cpu_layout', 'original')] +
                              (['--record-video'] if source.get('record_video',False) else []))
        if rc:
            pipeline_path = STORE/(label+'-pipeline.json')
            pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
            if pipeline.get('quota_stop') and pipeline.get('stages', {}).get('restore') == 0:
                receipt.setdefault('interrupted_attempts', []).append(
                    dict(label=label, seed=seed, returncode=rc, reason='quota_stop'))
                receipt['status'] = 'quota_stop'; save(); return 1
            receipt['runs'].append(dict(label=label, seed=seed, returncode=rc))
            receipt['status'] = 'trial_failed'; save(); return 1
        result = inspect(label, ROOT, STORE, args.max_wall_s)
        result['frozen_match'] = (result['manifest'] == source['manifest'] and result['profile'] == source['profile']
                                  and result['cpu_layout'] == source['cpu_layout'] and all(agreement([source, result]).values()))
        receipt['runs'].append(result); save()
        print(json.dumps(dict(label=label, wall_s=result['formal_wall_s'], passed=result['passed'],
                              frozen_match=result['frozen_match'])), flush=True)
        if not result['passed'] or not result['frozen_match']:
            receipt['status'] = 'acceptance_failed'; save(); return 1
        # The next run reads the external frozen snapshot. Reclaim only its
        # independently verified container source copy after full archival.
        from prune_policy_copies import prune, prune_analysis
        prune(label)
        prune_analysis(label)
        from compact_evidence import compact
        compact(label)
    receipt['status'] = 'passed'; save()
    if receipt['sampling']['method'] == 'uniform rejection without replacement':
        return subprocess.call([sys.executable, str(ROOT/'tools/speed_acceptance.py'),
                                *[r['label'] for r in receipt['runs']],
                                '--seed-manifest', str(output), '--max-wall-s', str(args.max_wall_s), '--output',
                                str(STORE/(args.prefix+'-acceptance.json'))])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

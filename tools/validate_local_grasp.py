"""Preregister and execute the three ACT candidates on all ten validation seeds.

Policy failures remain in the batch and do not cause replacement seeds. A new
reset is forbidden until the preceding attempt has restored its controller and
CPU allocation. The first attempt freezes all three profiles together.
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
    parser.add_argument('--ranking', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', args.prefix):
        raise ValueError('Invalid validation prefix')
    require_storage()
    ranking = json.loads(args.ranking.read_text())
    registry = json.loads(args.registry.read_text())
    candidates = ranking['top_three']
    seeds = registry['seed_splits']['validation']
    if (registry['status'] != 'complete' or ranking['status'] != 'complete'
            or ranking['final_test_read'] is not False or ranking['requested_seeds'] != seeds
            or len(set(seeds)) != 10 or len(candidates) != 3
            or len({c['update'] for c in candidates}) != 3):
        raise ValueError('Require the completed collection and three loss-ranked validation candidates')
    acceptance_path = EVIDENCE_ROOT/registry['traditional_acceptance']['relative_path']
    if digest(acceptance_path) != registry['traditional_acceptance']['sha256']:
        raise ValueError('Traditional acceptance changed')
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance['passed'] is not True:
        raise ValueError('Traditional acceptance is required before learned validation')
    baseline = acceptance['runs'][0]
    if inspect(baseline['label'], ROOT, STORE, 650.) != baseline:
        raise ValueError('Accepted baseline evidence changed')
    provenance = dict(ranking_sha256=digest(args.ranking), collection_registry_sha256=digest(args.registry),
        traditional_acceptance_sha256=digest(acceptance_path), prefix=args.prefix)
    if args.resume:
        batch = json.loads(args.output.read_text())
        if batch['provenance'] != provenance:
            raise ValueError('Cannot revise a preregistered validation batch')
        if batch['status'] == 'complete':
            return 0
        if batch['status'] not in ('running', 'quota_stop'):
            raise ValueError('Incomplete infrastructure evidence needs reconciliation before continuing')
    else:
        if args.output.exists():
            raise ValueError('Validation batch already exists')
        accepted_files = json.loads((STORE/'policies'/(baseline['label']+'-manifest.json')).read_text())
        for name, expected in accepted_files.items():
            if digest(ROOT/name) != expected:
                raise ValueError('Policy changed since traditional acceptance: '+name)
        import yaml
        original = yaml.safe_load((ROOT/baseline['profile']).read_text())
        if original['visual_cycle'].get('grasp_execution') != 'traditional':
            raise ValueError('Validation must derive from the accepted traditional configuration')
        plans = []
        for candidate in candidates:
            profile = Path('configs')/f'{args.prefix}-u{candidate["update"]:05d}.yaml'
            if (ROOT/profile).exists():
                raise ValueError('Validation profile already exists: '+str(profile))
            plans.append((candidate, profile))
        profiles, requested = {}, {}
        for candidate, profile in plans:
            config = yaml.safe_load(yaml.safe_dump(original))
            config['visual_cycle'].update(grasp_execution='local_act',
                local_act_model=candidate['path'], local_act_manifest=candidate['deployment_manifest'])
            (ROOT/profile).write_text(yaml.safe_dump(config, sort_keys=False))
            key = str(candidate['update'])
            profiles[key] = str(profile)
            requested[key] = [f'{args.prefix}-u{candidate["update"]:05d}-s{seed}' for seed in seeds]
        batch = dict(status='running', provenance=provenance,
            registered_at=datetime.now(timezone.utc).isoformat(), validation_seeds=seeds,
            candidate_rounds=requested, profiles=profiles, runs=[],
            frozen_source_label=requested[str(candidates[0]['update'])][0],
            final_test_read=False)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.pending.json')
        temporary.write_text(json.dumps(batch, indent=2)+'\n')
        temporary.replace(args.output)

    save()
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    for candidate in candidates:
        key = str(candidate['update'])
        for seed, label in zip(seeds, batch['candidate_rounds'][key], strict=True):
            if any(row['label'] == label for row in batch['runs']):
                continue
            quota = usage_guard.latest()
            batch['latest_quota'] = quota
            if quota['reached']:
                batch['status'] = 'quota_stop'; save(); return 1
            pipeline_path = STORE/(label+'-pipeline.json')
            if pipeline_path.exists():
                raise ValueError('Existing attempt needs outcome reconciliation; never rerun silently: '+label)
            batch.update(status='running', current_label=label)
            save()
            command = [sys.executable, str(ROOT/'tools/speed_trial.py'), label,
                '--seed', str(seed), '--parcels', '4', '--speed-profile', batch['profiles'][key],
                '--controller', baseline['controller'], '--cpu-layout', baseline['cpu_layout'], '--record-video']
            if label != batch['frozen_source_label']:
                command.extend(['--frozen-from', batch['frozen_source_label']])
            rc = subprocess.call(command)
            pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
            outcome = dict(label=label, seed=seed, checkpoint_update=candidate['update'], returncode=rc)
            try:
                outcome['acceptance'] = inspect(label, ROOT, STORE, 650.)
                outcome['runtime_matches_baseline'] = all(
                    outcome['acceptance'][field] == baseline[field]
                    for field in ('runtime', 'controller', 'cpu_layout', 'record_video'))
                source = next((row['acceptance'] for row in batch['runs']
                    if row['label'] == batch['frozen_source_label'] and 'acceptance' in row),
                    outcome['acceptance'])
                outcome['frozen_policy_matches'] = (
                    outcome['acceptance']['manifest'] == source['manifest']
                    and outcome['acceptance']['profile'] == batch['profiles'][key])
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
            if (outcome.get('runtime_matches_baseline') is not True
                    or outcome.get('frozen_policy_matches') is not True):
                batch['status'] = 'policy_or_runtime_mismatch'; save(); return 1
            if any(pipeline.get('stages', {}).get(stage) != 0 for stage in ('analysis', 'audit', 'archive')):
                batch['status'] = 'evidence_incomplete'; save(); return 1
            from prune_policy_copies import prune, prune_analysis
            from compact_evidence import compact
            prune(label); prune_analysis(label); compact(label)
            print(json.dumps(dict(label=label, checkpoint_update=candidate['update'],
                passed=outcome.get('acceptance', {}).get('passed', False))), flush=True)
    batch.update(status='complete', current_label=None)
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

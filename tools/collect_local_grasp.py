"""Preregister and execute 60 expert rounds after frozen traditional acceptance.

One outcome per seed is retained, including failures. Test-seed recordings are
sealed for final evaluation; the dataset converter cannot consume that split.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys

from evidence_storage import EVIDENCE_ROOT, STORE, ARCHIVE_ROOT, require_storage
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--expert-verification', required=True)
    parser.add_argument('--development-registry', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', args.prefix):
        parser.error('Invalid prefix')
    require_storage()
    import yaml
    acceptance = json.loads(args.acceptance.read_text())
    if (acceptance.get('passed') is not True or len(acceptance.get('runs', [])) < 10
            or not all(acceptance['checks'].values())):
        raise ValueError('A passing preregistered traditional acceptance report is required')
    for recorded in acceptance['runs']:
        if inspect(recorded['label'], ROOT, STORE, 650.) != recorded:
            raise ValueError('Accepted traditional evidence changed')
    baseline = acceptance['runs'][0]
    expert = inspect(args.expert_verification, ROOT, STORE, 650.)
    # Recording can add wall time, but must retain complete, guarded execution.
    required = {k:v for k,v in expert['checks'].items()
                if k not in ('under_wall_limit', 'old_timing_under_wall_limit')}
    if not all(required.values()):
        raise ValueError('Expert recorder needs a complete guarded development round')
    for field in ('manifest', 'runtime', 'controller', 'cpu_layout', 'record_video'):
        if expert[field] != baseline[field]:
            raise ValueError('Expert collection changed frozen runtime: '+field)
    frozen = STORE/'policies'/expert['label']
    base_profile = yaml.safe_load((frozen/baseline['profile']).read_text())
    expert_profile = yaml.safe_load((frozen/expert['profile']).read_text())
    if expert_profile['performance'].pop('record_grasp_skills', False) is not True:
        raise ValueError('Expert profile must enable the raw skill recorder')
    base_profile['performance'].pop('record_grasp_skills', None)
    if expert_profile != base_profile or base_profile['visual_cycle']['grasp_execution'] != 'traditional':
        raise ValueError('Expert profile must differ only by enabling recording')
    verification_raw = ARCHIVE_ROOT/expert['label']/'output/grasp-skills/summary.json'
    raw = json.loads(verification_raw.read_text())
    if raw.get('raw_recording_valid') is not True:
        raise ValueError('Development recording is invalid')
    # Use the installed learning environment, not an assumed host NumPy/ROS
    # installation. The NVMe directory is mounted without copying raw images.
    mount = json.loads((EVIDENCE_ROOT/'traditional-local-act-1107/training-storage-mount.json').read_text())
    pid = int(subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Pid}}',
                                      mount['container']], text=True).strip())
    if pid != mount['pid'] or Path(mount['host_source']).resolve() != EVIDENCE_ROOT.resolve():
        raise ValueError('Restore the verified NVMe mount after any algorithm-container restart')
    recording_in_container = Path(mount['container_target'])/verification_raw.parent.relative_to(EVIDENCE_ROOT)
    alignment_source = (ROOT/'scene1/skill_alignment.py').read_text()
    alignment_code = alignment_source + '\nimport sys\n_, rows = align_recording(sys.argv[1])\nprint(json.dumps([{k:v for k,v in row.items() if k != "frames"} for row in rows]))\n'
    segments = json.loads(subprocess.check_output(['docker', 'exec', '-i', mount['container'],
        '/root/miniforge3/envs/kls_dev/bin/python', '-', str(recording_in_container)],
        input=alignment_code, text=True))
    if len(segments) != 4 or not all(segment['accepted'] for segment in segments):
        raise ValueError('Development recording must produce four causally aligned successful skills')
    receipt = dict(relative_path=str(args.acceptance.resolve().relative_to(EVIDENCE_ROOT.resolve())),
                   sha256=digest(args.acceptance))
    if args.resume:
        registry = json.loads(args.registry.read_text())
        if (registry['traditional_acceptance'] != receipt or registry['expert_verification'] != expert
                or registry['prefix'] != args.prefix
                or registry['alignment_verification_sha256'] != hashlib.sha256(alignment_source.encode()).hexdigest()):
            raise ValueError('Cannot change a preregistered collection')
        if registry.get('status') == 'complete':
            return 0
    else:
        if args.registry.exists():
            raise ValueError('Registry exists; use --resume without changing seeds or policy')
        development = json.loads(args.development_registry.read_text())
        excluded = set(development['excluded_from_new_holdouts']) | set(development['development_seeds'])
        for path in STORE.glob('*-pipeline.json'):
            seed = json.loads(path.read_text()).get('seed')
            if type(seed) is int:
                excluded.add(seed)
        entropy = secrets.randbits(128)
        rng = random.Random(entropy)
        seeds = []
        while len(seeds) < 60:
            seed = rng.randrange(2**31)
            if seed not in excluded and seed not in seeds:
                seeds.append(seed)
        splits = dict(train=seeds[:40], validation=seeds[40:50], test=seeds[50:])
        registry = dict(schema_version=1, status='registered', prefix=args.prefix,
            traditional_acceptance=receipt, expert_verification=expert,
            alignment_verification_sha256=hashlib.sha256(alignment_source.encode()).hexdigest(),
            development_seeds=sorted(excluded), seed_splits=splits,
            seed_split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
            sampling=dict(method='uniform rejection without replacement', entropy=entropy), runs=[])
        args.registry.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.registry.with_suffix('.pending.json')
        temporary.write_text(json.dumps(registry, indent=2)+'\n')
        temporary.replace(args.registry)

    save()
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import usage_guard
    for split, seeds in registry['seed_splits'].items():
        for seed in seeds:
            if any(run['seed'] == seed for run in registry['runs']):
                continue
            quota = usage_guard.latest()
            registry['latest_quota'] = quota
            if quota['reached']:
                registry['status'] = 'quota_stop'; save(); return 1
            label = f'{args.prefix}-{split}-s{seed}'
            pipeline_path = STORE/(label+'-pipeline.json')
            if pipeline_path.exists():
                raise RuntimeError('Interrupted seed needs its outcome reconciled; never rerun or replace silently: '+label)
            registry.update(status='running', current_seed=seed, current_label=label)
            save()
            rc = subprocess.call([sys.executable, str(ROOT/'tools/speed_trial.py'), label,
                '--seed', str(seed), '--parcels', '4', '--frozen-from', expert['label'],
                '--speed-profile', expert['profile'], '--controller', expert['controller'],
                '--cpu-layout', expert['cpu_layout'], '--record-video'])
            pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
            result = dict(seed=seed, label=label, split=split, returncode=rc,
                          recording_relative=None, outcome='failed')
            try:
                result['acceptance'] = inspect(label, ROOT, STORE, 650.)
                result['outcome'] = 'complete' if result['acceptance']['checks']['complete'] else 'failed'
                result['frozen_match'] = all(result['acceptance'][field] == expert[field]
                    for field in ('manifest', 'profile', 'runtime', 'controller', 'cpu_layout', 'record_video'))
            except (OSError, KeyError, ValueError) as exc:
                result['inspection_error'] = str(exc)
            raw_dir = ARCHIVE_ROOT/label/'output/grasp-skills'
            if (raw_dir/'summary.json').is_file():
                result['recording_relative'] = str(raw_dir.relative_to(EVIDENCE_ROOT))
            registry['runs'].append(result)
            save()
            # No next reset until restoration is evidenced, even after failure.
            cpu_path = STORE/(label+'-cpu-allocation.json')
            cpu = json.loads(cpu_path.read_text()) if cpu_path.exists() else {}
            if pipeline.get('stages', {}).get('restore') != 0 or cpu.get('restored') is not True:
                registry['status'] = 'restoration_incomplete'; save(); return 1
            if pipeline.get('quota_stop'):
                registry['status'] = 'quota_stop'; save(); return 1
            # A policy failure is a retained expert outcome. An unarchived
            # run or a changed executable/runtime is not another sample from
            # the preregistered expert, even if restoration itself succeeded.
            if (result.get('frozen_match') is not True
                    or any(pipeline.get('stages', {}).get(stage) != 0
                           for stage in ('analysis', 'audit', 'archive'))):
                registry['status'] = 'evidence_or_runtime_mismatch'; save(); return 1
            if result.get('acceptance', {}).get('checks', {}).get('pipeline'):
                from prune_policy_copies import prune, prune_analysis
                from compact_evidence import compact
                prune(label); prune_analysis(label); compact(label)
            print(json.dumps(dict(seed=seed, split=split, outcome=result['outcome'])), flush=True)
    registry.update(status='complete', current_seed=None, current_label=None)
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Freeze an isolated speed candidate and run one guarded official reset."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'diagnostics/path-tracking-continuation-20261023'
from evidence_storage import STORE, require_storage
ALGO = 'kuavo-ros-noetic-s12-ready'
PYTHON = '/root/miniforge3/envs/kls_dev/bin/python'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--parcels', type=int, choices=[1,2,4], default=4)
    ap.add_argument('--speed-profile', default='configs/speed_baseline.yaml')
    ap.add_argument('--controller', choices=['origin_pd', 'arm_weight', 'wbc_warmstart', 'wbc_reduced'], default='origin_pd',
                    help='Registered isolated controller/task configuration; baseline restored after every trial')
    ap.add_argument('--frozen-from', help='Reuse an existing byte-verified policy snapshot for regression/holdouts')
    ap.add_argument('--record-video', action='store_true')
    from cpu_allocation import PROFILES
    ap.add_argument('--cpu-layout', choices=['original', *PROFILES], default='original')
    a = ap.parse_args()
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.label):
        ap.error('Invalid label')
    if a.frozen_from and not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.frozen_from):
        ap.error('Invalid frozen source label')
    require_storage()
    if shutil.disk_usage(STORE.parent).free < 5_000_000_000:
        raise RuntimeError('At least 5 GB free space is required on the evidence disk')
    STORE.mkdir(exist_ok=True)
    from cpu_allocation import allocation
    with open('/tmp/scene1-speed-trial.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another scene1 trial is still running or restoring') from None
        with allocation(a.cpu_layout, a.label):
            return execute(a)


def execute(a):
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.label):
        raise ValueError('Invalid label')
    if a.frozen_from and not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.frozen_from):
        raise ValueError('Invalid frozen source label')
    sys.path.insert(0, str(DRIVER))
    import usage_guard
    import environment
    environment.wait_idle(environment.ALGO)
    environment.wait_idle(environment.SIM)
    quota = usage_guard.latest()
    if quota['reached']:
        raise RuntimeError(('Live quota threshold reached', quota))
    from archive_idle_ros_logs import archive_if_needed
    archive_if_needed(a.label, environment)
    policy = STORE/'policies'/a.label
    source = ROOT
    expected = None
    if a.frozen_from:
        source = STORE/'policies'/a.frozen_from
        expected = json.loads((source.parent/(a.frozen_from+'-manifest.json')).read_text())
        files = []
        for name, digest in expected.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe frozen manifest path')
            if 'tests' in relative.parts or relative.name.startswith('test_'):
                raise ValueError('Legacy snapshot contains tests; freeze the current strategy as a new baseline')
            path = source/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(('Frozen policy changed', name))
            files.append(path)
    else:
        files = [ROOT/'run_scene1.py', ROOT/'run.sh', ROOT/'config.yaml', ROOT/'calibrate_servo.py']
        for directory in ['scene1', 'configs']:
            files += [p for p in (ROOT/directory).rglob('*') if p.is_file()
                      and '__pycache__' not in p.parts and p.suffix != '.pyc']
    policy.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for p in files:
        relative = p.relative_to(source)
        dst = policy/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        manifest[str(relative)] = hashlib.sha256(dst.read_bytes()).hexdigest()
    if expected is not None and manifest != expected:
        raise RuntimeError('Frozen snapshot copy failed verification')
    manifest_path = policy.parent/(a.label+'-manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    project = '/root/scene1_speed/'+a.label
    subprocess.run(['docker','exec',ALGO,'mkdir','-p',project],check=True)
    subprocess.run(['docker','cp',str(policy)+'/.',ALGO+':'+project],check=True)
    receipt = dict(label=a.label, seed=a.seed, parcels=a.parcels, quota=quota,
                   speed_profile=a.speed_profile, controller=a.controller,
                   frozen_from=a.frozen_from, cpu_layout=a.cpu_layout, record_video=a.record_video, stages={})
    output = STORE/(a.label+'-pipeline.json')
    def run(stage, command):
        receipt['current_stage'] = stage
        descriptor, scratch_name = tempfile.mkstemp(prefix=a.label+'-'+stage+'-', suffix='.log')
        os.close(descriptor)
        scratch = Path(scratch_name)
        receipt.setdefault('pending_logs', {})[stage] = str(scratch)
        output.write_text(json.dumps(receipt, indent=2))
        with scratch.open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        destination = STORE/(a.label+'-'+stage+'.log')
        shutil.copy2(scratch, destination)
        if hashlib.sha256(scratch.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError(('Stage log archive mismatch; local copy retained', str(scratch)))
        scratch.unlink()
        receipt['pending_logs'].pop(stage)
        receipt['stages'][stage] = rc
        output.write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(stage=stage, returncode=rc, label=a.label)), flush=True)
        return rc
    try:
        if run('switch',[sys.executable,str(DRIVER/'switch_wrist_pd.py'),a.controller]):
            raise RuntimeError('Controller switch failed')
        run('driver',[sys.executable,str(DRIVER/'full_experiment.py'),a.controller,a.label,
                      '--policy','executor_pad_dense','--policy-root',str(policy),
                      '--manifest',str(manifest_path),'--container-project',project,
                      '--speed-profile',a.speed_profile,
                      '--seed',str(a.seed),'--parcels',str(a.parcels),
                      '--max-steps',str({1:5000,2:9000,4:16800}[a.parcels])] + (['--record-video'] if a.record_video else []))
        d=DRIVER/a.label
        if usage_guard.latest()['reached']:
            receipt['quota_stop'] = True
            return 1  # Raw evidence was retained by the driver; restore immediately.
        if (d/'output/steps.jsonl').is_file() and any((d/name).is_file() for name in ('chain.jsonl','chain.jsonl.gz')):
            run('analysis',[sys.executable,str(DRIVER/'analyze_single.py'),a.label])
            run('audit',[sys.executable,str(DRIVER/'audit_run.py'),a.label])
        if (d/'experiment.json').is_file():
            run('archive',[sys.executable,str(DRIVER/'archive_completed.py'),a.label])
    finally:
        run('restore',[sys.executable,str(DRIVER/'switch_wrist_pd.py'),'baseline'])
        receipt['current_stage']='done'
        output.write_text(json.dumps(receipt,indent=2))
    return int(any(receipt['stages'].values()))

if __name__ == '__main__':
    raise SystemExit(main())

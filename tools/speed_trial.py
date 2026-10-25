"""Freeze an isolated speed candidate and run one guarded official reset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT/'diagnostics/path-tracking-continuation-20261023'
STORE = Path('/media/krist/KINGSTON/scene1-speed-20261025')
ALGO = 'kuavo-ros-noetic-s12-ready'
PYTHON = '/root/miniforge3/envs/kls_dev/bin/python'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--parcels', type=int, choices=[1,2,4], default=4)
    ap.add_argument('--speed-profile', default='configs/speed_baseline.yaml')
    ap.add_argument('--skip-tests', action='store_true')
    ap.add_argument('--frozen-from', help='Reuse an existing byte-verified policy snapshot for regression/holdouts')
    a = ap.parse_args()
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.label):
        ap.error('Invalid label')
    if a.frozen_from and not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.frozen_from):
        ap.error('Invalid frozen source label')
    sys.path.insert(0, str(DRIVER))
    import usage_guard
    import environment
    environment.wait_idle(environment.ALGO)
    environment.wait_idle(environment.SIM)
    quota = usage_guard.latest()
    if quota['reached']:
        raise RuntimeError(('10% quota stop', quota))
    policy = STORE/'policies'/a.label
    policy.mkdir(parents=True, exist_ok=False)
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
            path = source/relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(('Frozen policy changed', name))
            files.append(path)
    else:
        files = [ROOT/'run_scene1.py', ROOT/'run.sh', ROOT/'config.yaml', ROOT/'calibrate_servo.py']
        for directory in ['scene1', 'configs', 'tests']:
            files += [p for p in (ROOT/directory).rglob('*') if p.is_file()
                      and '__pycache__' not in p.parts and p.suffix != '.pyc']
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
                   speed_profile=a.speed_profile, frozen_from=a.frozen_from, stages={})
    output = STORE/(a.label+'-pipeline.json')
    def run(stage, command):
        receipt['current_stage'] = stage
        output.write_text(json.dumps(receipt, indent=2))
        with (STORE/(a.label+'-'+stage+'.log')).open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        receipt['stages'][stage] = rc
        output.write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(stage=stage, returncode=rc, label=a.label)), flush=True)
        return rc
    try:
        if not a.skip_tests and run('tests',['docker','exec','-w',project,ALGO,PYTHON,
                                           '-m','unittest','discover','-s','tests']):
            raise RuntimeError('Tests failed')
        if run('switch',[sys.executable,str(DRIVER/'switch_wrist_pd.py'),'origin_pd']):
            raise RuntimeError('Controller switch failed')
        run('driver',[sys.executable,str(DRIVER/'full_experiment.py'),'origin_pd',a.label,
                      '--policy','executor_pad_dense','--policy-root',str(policy),
                      '--manifest',str(manifest_path),'--container-project',project,
                      '--speed-profile',a.speed_profile,
                      '--seed',str(a.seed),'--parcels',str(a.parcels),
                      '--max-steps',str({1:5000,2:9000,4:16800}[a.parcels])])
        d=DRIVER/a.label
        if usage_guard.latest()['reached']:
            receipt['quota_stop'] = True
            return 1  # Raw evidence was retained by the driver; restore immediately.
        if (d/'output/steps.jsonl').is_file() and (d/'chain.jsonl').is_file():
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

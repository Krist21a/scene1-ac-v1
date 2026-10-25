"""Atomic controller-library switch; preserve source and require fully idle containers."""
import argparse
import datetime
import hashlib
import json
import subprocess
import experiment as e

ap = argparse.ArgumentParser()
ap.add_argument('version', choices=['baseline', 'candidate'])
args = ap.parse_args()
e.wait_idle(e.ALGO)
e.wait_idle(e.SIM)
base = e.HERE / 'baseline/controller-library/libhumanoid_controllers.so'
candidate = e.HERE / 'candidate-wrist-pd/libhumanoid_controllers.so'
target = '/root/kuavo_ws/devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = base if args.version == 'baseline' else candidate
allowed = [sha(base)] + ([sha(candidate)] if candidate.exists() else [])
subprocess.run(['docker', 'cp', str(source), e.SIM + ':/tmp/tracking-selected-wrist-pd'], check=True)
code = '''import pathlib,hashlib,os,shutil,json
t=pathlib.Path(%r);s=pathlib.Path('/tmp/tracking-selected-wrist-pd')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=sha(t);assert old in %r,old
assert sha(s)==%r
tmp=t.with_name(t.name+'.tracking-switch');shutil.copy2(s,tmp);tmp.chmod(t.stat().st_mode);os.replace(tmp,t)
print(json.dumps(dict(before=old,after=sha(t))))
''' % (target, allowed, sha(source))
result = json.loads(e.command('docker', 'exec', e.SIM, 'python3', '-c', code))
receipt = dict(at=datetime.datetime.now().astimezone().isoformat(), factor='wrist torque PD',
               version=args.version, file=target, **result,
               source_note='Workspace source untouched; isolated object and library compiled with original cached recipe')
with (e.HERE / 'switches.jsonl').open('a') as f:
    f.write(json.dumps(receipt) + '\n')
print(json.dumps(receipt, indent=2))

"""Reversible shared executable switch; requires both containers fully idle."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import experiment as e

ap=argparse.ArgumentParser()
ap.add_argument('version',choices=['baseline','candidate'])
args=ap.parse_args()
e.wait_idle(e.ALGO);e.wait_idle(e.SIM)
base=e.HERE/'baseline/vr-reference/humanoid_VR_hand_control'
candidate=e.HERE/'candidate-reference-origin/humanoid_VR_hand_control'
selected=base if args.version=='baseline' else candidate
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=sha(selected)
allowed=[sha(base)]+([sha(candidate)] if candidate.exists() else [])
target='/root/kuavo_ws/devel/.private/humanoid_interface_ros/lib/humanoid_interface_ros/humanoid_VR_hand_control'
subprocess.run(['docker','cp',str(selected),e.SIM+':/tmp/tracking-selected-controller'],check=True)
code='''import pathlib,hashlib,os,shutil,json
t=pathlib.Path(%r);s=pathlib.Path('/tmp/tracking-selected-controller')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=sha(t);assert old in %r,old
assert sha(s)==%r
tmp=t.with_name(t.name+'.tracking-switch');shutil.copy2(s,tmp);tmp.chmod(t.stat().st_mode);os.replace(tmp,t)
print(json.dumps(dict(before=old,after=sha(t),file=str(t))))
'''%(target,allowed,expected)
receipt=json.loads(e.command('docker','exec',e.SIM,'python3','-c',code))
receipt.update(version=args.version,at=datetime.datetime.now().astimezone().isoformat(),
               source_note='Workspace source untouched; candidate compiled from preserved isolated source with build-manifest recipe')
with (e.HERE/'switches.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
print(json.dumps(receipt,indent=2))

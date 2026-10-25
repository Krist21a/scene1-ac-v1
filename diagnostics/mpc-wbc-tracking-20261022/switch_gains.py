"""Atomic task.info switch, with idle check and exact before/after hashes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import experiment as e

ap=argparse.ArgumentParser();ap.add_argument('version',choices=['baseline','candidate']);args=ap.parse_args()
e.wait_idle(e.ALGO);e.wait_idle(e.SIM)
relative='src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/task.info'
base=e.HERE/'baseline'/relative;candidate=e.HERE/'candidate-wrist-gains/task.info'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=e.WS/relative;old=sha(target);assert old in [sha(base),sha(candidate)]
source=base if args.version=='baseline' else candidate
tmp=target.with_name('task.info.tracking-switch');shutil.copy2(source,tmp);os.replace(tmp,target)
r=dict(at=datetime.datetime.now().astimezone().isoformat(),version=args.version,file=str(target),before=old,after=sha(target))
with (e.HERE/'switches.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps(r,indent=2))

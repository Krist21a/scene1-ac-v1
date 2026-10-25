"""Read compact live trial progress without copying the complete timing log."""
import argparse
import json
from pathlib import Path
import re
import subprocess

REMOTE = r'''
import json,sys
from pathlib import Path
from collections import Counter
paths=sorted((Path('/root/scene1_speed')/sys.argv[1]/'outputs').glob('*/performance.jsonl'))
if not paths:
    print(json.dumps(dict(state='initializing_or_archived')));raise SystemExit
first=last=None;n=complete=0;completions=[];counts=Counter();times=Counter()
for line in paths[-1].open():
    try:r=json.loads(line)
    except ValueError:continue
    first=first or r;last=r;n+=1
    complete+=r.get('phase')=='visual_cycle_complete'
    if r.get('phase')=='visual_cycle_complete':
        completions.append(round(r['started_monotonic_s']+r['cycle_s']-first['started_monotonic_s'],3))
    for k,v in r['spans_s'].items():times[k]+=v
    counts[str(r.get('control_period_s','original'))]+=1
print(json.dumps(dict(steps=n,formal_elapsed_s=last['started_monotonic_s']+last['cycle_s']-first['started_monotonic_s'],
                     last_phase=last.get('phase'),completed_cycles=complete,completion_wall_s=completions,period_counts=counts,
                     spans_s={k:round(v,3) for k,v in times.items()})) if n else '{}')
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    label = parser.parse_args().label
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        parser.error('Invalid label')
    store = Path('/media/krist/KINGSTON/scene1-speed-20261025')
    pipeline = store/(label+'-pipeline.json')
    if pipeline.exists():
        state = json.loads(pipeline.read_text())
        print(json.dumps(dict(stage=state['current_stage'],stages=state['stages'])))
    subprocess.run(['docker','exec','kuavo-ros-noetic-s12-ready',
                    '/root/miniforge3/envs/kls_dev/bin/python','-c',REMOTE,label],check=True)

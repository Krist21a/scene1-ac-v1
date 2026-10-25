"""Read only the latest complete policy row; never send a robot command."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
run = root/sys.argv[1]
state = json.loads((run/'experiment.json').read_text())
print(json.dumps({k:state.get(k) for k in ['label','status','started_at','ended_at','error','policy_summary']}, ensure_ascii=False))
code = '''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);runs=sorted(root.glob('*/steps.jsonl'))
if not runs:print('No policy step yet');sys.exit()
p=runs[-1]
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-262144));lines=f.read().splitlines()
for line in reversed(lines):
 try:r=json.loads(line)
 except ValueError:continue
 c=r.get('cartesian_alignment',{});out=dict(run=p.parent.name,step=r['step'],state=c.get('state'),stage=c.get('stage'),regrasp_stage=c.get('regrasp_stage'),phase_steps=c.get('phase_steps'),motion=r.get('motion'),retreat_error_m=c.get('retreat_error_m'),sweep=c.get('command_sweep',{}).get('filtered'),goal_check=c.get('retreat_goal_check'),settlement=c.get('settlement',{}).get('state'));print(json.dumps(out));break
'''
subprocess.run(['docker','exec','kuavo-ros-noetic-s12-ready','python3','-c',code,state['policy_root']+'/outputs'],check=True)

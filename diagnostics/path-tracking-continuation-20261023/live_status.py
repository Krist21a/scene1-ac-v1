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
checkpoint=p.parent/'visual-cycle-checkpoint.json'
completed=json.loads(checkpoint.read_text()).get('controller',{}).get('completed') if checkpoint.is_file() else None
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-262144));lines=f.read().splitlines()
for line in reversed(lines):
 try:r=json.loads(line)
 except ValueError:continue
 c=r.get('cartesian_alignment',{});goal=c.get('retreat_goal_check') or {};out=dict(run=p.parent.name,step=r['step'],state=c.get('state'),stage=c.get('stage'),regrasp_stage=c.get('regrasp_stage'),phase_steps=c.get('phase_steps'),completed_parcels=c.get('completed_parcels',completed),orientation_error_degrees=c.get('orientation_error_degrees'),motion=r.get('motion'),release_pose_hold=c.get('release_pose_hold'),release_preparation=c.get('release_preparation'),retreat_error_m=c.get('retreat_error_m'),sweep=c.get('command_sweep',{}).get('filtered'),goal_check={k:goal.get(k) for k in ['certified','solver_method','nominal_steps','position_residual_m','orientation_residual_degrees']} if goal else None,settlement=c.get('settlement',{}).get('state'));print(json.dumps(out));break
'''
if state.get('ended_at') and (run/'output/steps.jsonl').is_file():
    subprocess.run([sys.executable,'-c',code,str(run)],check=True)
else:
    subprocess.run(['docker','exec','kuavo-ros-noetic-s12-ready','python3','-c',code,state['policy_root']+'/outputs'],check=True)

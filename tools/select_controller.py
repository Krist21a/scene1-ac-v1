"""Install exact A+C or rollback in idle simulator; dry-run unless --apply."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('version',choices=['ac','baseline']);p.add_argument('--container',default='kuavo_challenge_container_GPU_046e4301');p.add_argument('--apply',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1]
def run(*args):return subprocess.check_output(args,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'evidence/control-runtime-manifest.json').read_text())['files']
base='/root/kuavo_ws/'
items=[('controllers/'+('A' if a.version=='ac' else 'baseline')+'/humanoid_VR_hand_control',base+'devel/.private/humanoid_interface_ros/lib/humanoid_interface_ros/humanoid_VR_hand_control'),('controllers/'+('C' if a.version=='ac' else 'baseline')+'/libhumanoid_controllers.so',base+'devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'),('controllers/baseline-config/dynamic_qr.info',base+'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/dynamic_qr.info'),('controllers/baseline-task/task.info',base+'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/task.info')]
allowed=list(m.values())+['225d217d47741f088830d0a7c038e8a76c465d9948f4a56f194b3524f58677de','3f32e8d8858a9352b674d15cdb6578fdf8e4b96429ccfe46fffd81a881d95d17']
for source,target in items:
 expected=m.get(source,allowed[-2] if source.endswith('dynamic_qr.info') else allowed[-1]);assert sha(r/source)==expected,source
 print(source,'->',target,expected)
if not a.apply:
 print('Dry run only; use --apply with both simulation and policy stopped.');raise SystemExit()
# Match original deployment's conservative idle guard before changing any file.
for container in [a.container,'kuavo-ros-noetic-s12-ready']:
 lines=run('docker','exec',container,'ps','-eo','pid,args').splitlines()[1:]
 active=[x for x in lines if not any(x.strip().endswith(s) for s in [' zsh',' /bin/bash',' ps -eo pid,args'])]
 assert not active,('Container not idle',container,active)
plans=[]
for i,(source,target) in enumerate(items):
 staged='/tmp/scene1-ac-v1-switch-'+str(i)
 subprocess.run(['docker','cp',str(r/source),a.container+':'+staged],check=True)
 plans.append((staged,target,sha(r/source)))
code='''import pathlib,hashlib,os,shutil,json
plans=json.loads(%r);allowed=json.loads(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for s,t,h in plans:
 assert sha(pathlib.Path(s))==h
 assert sha(pathlib.Path(t)) in allowed,(t,'Unknown current artifact')
for s,t,h in plans:
 t=pathlib.Path(t);tmp=t.with_name(t.name+'.ac-v1-switch')
 shutil.copy2(s,tmp);tmp.chmod(t.stat().st_mode);os.replace(tmp,t)
 assert sha(t)==h
print(json.dumps({t:h for s,t,h in plans}))
'''%(json.dumps(plans),json.dumps(allowed))
print(run('docker','exec',a.container,'python3','-c',code))

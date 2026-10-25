import sys,json,subprocess,time
p='/home/krist/kuavo_course/scene1_act_optimization/diagnostics/path-tracking-continuation-20261023/'
time.sleep(float(sys.argv[2]) if len(sys.argv)>2 else 0.)
r=subprocess.run(['sudo','-n','python3',p+'live_status.py',sys.argv[1]],text=True,capture_output=True,check=True)
for line in r.stdout.splitlines():
 try:v=json.loads(line)
 except ValueError:print(line);continue
 keys=['step','state','stage','completed_parcels','status','ended_at','error','policy_summary']
 out={k:v[k] for k in keys if k in v}
 if v.get('release_preparation'):out['preparation']={k:v['release_preparation'][k] for k in ['tracking_error_rad','settled_movement_rad','stable_steps']}
 if v.get('sweep'):out['sweep_clear']=v['sweep']['clear']
 print(json.dumps(out))

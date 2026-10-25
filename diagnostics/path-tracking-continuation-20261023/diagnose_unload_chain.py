import json,sys,pathlib,math
p=pathlib.Path(__file__).resolve().parent;r=p/sys.argv[1];rows=[json.loads(s) for s in (r/'output/steps.jsonl').read_text().splitlines()];release=next(x for x in reversed(rows) if 'release_reference_transfer' in x['cartesian_alignment']);selected=[x for x in rows if release['step']-3<=x['step']];best={x['step']:{} for x in selected};topics=['/humanoid_controller/optimizedState_wbc_mrt_origin','/humanoid_controller/optimizedInput_wbc_mrt_origin','/humanoid_controller/optimizedState_mrt_origin','/humanoid_controller/optimizedInput_mrt_origin','/humanoid_controller/wbc_planned_joint_acc','/sensors_data_raw','/humanoid_mpc_target_arm']
for line in (r/'chain.jsonl').open():
 a=json.loads(line);k=a.get('topic')
 if k not in topics:continue
 t=a['callback_monotonic_ns']/1e9
 if t<selected[0]['send_monotonic_s']-1:continue
 if t>selected[-1]['send_monotonic_s']+1:continue
 for row in selected:
  d=abs(t-row['send_monotonic_s']);b=best[row['step']]
  if k in b and d>=b[k]['distance_s']:continue
  m=a['message']
  if k=='/sensors_data_raw':v=dict(q=m['joint_data']['joint_q'][20:27],v=m['joint_data']['joint_v'][20:27],tau=m['joint_data']['joint_torque'][20:27])
  elif k=='/humanoid_mpc_target_arm':v=m
  else:v=m['data'][-14:] if 'wbc' in k else m['data'][-8:]
  b[k]=dict(distance_s=d,value=v)
out=dict(label=r.name,release_step=release['step'],rows=[dict(step=x['step'],sent=x['sent_action'][8:15],streams=best[x['step']]) for x in selected]);(p/(r.name+'-unload-chain.json')).write_text(json.dumps(out,indent=2)+'\n')
for x in out['rows']:
 print('step',x['step'])
 for k,v in x['streams'].items():
  if k!='/humanoid_mpc_target_arm':print(k.split('/')[-1],v['value'])

import cProfile, io, json, pstats, sys, time
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent
policy=root/sys.argv[1]
sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from run_scene1 import make_residual_assist
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'retreat-preflight-input.json').read_text())
r,previous=f['row'],f['previous'];cert=r['cartesian_alignment']['retreat_goal_check'];base=np.array(f['release']['proposed_action'])
low=np.r_[kin.limits[0][:,0],0,kin.limits[1][:,0],0]
high=np.r_[kin.limits[0][:,1],1,kin.limits[1][:,1],1]
a=make_residual_assist(cfg,kin,low,high,10.)
a.wrist_config=cfg['wrist_assist']
a.weighing_target=np.r_[cert['observed_parcel']['position_base_m'][:2],r['rgbd_table_z_base_m']]
a.offset=np.array(previous['proposed_action'])-base;a.observe_command(previous['sent_action'])
a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=4
snapshot=dict(candidates=[cert['observed_parcel']],table_z_base_m=r['rgbd_table_z_base_m'],stamp=r['rgbd_stamp'])
p=cProfile.Profile();p.enable();start=time.monotonic()
command,info=a._pending_action(np.array(r['observation_state']),snapshot)
elapsed=time.monotonic()-start;p.disable()
stream=io.StringIO();pstats.Stats(p,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(25)
(root/(sys.argv[1]+'-preflight-profile.txt')).write_text(stream.getvalue())
out=dict(elapsed_s=elapsed,command=command.tolist(),certificate=info['retreat_goal_check'])
(root/(sys.argv[1]+'-preflight-replay.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(elapsed_s=elapsed,certified=out['certificate']['certified'],steps=out['certificate']['nominal_steps'])))
print(stream.getvalue())

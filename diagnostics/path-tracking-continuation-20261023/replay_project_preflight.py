"""Time the complete preflight on actual initial retreat observations."""
import json,sys,time
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-project';sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from run_scene1 import make_residual_assist
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'early-retreat-block-input.json').read_text());rows=f['opening_rows']
i=next(i for i,r in enumerate(rows) if r['cartesian_alignment'].get('retreat_goal_check'))
r=rows[i];previous=rows[i-1];cert=r['cartesian_alignment']['retreat_goal_check'];base=np.array(f['release']['proposed_action'])
a=make_residual_assist(cfg,kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1],10.)
a.wrist_config=cfg['wrist_assist'];a.weighing_target=np.r_[cert['observed_parcel']['position_base_m'][:2],r['rgbd_table_z_base_m']]
a.offset=np.array(previous['proposed_action'])-base;a.observe_command(previous['sent_action'])
a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=4
snapshot=dict(candidates=[cert['observed_parcel']],table_z_base_m=r['rgbd_table_z_base_m'],stamp=r['rgbd_stamp'])
start=time.monotonic();command,info=a._pending_action(np.array(r['observation_state']),snapshot)
out=dict(elapsed_s=time.monotonic()-start,command=command.tolist(),certificate=info['retreat_goal_check'])
(root/'project-preflight-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(elapsed_s=out['elapsed_s'],certified=out['certificate']['certified'],steps=out['certificate']['nominal_steps'])))

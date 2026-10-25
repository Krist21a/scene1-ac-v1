"""Kinematic feasibility after a hypothetical pose-preserving reference transfer."""
import json,sys,time
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent;policy=root/'candidate-policy-executor-mpc-hold';sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from run_scene1 import make_residual_assist
from scene1.command_sweep import sampled_sweep
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
k=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
low=np.r_[k.limits[0][:,0],0,k.limits[1][:,0],0];high=np.r_[k.limits[0][:,1],1,k.limits[1][:,1],1]
results=[]
for x in json.loads((root/'mpc-hold-replay-inputs.json').read_text()):
    observed=np.array(x['release']['observation_state']);base=np.array(x['release']['proposed_action'])
    base[8:12]=x['previous_sent'][8:12];base[12:15]=observed[12:15];base[15]=0.
    a=make_residual_assist(cfg,k,low,high,10.);a.wrist_config=cfg['wrist_assist']
    a.weighing_target=np.r_[x['target']['position_base_m'][:2],x['plane']]
    a.observe_command(base);a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=4
    observed[15]=0.
    snapshot=dict(candidates=[x['target']],table_z_base_m=x['plane'],stamp=x['stamp'])
    out=dict(label=x['label'],initial_reference_sweep=sampled_sweep(k,observed[8:15],base[8:15],x['target'],x['plane']))
    start=time.monotonic()
    try:
        command,info=a._pending_action(observed,snapshot)
        out.update(certified=info['retreat_goal_check']['certified'],nominal_steps=info['retreat_goal_check']['nominal_steps'],
            position_residual_m=info['retreat_goal_check']['position_residual_m'])
    except Exception as error:
        out.update(certified=False,error=repr(error))
    out['elapsed_s']=time.monotonic()-start;results.append(out)
out=dict(cases=results,limitations='Pose held at recorded release observation with new reference already applied. Later observed parcel geometry reused. This is kinematic screening only, not an unload dynamics simulation or success evidence.')
(root/'mpc-hold-replay.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

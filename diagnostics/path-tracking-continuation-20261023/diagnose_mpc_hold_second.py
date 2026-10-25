"""Replay exact second release and record the first failed nominal segment."""
import json, sys
from pathlib import Path
import numpy as np
import yaml
root=Path(__file__).resolve().parent
policy=root/'candidate-policy-executor-mpc-hold'
sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from run_scene1 import make_residual_assist
import scene1.residual as residual
import scene1.retreat_projection as projection_module
if '--omit-fixed-constraint' in sys.argv:
    import inspect
    source=inspect.getsource(projection_module.project_retreat_reference)
    assert 'for a in fractions])' in source
    source=source.replace('for a in fractions])','for a in fractions[1:]])')
    exec(source,projection_module.__dict__)
    residual.project_retreat_reference=projection_module.project_retreat_reference
from scene1.command_sweep import sampled_sweep
from scene1.retreat_projection import FingerSeparation
cfg=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
k=UrdfArmKinematics(description=(policy/'tests/fixtures/release_robot.urdf').read_text())
f=json.loads((root/'mpc-hold-second-failure-input.json').read_text())
release,previous,failed=f['release'],f['previous'],f['failed']
base=np.array(release['proposed_action']); measured=np.array(failed['observation_state'])
target=failed['rgbd_candidates'][0];plane=failed.get('rgbd_table_z_base_m',previous['rgbd_table_z_base_m'])
low=np.r_[k.limits[0][:,0],0,k.limits[1][:,0],0];high=np.r_[k.limits[0][:,1],1,k.limits[1][:,1],1]
a=make_residual_assist(cfg,k,low,high,10.);a.wrist_config=cfg['wrist_assist']
a.weighing_target=np.r_[release['cartesian_alignment']['weighing_surface']['position_base_m'][:2],plane]
a.offset=np.array(previous['proposed_action'])-base;a.observe_command(previous['sent_action'])
a._begin('short_retreat',1,base,extent=np.array([.11,.04,.1]),surface={});a.pending['steps']=4
snapshot=dict(candidates=failed['rgbd_candidates'],table_z_base_m=plane,stamp=previous['rgbd_stamp'])
records=[];original=residual.project_retreat_reference
solves=[];original_minimize=projection_module.minimize
def solve_record(*args,**kwargs):
    result=original_minimize(*args,**kwargs)
    solves.append(dict(nominal_index=len(records),success=bool(result.success),message=str(result.message),
        x=result.x.tolist(),iterations=int(result.nit),minimum_constraint=float(np.min(kwargs['constraints'][0]['fun'](result.x))),
        initial_minimum_constraint=float(np.min(kwargs['constraints'][0]['fun'](args[1])))))
    return result
projection_module.minimize=solve_record
def wrapped(*args,**kwargs):
    kin,obs,proposed,ref,prev,lower,upper,step,t,p,mc=args
    sep=FingerSeparation(t,p)
    row=dict(index=len(records),measured=obs.tolist(),proposed=proposed.tolist(),reference=ref.tolist(),
        previous=prev.tolist(),step=step,measured_gap=sep.gaps(kin.matrix(obs[8:15],1)).tolist(),
        proposed_sweep=sampled_sweep(kin,obs[8:15],proposed[8:15],t,p),
        reference_sweep=sampled_sweep(kin,obs[8:15],ref[8:15],t,p))
    try:
        chosen,audit=original(*args,**kwargs);row.update(chosen=chosen.tolist(),audit=audit);return chosen,audit
    except Exception as e:
        row['error']=repr(e);raise
    finally:records.append(row)
residual.project_retreat_reference=wrapped
out=dict(target=target,plane=plane,initial_measured_tcp=k.matrix(measured[8:15],1).tolist(),
         previous_tcp=k.matrix(np.array(previous['sent_action'])[8:15],1).tolist())
try:
    command,info=a._pending_action(measured,snapshot);out.update(command=command.tolist(),info=info)
except Exception as e:out['error']=repr(e)
out['records']=records
out['solves']=solves
name='mpc-hold-second-fixed-constraint-replay.json' if '--omit-fixed-constraint' in sys.argv else 'mpc-hold-second-failure-replay.json'
(root/name).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(error=out.get('error'),records=len(records),solves=solves,
    certificate={key:out.get('info',{}).get('retreat_goal_check',{}).get(key) for key in ['certified','nominal_steps','position_residual_m']}),indent=2))

"""Full perception/certificate regression on the newly retained missing frame."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import yaml

root = Path(__file__).resolve().parent
policy = root/'candidate-policy-executor-spatial'
sys.path.insert(0, str(policy))
from run_scene1 import read_manipulation_rgbd
from scene1.rgbd import metric_candidates
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.release_retreat import axis_retreat_goal, RetreatPreflightError
from scene1.retreat_certificate import certify_retreat_execution

fixture=json.loads((root/'blocked-geometry-input.json').read_text())
config=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
k=UrdfArmKinematics(description=Path('/tmp/scene1_robot.urdf').read_text())
with np.load(root/'blocked-release-sensors.npz') as raw:
    snapshot={key:raw[key].copy() for key in ['rgb','depth_m','intrinsic','camera_to_base']}
snapshot['candidates'],snapshot['table_z_base_m']=metric_candidates(
    snapshot['rgb'],snapshot['depth_m'],snapshot['intrinsic'],snapshot['camera_to_base'],
    config['cartesian_alignment'],stride=2)
snapshot.update(stamp=fixture['failed']['rgbd_stamp'],candidate_spatial_components=False)
low=np.r_[np.full(7,-3.14),0.,np.full(7,-3.14),0.]
high=np.r_[np.full(7,3.14),1.,np.full(7,3.14),1.]
assist=ResidualAssist(config['visual_residual'],k,low,high)
assist.wrist_config=config['wrist_assist']
assist.weighing_target=np.r_[fixture['pad_xy'],snapshot['table_z_base_m']]
transfer=fixture['release']['cartesian_alignment']['release_reference_transfer']
base=np.array(fixture['last']['proposed_action']);base[8:15]=transfer['original_reference_rad']
assist.offset=np.array(fixture['last']['proposed_action'])-base
assist.observe_command(fixture['last']['sent_action'])
assist.unwinding_placement=bool(np.max(abs(assist.offset[8:15]))>config['visual_residual']['max_joint_offset_rad'])
assist.pending=dict(kind='short_retreat',steps=12)
def read(**kw):
    out=snapshot.copy()
    out['candidates'],out['table_z_base_m']=metric_candidates(
        out['rgb'],out['depth_m'],out['intrinsic'],out['camera_to_base'],config['cartesian_alignment'],
        stride=kw['candidate_stride'],spatial_components=kw['spatial_components'])
    out['candidate_spatial_components']=kw['spatial_components']
    return out
recovered=read_manipulation_rgbd(SimpleNamespace(read=read),assist,object(),None,config)
measured=np.array(fixture['failed']['observation_state'])
mc=dict(alpha=config['motion']['smoothing_alpha'],max_velocity=config['cartesian_alignment']['approach_joint_velocity_rad_s'],period=.1)
def validator(goal,parcel):
    return certify_retreat_execution(assist,base,measured,goal,parcel,recovered['table_z_base_m'],128,mc)
try:
    goal,audit=axis_retreat_goal(k,measured,base,assist.low,assist.high,assist.offset,
        config['wrist_assist']['max_regrasp_joint_offset_rad'],assist.unwinding_placement,recovered,
        assist.weighing_target[:2],[.11,.04,.1],.06,.07,.014,execution_validator=validator)
except RetreatPreflightError as error:
    audit=error.retreat_preflight_audit
report=dict(original_candidates=snapshot['candidates'],recovered_candidates=recovered['candidates'],
            same_depth_pixels=bool(np.array_equal(snapshot['depth_m'],recovered['depth_m'])),
            same_plane=snapshot['table_z_base_m']==recovered['table_z_base_m'],certificate=audit,
            commands_sent=False)
(root/'blocked-geometry-certificate.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['certified','reason','nominal_steps','position_residual_m','orientation_residual_degrees']},indent=2))

"""Counterfactual release hold; actual failed measured collision stays rejected.

Run in the numerical container next to the registered inputs. This exercises
the implemented release and rate limiter but makes no dynamics claim.
"""
from pathlib import Path
import sys
import json
import numpy as np
import yaml

root = Path(__file__).resolve().parent
policy = root/'candidate-policy-executor-release'
sys.path.insert(0, str(policy))
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.retreat_certificate import certify_retreat_execution
from scene1.release_retreat import axis_retreat_goal, RetreatPreflightError
from scene1.motion import MotionFilter

fixture = json.loads((root/'blocked-geometry-input.json').read_text())
geometry = json.loads((root/'blocked-geometry-certificate.json').read_text())
config = yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kinematics = UrdfArmKinematics(description=Path('/tmp/scene1_robot.urdf').read_text())
release = fixture['release']
transfer = release['cartesian_alignment']['release_reference_transfer']
measured = np.array(release['observation_state'])
base = np.array(release['proposed_action'])
base[8:15] = transfer['previous_command_rad']
base[15] = .98
reference = base.copy()
reference[8:15] = transfer['original_reference_rad']
assist = ResidualAssist(config['visual_residual'], kinematics,
    np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],
    np.r_[np.full(7,3.14),1,np.full(7,3.14),1])
assist.wrist_config = config['wrist_assist']
plane = -.08365308327559412
assist.weighing_target = np.r_[fixture['pad_xy'],plane]
assist.geometry[1] = dict(extent=np.array([.11,.04,.1]))
assist._view_ready = lambda *args: True  # Already accepted in the retained release.
assist._begin('weighing',1,base,stage='release',reference_base=reference,
              requested_open=0.,surface={},hold_grip=.98)
held, info = assist._pending_action(measured,dict(candidates=release['rgbd_candidates'],
    table_z_base_m=release['rgbd_table_z_base_m']))
motion = MotionFilter(assist.low,assist.high,1.,.6)
motion.reset(base)
commands = []
for index in range(12):
    sent, audit = motion.apply(held,.1)
    commands.append(dict(motion=audit,joints=sent[8:15].tolist()))
assist.observe_command(sent)
measured = sent.copy()
parcel = geometry['certificate']['observed_parcel']
motion_config = dict(alpha=1.,max_velocity=.6,period=.1)
def validator(goal,target):
    return certify_retreat_execution(assist,assist.pending['base'],measured,goal,
        target,plane,128,motion_config)
try:
    goal, audit = axis_retreat_goal(kinematics,measured,assist.pending['base'],assist.low,
        assist.high,assist.offset,config['wrist_assist']['max_regrasp_joint_offset_rad'],False,
        dict(candidates=[parcel],stamp=fixture['failed']['rgbd_stamp']),
        assist.weighing_target[:2],[.11,.04,.1],.06,.07,.014,execution_validator=validator)
except RetreatPreflightError as error:
    audit = error.retreat_preflight_audit
out = dict(transfer=info['release_reference_transfer'],filter_commands=commands,
    nominal_after_hold=audit,actual_failed_frame_still_rejected=not geometry['certificate']['certified'],
    limitation='Counterfactual perfect-tracking hold from real release feedback with later observed parcel; not a replay of actual dynamics. Actual failed measured frame remains rejected.')
(root/'physical-release-replay.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(certified=audit['certified'],steps=audit['nominal_steps'],reason=audit['reason'])))

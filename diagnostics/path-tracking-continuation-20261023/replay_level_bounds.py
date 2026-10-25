"""Replay supported level control with original and physical-only offset bounds.

The feedback response and retained static error are sensitivity scenarios, not
an identified dynamics model. No commands are sent.
"""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

root=Path(__file__).resolve().parent
policy=root/'candidate-policy-executor-physical'
sys.path.insert(0,str(policy))
from scene1.kinematics import UrdfArmKinematics
from scene1.residual import ResidualAssist
from scene1.motion import MotionFilter
from scene1.command_sweep import command_sweep_audit

fixture=json.loads((root/'level-bound-input.json').read_text())
config=yaml.safe_load((policy/'configs/visual_cycle_four_parcels.yaml').read_text())
kin=UrdfArmKinematics(description=Path('/tmp/scene1_robot.urdf').read_text())
last,entry=fixture['last'],fixture['entry']
context=last['cartesian_alignment']['command_sweep']
target=context['target'];plane=context['plane_m']
origin=kin.matrix(np.array(entry['observation_state'])[8:15],1)[:3,3]
major=np.array(target['major_axis_xy'])
major/=np.linalg.norm(major)
forward=major if major@(np.array(target['position_base_m'])[:2]-origin[:2])>=0 else -major
closing=np.r_[-forward[1],forward[0],0.]
pitch=np.deg2rad(config['wrist_assist']['regrasp_pitch_degrees'])
zaxis=-np.r_[forward*np.cos(pitch),-np.sin(pitch)]
rotation=Rotation.from_matrix(np.column_stack((closing,np.cross(zaxis,closing),zaxis)))
offset=np.array(last['cartesian_alignment']['offset_rad'])
base=np.array(last['proposed_action']);base[8:15]-=offset
report=[]
for mode,cap in [('old_phase_cap',.5235987756),('physical_joint_bounds',2*np.pi)]:
 for keep_bias in [False,True]:
  assist=ResidualAssist(config['visual_residual'],kin,np.r_[np.full(7,-3.14),0,np.full(7,-3.14),0],np.r_[np.full(7,3.14),1,np.full(7,3.14),1])
  assist.wrist_config=copy.deepcopy(config['wrist_assist'])
  assist.wrist_config['max_regrasp_joint_offset_rad']=cap
  assist.weighing_target=np.array(target['position_base_m'])
  assist.wrist_projector=object()
  assist.offset[8:15]=offset
  assist._wrist_measure=lambda *a,**kw:None
  assist._begin('pregrasp',1,base,target=copy.deepcopy(target),plane=plane,
      view_position=origin,grasp_rotation=rotation,regrasp_stage='level',surface={},extent=np.array([.11,.04,.1]))
  measured=np.array(last['observation_state'])
  bias=measured-np.array(last['sent_action']) if keep_bias else np.zeros(16)
  motion=MotionFilter(assist.low,assist.high,1.,.6);motion.reset(np.array(last['sent_action']))
  records=[]
  # Full original phase budget here is a fresh-run sensitivity, not a resume
  # granting time to the real timed-out trial.
  for i in range(config['wrist_assist']['max_level_regrasp_steps']):
   try:
    proposed,info=assist._pending_action(measured,dict(candidates=last['rgbd_candidates']))
   except RuntimeError as error:
    records.append(dict(error=str(error)));break
   previous=motion.previous.copy()
   sent,limit=motion.apply(proposed,.1)
   sweep=command_sweep_audit(kin,measured,proposed,sent,previous,assist.command_sweep_context)
   records.append(dict(step=i,angle=info['orientation_error_degrees'],position=np.hypot(info['xy_error_m'],info['z_error_m']),motion=limit,sweep=sweep))
   if limit['bounds_clipped'] or not sweep['proposed']['clear'] or not sweep['filtered']['clear']:break
   measured+=.25*(sent+bias-measured)
   if assist.pending['regrasp_stage']!='level':break
  report.append(dict(mode=mode,retained_static_error=keep_bias,level_complete=assist.pending['regrasp_stage']!='level',steps=len(records),records=records))
(root/'level-bound-replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([dict(mode=x['mode'],retained_bias=x['retained_static_error'],passed=x['level_complete'],steps=x['steps'],last_angle=x['records'][-1].get('angle')) for x in report],indent=2))

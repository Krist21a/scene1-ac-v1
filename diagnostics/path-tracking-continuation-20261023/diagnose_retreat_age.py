"""Replay the exact guard using raw feedback nearest the actual send instant."""
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'candidate-policy-executor-obb'))
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
from scene1.command_sweep import sampled_sweep
from scene1.retreat_projection import FingerSeparation

x = json.loads((root/'retreat-age-input.json').read_text())
k = UrdfArmKinematics(description=(root/'candidate-policy-executor-obb/tests/fixtures/release_robot.urdf').read_text())
r = x['row']
target = r['cartesian_alignment']['retreat_goal_check']['observed_parcel']
plane = r['rgbd_table_z_base_m']
separation = FingerSeparation(target, plane)
send = r['send_monotonic_s']
before = max((v for v in x['sensors'] if v['t'] <= send), key=lambda v: v['t'])
states = [('planning_observation', r['observation_state'][8:15]),
          ('raw_sensor_before_send', before['q'][7:14]),
          ('env_step_feedback', r['measured_state'][8:15]),
          ('next_failed_observation', x['failed']['observation_state'][8:15])]
out = dict(label=x['label'], step=r['step'],
    planning_receipt_age_at_send_s=send-r['observation_receipts']['/sensors_data_raw']['receipt_monotonic_s'],
    latest_background_receipt_age_at_send_s=r['sensor_age_before_send_s']['/sensors_data_raw'],
    nearest_raw_before_send_age_s=send-before['t'], cases=[])
origin = k.matrix(np.array(states[0][1]), 1)
for label, q in states:
    q = np.array(q)
    pose = k.matrix(q, 1)
    out['cases'].append(dict(label=label, q_rad=q.tolist(), tcp_m=pose[:3, 3].tolist(),
        translation_from_planning_observation_m=float(np.linalg.norm(pose[:3,3]-origin[:3,3])),
        original_box_clear=open_fingers_clear(pose,target,plane),
        finger_separation_gaps_m=separation.gaps(pose).tolist(),
        full_sweep_to_same_sent_command=sampled_sweep(k,q,np.array(r['sent_action'][8:15]),target,plane)))
out['limitations'] = ['Raw feedback uses callback arrival time; exact actuator application is unavailable.',
    'Uses the unchanged observed oriented parcel box and 3mm margin, not simulator object truth.',
    'This diagnoses freshness; it does not prove a particular dynamics model or remedy.']
(root/'retreat-age-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

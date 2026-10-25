"""Reproduce observed snapshots and sweep omissions without running ROS."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'candidate-policy-axis'))
from scene1.kinematics import UrdfArmKinematics
from scene1.release_retreat import axis_retreat_goal
from scene1.command_sweep import command_sweep_audit

k = UrdfArmKinematics(description=(HERE/'robot.urdf').read_text())
low = np.r_[k.limits[0][:, 0], 0., k.limits[1][:, 0], 0.]
high = np.r_[k.limits[0][:, 1], 1., k.limits[1][:, 1], 1.]
retreats = []
for item in json.loads((HERE/'replay-inputs.json').read_text()):
    f, row, release = item['fixture'], item['retreat'], item['release']
    transfer = release['cartesian_alignment'].get('release_reference_transfer')
    result = dict(label=f['label'], parcel=f['parcel'], step=row['step'])
    if not transfer:
        result['status'] = 'direction_only_original_reference_not_logged'
        pose = k.matrix(np.asarray(row['observation_state'])[8:15], 1)
        axis = pose[:2, 2]/np.linalg.norm(pose[:2, 2])
        near = [c for c in row['rgbd_candidates'] if np.linalg.norm(
            np.asarray(c['position_base_m'])[:2]-f['pad_xy']) < .12]
        result['current_candidates_on_pad'] = len(near)
        result['away_dot_m'] = float(axis@(pose[:2, 3]-np.asarray(near[0]['position_base_m'])[:2])) if len(near)==1 else None
    else:
        reference = np.array(row['act_action']); reference[8:15] = transfer['original_reference_rad']
        offset = np.array(release['proposed_action'])-reference
        try:
            goal, report = axis_retreat_goal(k, row['observation_state'], reference, low, high,
                offset, .5235987756, True,
                dict(candidates=row['rgbd_candidates'], stamp=row['rgbd_stamp']),
                np.array(f['pad_xy']), [.11, .04, .1], .06, .07, .014)
            result.update(status='bounded_goal_accepted', audit=report)
        except (RuntimeError, ValueError) as exc:
            result.update(status='rejected', error=str(exc))
    retreats.append(result)
sweeps = {}
for source in HERE.glob('*-sweep-inputs.json'):
    mismatch = []; total = 0
    for item in json.loads(source.read_text()):
        row = item['row']; ca = row['cartesian_alignment']
        if ca.get('regrasp_stage') == 'reobserve':
            continue
        total += 1
        audit = command_sweep_audit(k, row['observation_state'], row['proposed_action'],
                                   row['sent_action'], row['sent_action'],
                    dict(target=item['target'], plane=item['plane'], stage=ca['regrasp_stage']))
        if ca.get('open_fingers_clear') and not audit['filtered']['clear']:
            mismatch.append(dict(step=row['step'], audit=audit))
    sweeps[source.name] = dict(examined=total, legacy_clear_but_filtered_blocked=mismatch)
out = dict(retreats=retreats, sweeps=sweeps,
           limitations=['No dynamics or task success claim',
                        'Historical six cases lack logged original release reference: no budget feasibility claim',
                        'Bounded replay uses URDF limits; live preflight also intersects SDK action bounds',
                        'Sampled open inner fingers, not continuous or whole robot collision proof'])
(HERE/'path-replay.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(dict(retreats=retreats, sweep_counts={k:dict(examined=v['examined'],
    legacy_clear_but_filtered_blocked=len(v['legacy_clear_but_filtered_blocked'])) for k,v in sweeps.items()}), indent=2))

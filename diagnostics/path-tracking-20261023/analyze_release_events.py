"""Observed FK motion between weighing release and first retreat correction."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ap=argparse.ArgumentParser()
for name in ['steps','urdf','output','project']:
    ap.add_argument('--'+name,required=True)
a=ap.parse_args();sys.path.insert(0,a.project)
from scene1.kinematics import UrdfArmKinematics
k=UrdfArmKinematics(description=Path(a.urdf).read_text())
rows=[json.loads(s) for s in Path(a.steps).open()]
out=[]
for i,r in enumerate(rows):
    if r['cartesian_alignment']['state']!='weighing_release':
        continue
    segment=[r]
    for following in rows[i+1:]:
        info=following['cartesian_alignment']
        if info['state'] not in ['small_short_retreat','short_retreat_complete']:
            break
        segment.append(following)
        if 'retreat_error_m' in info:
            break
    reference_pose=k.matrix(r['observation_state'][8:15],1)
    reference_rotation=Rotation.from_matrix(reference_pose[:3,:3])
    records=[]
    for v in segment:
        pose=k.matrix(v['observation_state'][8:15],1)
        records.append(dict(step=v['step'],elapsed_to_send_s=v['send_monotonic_s']-r['send_monotonic_s'],
            translation_from_release_m=float(np.linalg.norm(pose[:3,3]-reference_pose[:3,3])),
            rotation_from_release_degrees=float(np.rad2deg((Rotation.from_matrix(pose[:3,:3])*reference_rotation.inv()).magnitude())),
            measured_gripper=v['observation_state'][15],
            first_retreat_command='retreat_error_m' in v['cartesian_alignment']))
    transfer=r['cartesian_alignment'].get('release_reference_transfer')
    previous=np.asarray(transfer['previous_command_rad'] if transfer else rows[max(0,i-1)]['sent_action'][8:15])
    q=np.asarray(r['observation_state'][8:15])
    out.append(dict(release_step=r['step'],release_reference_transfer=transfer,
        pre_release_observation_minus_previous_command_rad=(q-previous).tolist(),
        release_filtered_command_minus_observation_rad=(np.asarray(r['sent_action'][8:15])-q).tolist(),
        first_retreat_step=records[-1]['step'] if records[-1]['first_retreat_command'] else None,
        maximum_observed_translation_before_retreat_m=max(v['translation_from_release_m'] for v in records),
        maximum_observed_rotation_before_retreat_degrees=max(v['rotation_from_release_degrees'] for v in records),
        records=records))
report=dict(events=out,interpretation='FK from logged observation_state in robot base. Includes first retreat input observation before sending its new command; exact actuator application time missing. Initial command errors/poses vary across runs, so this is not a same-state causal estimate or static tracking bias.')
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:e[k] for k in ['release_step','first_retreat_step','maximum_observed_translation_before_retreat_m','maximum_observed_rotation_before_retreat_degrees']} for e in out],indent=2))

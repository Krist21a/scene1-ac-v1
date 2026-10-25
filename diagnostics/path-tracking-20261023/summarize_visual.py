"""Logged visual corrections, explicitly separated from total command travel."""
import argparse
import collections
import json
import math
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('steps');ap.add_argument('--output',required=True);a=ap.parse_args()
rows=[json.loads(l) for l in open(a.steps)]
result=collections.defaultdict(lambda:dict(offset_variation_rad=0.,offset_records=0,offset_pairs=0,receiver_lateral_variation_m=0.,receiver_records=0,receiver_pairs=0,pose_position_sum_m=0.,pose_angle_sum_rad=0.,pose_records=0))
previous=None
for r in rows:
    info=r['cartesian_alignment'];phase=info.get('state','unknown');out=result[phase]
    offset=info.get('offset_rad')
    if offset is not None:out['offset_records']+=1
    lateral=info.get('receiver_lateral_offset_m')
    if lateral is not None:out['receiver_records']+=1
    if previous and previous.get('state')==phase:
        old=previous.get('offset_rad')
        if offset is not None and old is not None and len(old)==len(offset):
            out['offset_pairs']+=1
            out['offset_variation_rad']+=math.sqrt(sum((x-y)**2 for x,y in zip(offset,old)))
        old=previous.get('receiver_lateral_offset_m')
        if lateral is not None and old is not None:
            out['receiver_pairs']+=1
            out['receiver_lateral_variation_m']+=abs(lateral-old)
    for error in info.get('pose_errors',[]):
        if error:
            out['pose_position_sum_m']+=error['position_m'];out['pose_angle_sum_rad']+=error['angle_rad'];out['pose_records']+=1
    previous=info
for v in result.values():
    if not v['offset_pairs']:v['offset_variation_rad']=None
    if not v['receiver_pairs']:v['receiver_lateral_variation_m']=None
    v['pose_position_mean_m']=v['pose_position_sum_m']/v['pose_records'] if v['pose_records'] else None
    v['pose_angle_mean_rad']=v['pose_angle_sum_rad']/v['pose_records'] if v['pose_records'] else None
report=dict(phases=dict(result),interpretation='Within-phase variation of logged offset_rad and receiver_lateral_offset_m only. Missing fields are missing coverage, not zero corrections. Pose error uses policy FK/goal, not independent TCP truth.')
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')

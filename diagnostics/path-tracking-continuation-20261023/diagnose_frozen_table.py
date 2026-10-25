"""Read-only RGB-D tabletop edge evidence for a retained held-out failure."""
import sys,json
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,sys.argv[1])
from scene1.visual_pickup import table_hull
from scene1.table_reference import table_boundary,table_frame,table_motion
p=Path(sys.argv[2]); first=int(sys.argv[3])
rows={r['step']:r for r in map(json.loads,(p/'steps.jsonl').read_text().splitlines())}
reference=None;out=[]
for file in sorted(p.glob('sensors-*.npz')):
    n=int(file.stem.split('-')[1])
    if n<first:continue
    s=dict(np.load(file));s['table_z_base_m']=rows[n]['rgbd_table_z_base_m']
    hull=table_hull(s);boundary=table_boundary(s);frame=table_frame(hull,s['table_z_base_m'],boundary)
    if reference is None and frame is not None:reference=frame
    item=dict(step=n,table_reference_hold=rows[n]['cartesian_alignment'].get('table_reference_hold',False),boundary_points=len(boundary),frame=frame,motion=table_motion(reference,frame))
    if hull is not None:
        rect=cv2.minAreaRect(hull);corners=cv2.boxPoints(rect);edges=np.roll(corners,-1,axis=0)-corners;lengths=np.linalg.norm(edges,axis=1)
        item.update(area_ratio=cv2.contourArea(hull)/np.prod(rect[1]),lengths=lengths,edges=[])
        for i,edge in enumerate(edges):
            normal=np.array([-edge[1],edge[0]])/lengths[i];selected=boundary[abs((boundary-corners[i])@normal)<.008]
            span=float(np.ptp(selected@(edge/lengths[i]))) if len(selected)>1 else 0.
            item['edges'].append(dict(points=len(selected),span_m=span,required_span_m=max(.075,.08*lengths[i]),corners=corners[[i,(i+1)%4]]))
    out.append(item)
print(json.dumps(dict(reference=reference,frames=out,interpretation='Recorded RGB-D only; no simulator object truth; original thresholds unchanged.'),default=lambda x:x.tolist(),indent=2))

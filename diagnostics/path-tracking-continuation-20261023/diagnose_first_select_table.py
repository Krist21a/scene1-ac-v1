import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/scene1_path_tracking_continuation_20261023/candidate-policy-executor-first-select')
from scene1.visual_pickup import table_hull
from scene1.table_reference import table_frame,table_boundary,table_motion
p=Path('/tmp/first-select-output'); rows={r['step']:r for r in map(json.loads,(p/'steps.jsonl').read_text().splitlines())};ref=None;out=[]
for f in sorted(p.glob('sensors-*.npz')):
 n=int(f.stem.split('-')[1]);r=rows[n]
 if n<138:continue
 s=dict(np.load(f));s['table_z_base_m']=r['rgbd_table_z_base_m'];h=table_hull(s);b=table_boundary(s);frame=table_frame(h,s['table_z_base_m'],b)
 if ref is None and frame is not None:ref=frame
 d=dict(step=n,boundary_points=len(b),frame=frame)
 if frame is not None:
  d.update(delta=frame['center']-ref['center'],size_change=frame['size']-ref['size'],motion=table_motion(ref,frame))
 out.append(d)
print(json.dumps(out,default=lambda a:a.tolist(),indent=2))

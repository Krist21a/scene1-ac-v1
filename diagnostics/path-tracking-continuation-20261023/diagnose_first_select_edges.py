exec(open('/tmp/diagnose_first_select_table.py').read().split('for f in sorted')[0])
import cv2
out=[]
for n in [138,540,560,580]:
 s=dict(np.load(p/('sensors-%04d.npz'%n)));s['table_z_base_m']=rows[n]['rgbd_table_z_base_m'];h=table_hull(s);b=table_boundary(s);rect=cv2.minAreaRect(h);corners=cv2.boxPoints(rect);edges=np.roll(corners,-1,axis=0)-corners;lens=np.linalg.norm(edges,axis=1);e=[]
 for i,edge in enumerate(edges):
  normal=np.array([-edge[1],edge[0]])/lens[i];sel=b[abs((b-corners[i])@normal)<.008];span=float(np.ptp(sel@(edge/lens[i]))) if len(sel)>1 else 0;e.append(dict(points=len(sel),span=span,required=max(.075,.08*lens[i]),corners=[corners[i].tolist(),corners[(i+1)%4].tolist()]))
 out.append(dict(step=n,area_ratio=cv2.contourArea(h)/np.prod(rect[1]),lengths=lens.tolist(),edges=e))
print(json.dumps(out,indent=2))

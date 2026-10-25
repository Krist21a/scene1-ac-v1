exec(open('/tmp/diagnose_first_select_table.py').read().split('for f in sorted')[0])
import cv2
from scipy.optimize import least_squares
s=dict(np.load(p/'sensors-0138.npz'));s['table_z_base_m']=rows[138]['rgbd_table_z_base_m'];ref=table_frame(table_hull(s),s['table_z_base_m'],table_boundary(s))
out=[]
for n in [540,560,580]:
 s=dict(np.load(p/('sensors-%04d.npz'%n)));s['table_z_base_m']=rows[n]['rgbd_table_z_base_m'];h=table_hull(s);b=table_boundary(s)
 rect=cv2.minAreaRect(h);corners=cv2.boxPoints(rect);edges=np.roll(corners,-1,axis=0)-corners;lens=np.linalg.norm(edges,axis=1)
 groups=[]
 for i,edge in enumerate(edges):
  normal=np.array([-edge[1],edge[0]])/lens[i];sel=b[abs((b-corners[i])@normal)<.008]
  if len(sel)<6 or np.ptp(sel@(edge/lens[i]))<max(.075,.08*lens[i]):continue
  mid=np.mean(sel,axis=0);local=(mid-ref['center'][:2])@ref['rotation'];axis=int(np.argmax(abs(local)/(ref['size']/2)));sign=np.sign(local[axis]);groups.append((axis,sign,sel))
 def residual(x):
  yaw=x[2];R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])@ref['rotation'];c=ref['center'][:2]+x[:2]
  return np.concatenate([((g-c)@R)[:,axis]-sign*ref['size'][axis]/2 for axis,sign,g in groups])
 fit=least_squares(residual,[0,0,0],loss='soft_l1',f_scale=.002,max_nfev=100)
 out.append(dict(step=n,edges=[(a,float(s),len(g)) for a,s,g in groups],delta=fit.x.tolist(),residual95=float(np.quantile(abs(residual(fit.x)),.95)),within_original_30mm=bool(np.linalg.norm(fit.x[:2])<=.03)))
print(json.dumps(out,indent=2))

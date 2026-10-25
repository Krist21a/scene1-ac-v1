import sys,json
from pathlib import Path
import numpy as np,cv2
sys.path.insert(0,'/root/scene1_path_tracking_continuation_20261023/candidate-policy-executor-first-bin')
from scene1.visual_cycle import cloud,bin_rim,rim_edges_supported
s=dict(np.load('/tmp/first-bin-0320.npz'));s['stamp']=320.;s['table_z_base_m']=-0.08373318791389783;p,h=cloud(s);z=s['table_z_base_m'];sel=(h[:,0]>=40)&(h[:,0]<=90)&(h[:,1]>=80)&(p[:,2]>z+.12)&(p[:,2]<z+.5)&(p[:,0]>.15)&(p[:,0]<1.2)&(abs(p[:,1])<.8);p=p[sel];top=np.percentile(p[:,2],95);rim=p[abs(p[:,2]-top)<.009];rect=cv2.minAreaRect(rim[:,:2].astype(np.float32));print(json.dumps(dict(plane=z,points=len(p),top=top,rim_points=len(rim),size=rect[1],supported=rim_edges_supported(rim,cv2.boxPoints(rect)),height_percentiles=np.percentile(p[:,2],[80,90,95,98,99,100]).tolist(),rim=bin_rim(s)),default=lambda x:x.tolist(),indent=2))
from scipy.spatial import cKDTree
pairs=cKDTree(rim).query_pairs(.02);parent=list(range(len(rim)))
def root(i):
 while parent[i]!=i:
  parent[i]=parent[parent[i]];i=parent[i]
 return i
for a,b in pairs:parent[root(a)]=root(b)
groups={}
for i in range(len(rim)):groups.setdefault(root(i),[]).append(i)
out=[]
for ix in sorted(groups.values(),key=len,reverse=True):
 g=rim[ix];rect=cv2.minAreaRect(g[:,:2].astype(np.float32));out.append(dict(points=len(g),center=np.mean(g,axis=0).tolist(),size=rect[1],supported=rim_edges_supported(g,cv2.boxPoints(rect))))
print(json.dumps(out,indent=2))
g=rim[max(groups.values(),key=len)];cs=cv2.boxPoints(cv2.minAreaRect(g[:,:2].astype(np.float32)))
for a,b in zip(cs,np.roll(cs,-1,axis=0)):
 t=b-a;length=np.linalg.norm(t);t/=length;rel=g[:,:2]-a;along=rel@t;near=(abs(rel@[-t[1],t[0]])<.009)&(along>=0)&(along<=length);ss=np.sort(along[near]);gg=np.split(ss,np.flatnonzero(np.diff(ss)>.02)+1);print('edge',len(ss),[(len(v),float(np.ptp(v)) if len(v) else 0) for v in gg])

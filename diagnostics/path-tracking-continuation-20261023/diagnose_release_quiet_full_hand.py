"""Offline public-robot mesh check against unchanged observed inflated parcel.

This is diagnostic only; no scene truth or simulator internals are read.
The live collision guard is not changed by this script.
"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'candidate-policy-executor-release-quiet'))
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
description=(root/'candidate-policy-executor-release-quiet/tests/fixtures/release_robot.urdf').read_text()
kinematics=UrdfArmKinematics(description=description)
urdf=ET.fromstring(description)
by_child={j.find('child').attrib['link']:j for j in urdf.findall('joint')}
raw=(root/'inner_finger_fine_new.STL').read_bytes()
count=int(np.frombuffer(raw,dtype='<u4',count=1,offset=80)[0])
triangles=np.frombuffer(raw,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=count,offset=84)
vertices=np.unique(triangles['v'].reshape(-1,3),axis=0)
vertices=vertices[ConvexHull(vertices).vertices]
sync=json.loads((root/'release-quiet-synchronized-robot.json').read_text())
angle=sync['nearest']['/gripper/state']['record']['message']['position'][1]
fingers=[];finger_triangles=[]
for link in urdf.findall('link'):
    name=link.attrib['name']
    if not name.startswith('right_gripper_'):continue
    mesh=link.find('visual/geometry/mesh')
    if mesh is None:continue
    path=root/'public-2f85'/Path(mesh.attrib['filename']).name
    raw=path.read_bytes();count=int(np.frombuffer(raw,dtype='<u4',count=1,offset=80)[0]);data=np.frombuffer(raw,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=count,offset=84)
    frame=name;chain=[]
    while frame!='zarm_r7_link':
        joint=by_child[frame];chain.append(joint);frame=joint.find('parent').attrib['link']
    matrix=np.eye(4);matrix[:3,3]=-kinematics.tool_xyz
    for joint in reversed(chain):
        attrs=joint.find('origin').attrib;origin=np.eye(4)
        origin[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in attrs.get('rpy','0 0 0').split()]).as_matrix()
        origin[:3,3]=[float(v) for v in attrs.get('xyz','0 0 0').split()];matrix=matrix@origin
        if joint.attrib['type']!='fixed' or 'inner_knuckle_joint' in joint.attrib['name']:
            rot=np.eye(4);rot[:3,:3]=Rotation.from_rotvec(np.array([1.,0.,0.])*angle).as_matrix();matrix=matrix@rot
    tri=data['v']@matrix[:3,:3].T+matrix[:3,3];finger_triangles.append(tri)
    v=np.unique(tri.reshape(-1,3),axis=0);fingers.append(v[ConvexHull(v).vertices])

fixture=json.loads((root/'release-quiet-failure-input.json').read_text())
component=json.loads((root/'release-quiet-component-diagnostic.json').read_text())
c=component['candidate'];points=np.array(c['debug_points']);pixels=np.array(c['debug_pixels'])
pose=kinematics.matrix(np.array(sync['nearest']['/sensors_data_raw']['record']['message']['joint_data']['joint_q'])[20:27],1)
rows=[];excluded=np.zeros(len(points),dtype=bool)
for finger in fingers:
 world=finger@pose[:3,:3].T+pose[:3,3];h=ConvexHull(world)
 d=np.max(points@h.equations[:,:3].T+h.equations[:,3],axis=1)
 excluded |= d<=.003
 rows.append(dict(within_3mm=int((d<=.003).sum()),within_5mm=int((d<=.005).sum()),nearest_signed_plane_distance=float(d.min()),pixels_within_5mm=pixels[d<=.005].tolist()))
out=dict(total_points=len(points),public_open_mesh=True,robot_joint_feedback=True,diagnostic_only=True,fingers=rows)
(root/'release-quiet-synchronized-robot-points.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

from scene1.rgbd import oriented_footprint
kept=points[~excluded];major,extent,center=oriented_footprint(kept[:,:2]);clean=dict(c)
clean.update(oriented_center_base_m=[*center.tolist(),c['oriented_center_base_m'][2]],major_axis_xy=major.tolist(),oriented_extent_m=extent.tolist(),top_z_base_m=float(np.percentile(kept[:,2],95)))
for key in list(clean):
 if key.startswith('debug'):del clean[key]
plane=component['plane'];out['diagnostic_exclusion_only']=dict(excluded=int(excluded.sum()),remaining=len(kept),candidate=clean,original_clear=open_fingers_clear(pose,c,plane),after_exclusion_clear=open_fingers_clear(pose,clean,plane),not_online=True)
(root/'release-quiet-synchronized-robot-points.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['diagnostic_exclusion_only']))

raw_sensor=np.load(root/'release-quiet-failed-sensors.npz');camera=raw_sensor['camera_to_base'][:3,3]
rays=points-camera;length=np.linalg.norm(rays,axis=1);ray_unit=rays/length[:,None]
nearest=np.full(len(points),np.inf)
for tri in finger_triangles:
 world=tri@pose[:3,:3].T+pose[:3,3];a=world[:,0];e1=world[:,1]-a;e2=world[:,2]-a
 for i,direction in enumerate(ray_unit):
  h=np.cross(direction,e2);det=np.sum(e1*h,axis=1);valid=np.abs(det)>1e-12
  inv=np.zeros(len(det));inv[valid]=1/det[valid];delta=camera-a
  u=np.sum(delta*h,axis=1)*inv;q=np.cross(delta,e1);v=q@direction*inv;t=np.sum(e2*q,axis=1)*inv
  valid &= (u>=0)&(v>=0)&(u+v<=1)&(t>0)
  if valid.any():nearest[i]=min(nearest[i],float(t[valid].min()))
distance=nearest-length;match=np.isfinite(nearest)&(np.abs(distance)<=.003)
out['triangle_ray_check']=dict(mesh_triangle_count=len(triangles),matches=int(match.sum()),pixels=pixels[match].tolist(),signed_depth_errors_m=distance[match].tolist(),ray_hits=int(np.isfinite(nearest).sum()))
kept=points[~match];major,extent,center=oriented_footprint(kept[:,:2]);rayclean=dict(clean);rayclean.update(oriented_center_base_m=[*center.tolist(),clean['oriented_center_base_m'][2]],major_axis_xy=major.tolist(),oriented_extent_m=extent.tolist(),top_z_base_m=float(np.percentile(kept[:,2],95)))
out['triangle_ray_check'].update(after_exclusion_clear=open_fingers_clear(pose,rayclean,plane),candidate=rayclean)
(root/'release-quiet-full-hand-ray-depth-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['triangle_ray_check']))

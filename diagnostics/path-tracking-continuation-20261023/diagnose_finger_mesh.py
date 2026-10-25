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
sys.path.insert(0,str(root/'candidate-policy-executor-context'))
from scene1.kinematics import UrdfArmKinematics
from scene1.clearance import open_fingers_clear
description=Path('/tmp/scene1_robot.urdf').read_text()
kinematics=UrdfArmKinematics(description=description)
urdf=ET.fromstring(description)
by_child={j.find('child').attrib['link']:j for j in urdf.findall('joint')}
raw=(root/'inner_finger_fine_new.STL').read_bytes()
count=int(np.frombuffer(raw,dtype='<u4',count=1,offset=80)[0])
triangles=np.frombuffer(raw,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),count=count,offset=84)
vertices=np.unique(triangles['v'].reshape(-1,3),axis=0)
vertices=vertices[ConvexHull(vertices).vertices]
fingers=[]
for side in ['left','right']:
    frame='right_gripper_'+side+'_inner_finger';chain=[]
    while frame!='zarm_r7_link':
        joint=by_child[frame];chain.append(joint);frame=joint.find('parent').attrib['link']
    matrix=np.eye(4);matrix[:3,3]=-kinematics.tool_xyz
    for joint in reversed(chain):
        attrs=joint.find('origin').attrib;origin=np.eye(4)
        origin[:3,:3]=Rotation.from_euler('xyz',[float(v) for v in attrs.get('rpy','0 0 0').split()]).as_matrix()
        origin[:3,3]=[float(v) for v in attrs.get('xyz','0 0 0').split()]
        matrix=matrix@origin
    points=vertices@matrix[:3,:3].T+matrix[:3,3]
    fingers.append(points)

fixture=json.loads((root/'hold-drift-input.json').read_text())
target=fixture['failed']['retreat_preflight_audit']['observed_parcel']
plane=fixture['last']['rgbd_table_z_base_m']
major=np.array(target['major_axis_xy']);major/=np.linalg.norm(major)
rotation=np.column_stack(([*major,0.],[-major[1],major[0],0.],[0,0,1]))
height=target['top_z_base_m']-plane
center=np.array(target['position_base_m']);center[2]=plane+height/2
half=np.r_[target['oriented_extent_m'],height]/2+.003
out=dict(robot_mesh='inner_finger_fine_new.STL',margin_m=.003,parcel=target,plane_m=plane,
    finger_bounds_tcp=[dict(low=x.min(0).tolist(),high=x.max(0).tolist()) for x in fingers],cases=[])
for label,action in [('release',fixture['release']['observation_state']),
                     ('failed',fixture['failed']['observation_state']),
                     ('command',fixture['last']['sent_action'])]:
    pose=kinematics.matrix(np.array(action)[8:15],1)
    collisions=[]
    for points in fingers:
        world=points@pose[:3,:3].T+pose[:3,3]
        local=(world-center)@rotation
        hull=ConvexHull(local)
        lp=linprog(np.zeros(3),A_ub=hull.equations[:,:3],b_ub=-hull.equations[:,3],
                   bounds=list(zip(-half,half)),method='highs')
        collisions.append(dict(convex_mesh_intersects=lp.success,solver_status=lp.status,
            intersection_point_parcel_frame=None if not lp.success else lp.x.tolist()))
    out['cases'].append(dict(label=label,box_guard_clear=open_fingers_clear(pose,target,plane),
        tcp_m=pose[:3,3].tolist(),fingers=collisions))
(root/'finger-mesh-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

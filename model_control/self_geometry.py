"""Identify visible robot surface points from public hand meshes and sensor FK.

Only the candidate-segmentation copy of depth is filtered. Original RGB/depth,
calibration, payload observations and acceptance evidence are never changed.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene1.rgbd import metric_candidates


class RobotSurfaceFilter:
    def __init__(self, kinematics, description, config, assets=None):
        self.k, self.config = kinematics, config
        assets = Path(assets or Path(__file__).resolve().parent/'assets/gripper_meshes')
        manifest = json.loads((assets/'manifest.json').read_text())
        hashes = {item['name']: item['sha256'] for item in manifest['assets']}
        urdf = ET.fromstring(description)
        self.urdf_sha256 = hashlib.sha256(description.encode()).hexdigest()
        joints = {joint.find('child').attrib['link']:joint for joint in urdf.findall('joint')}
        self.meshes = {}
        self.links = []
        for link in urdf.findall('link'):
            name = link.attrib['name']
            if not name.startswith(('left_gripper_', 'right_gripper_')):
                continue
            visual = link.find('visual')
            if visual is None or visual.find('geometry/mesh') is None:
                continue
            mesh = visual.find('geometry/mesh')
            filename = mesh.attrib['filename'].rsplit('/',1)[-1]
            if filename not in hashes:
                raise RuntimeError(f'Unregistered public gripper mesh: {filename}')
            if filename not in self.meshes:
                raw = (assets/filename).read_bytes()
                if hashlib.sha256(raw).hexdigest() != hashes[filename]:
                    raise RuntimeError(f'Public mesh hash mismatch: {filename}')
                count = int(np.frombuffer(raw,dtype='<u4',count=1,offset=80)[0])
                dtype = np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
                if len(raw) != 84+50*count:
                    raise ValueError('Expected binary STL with complete triangles')
                triangles = np.frombuffer(raw,dtype=dtype,count=count,offset=84)['v'].astype(float)
                samples=[]
                for triangle in triangles:
                    longest = max(np.linalg.norm(triangle[a]-triangle[b]) for a,b in ((0,1),(1,2),(2,0)))
                    n=max(1,int(np.ceil(longest/.001)))
                    if n > 500:
                        raise ValueError('Unexpected metre-scale robot hand mesh')
                    for a in range(n+1):
                        b=np.arange(n+1-a)
                        samples.append(triangle[0]+(a/n)*(triangle[1]-triangle[0])+(b/n)[:,None]*(triangle[2]-triangle[0]))
                points = np.unique(np.round(np.concatenate(samples),7),axis=0)
                self.meshes[filename] = dict(tree=cKDTree(points),low=points.min(axis=0),high=points.max(axis=0),sha256=hashes[filename])
            scale = np.array([float(x) for x in mesh.attrib.get('scale','1 1 1').split()])
            if scale.shape != (3,) or not np.allclose(scale,1):
                raise ValueError('Registered gripper meshes must use original metre scale')
            side=0 if name.startswith('left_') else 1
            stop = ('zarm_l7_link','zarm_r7_link')[side]
            frame=name; chain=[]
            while frame != stop:
                joint=joints[frame]; chain.append(joint); frame=joint.find('parent').attrib['link']
            transform=np.eye(4); transform[:3,3]=-self.k.tool_xyz
            moving=False
            for joint in reversed(chain):
                transform=transform@self._origin(joint.find('origin'))
                moving |= joint.attrib['type'] != 'fixed'
            transform=transform@self._origin(visual.find('origin'))
            # Only the base itself has no finger-linkage dependency. URDF
            # fixed follower links also move in the physical four-bar linkage.
            moving |= name not in ('left_gripper_base','right_gripper_base')
            self.links.append(dict(name=name,side=side,filename=filename,transform=transform,moving=moving))

    @staticmethod
    def _origin(node):
        attrs={} if node is None else node.attrib
        result=np.eye(4)
        result[:3,3]=[float(x) for x in attrs.get('xyz','0 0 0').split()]
        result[:3,:3]=Rotation.from_euler('xyz',[float(x) for x in attrs.get('rpy','0 0 0').split()]).as_matrix()
        return result

    def apply(self, snapshot, measured, commanded):
        joints=np.asarray(snapshot.get('rgb_joint_positions_rad'),dtype=float)
        if joints.shape != (29,) or not np.isfinite(joints).all():
            raise ValueError('Self-surface filtering requires RGB-aligned 29-joint sensor metadata')
        measured=np.asarray(measured,float);commanded=np.asarray(commanded,float)
        if measured.shape != (16,) or commanded.shape != (16,) or not np.isfinite(np.r_[measured,commanded]).all():
            raise ValueError('Self-surface filtering requires finite current/commanded gripper states')
        depth=snapshot['depth_m'];intrinsic=snapshot['intrinsic'];camera=snapshot['camera_to_base']
        v,u=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
        z=depth[v,u].ravel();u=u.ravel();v=v.ravel()
        points=np.column_stack(((u-intrinsic[0,2])*z/intrinsic[0,0],(v-intrinsic[1,2])*z/intrinsic[1,1],z))
        points=points@camera[:3,:3].T+camera[:3,3]
        valid=(z>.1)&np.isfinite(points).all(axis=1)
        masked=np.zeros(len(points),bool);counts={}
        valid_indices=np.flatnonzero(valid)
        point_tree=cKDTree(points[valid_indices])
        tools=[self.k.matrix(joints[13:20],0),self.k.matrix(joints[20:27],1)]
        for link in self.links:
            grip=link['side']*8+7
            # No temporally aligned finger joints are available in RGB-D.
            # During any closing/loaded state only the rigid palm is usable.
            if link['moving'] and max(measured[grip],commanded[grip]) > .05:
                continue
            matrix=tools[link['side']]@link['transform']
            mesh=self.meshes[link['filename']]
            center=(mesh['low']+mesh['high'])/2
            radius=np.linalg.norm((mesh['high']-mesh['low'])/2+.002)+1e-9
            world_center=matrix[:3,:3]@center+matrix[:3,3]
            nearby=point_tree.query_ball_point(world_center,radius)
            indices=valid_indices[np.asarray(nearby,dtype=int)]
            indices=indices[~masked[indices]]
            if not len(indices):continue
            local=(points[indices]-matrix[:3,3])@matrix[:3,:3]
            candidate=np.all(local>=mesh['low']-.002,axis=1)&np.all(local<=mesh['high']+.002,axis=1)
            indices=indices[candidate];local=local[candidate]
            if not len(indices):continue
            distance=mesh['tree'].query(local,distance_upper_bound=.002)[0]
            remove=indices[distance<=.002]
            masked[remove]=True;counts[link['name']]=int(len(remove))
        filtered=depth.copy();filtered[v[masked],u[masked]]=0.
        if masked.any():
            candidates,plane=metric_candidates(snapshot['rgb'],filtered,intrinsic,camera,self.config,
                stride=2,spatial_components=True,component_max_gap_m=snapshot.get('component_max_gap_m', .02))
        else:
            candidates,plane=snapshot['candidates'],snapshot['table_z_base_m']
        if abs(plane-snapshot['table_z_base_m']) > .003:
            raise RuntimeError('Robot surface removal changed observed table plane over 3 mm')
        result=dict(snapshot)
        result['unfiltered_candidates']=snapshot['candidates']
        result['candidates']=candidates
        result['robot_surface_filter']=dict(surface_distance_m=.002,mesh_sample_spacing_m=.001,sampling_stride=2,
            removed_points=int(masked.sum()),per_link=counts,urdf_sha256=self.urdf_sha256,
            image_joint_stamp_s=snapshot['rgb_joint_stamp_s'],rgb_stamp_s=snapshot['stamp'],
            original_depth_preserved=True,raw_candidate_count=len(snapshot['candidates']),filtered_candidate_count=len(candidates))
        return result

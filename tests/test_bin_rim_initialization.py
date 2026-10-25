import json
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from scene1.visual_cycle import bin_rim, cloud, rim_edges_supported, track_bin_rim


class BinRimInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=Path(__file__).parent/'fixtures'
        cls.fixture=json.loads((directory/'bin_rim_outlier_v53.json').read_text())
        cls.raw=np.load(directory/'bin_rim_outlier_v53.npz')

    def snapshot(self,step):
        row=next(x for x in self.fixture['frames'] if x['step']==step)
        return dict(row,**{k:self.raw[f'frame_{step}_'+k] for k in ('rgb','depth_m','intrinsic','camera_to_base')})

    def rim_points(self,snapshot):
        p,hsv=cloud(snapshot);plane=snapshot['table_z_base_m']
        selected=((hsv[:,0]>=40)&(hsv[:,0]<=90)&(hsv[:,1]>=80)&(p[:,2]>plane+.12)
                  &(p[:,2]<plane+.5)&(p[:,0]>.15)&(p[:,0]<1.2)&(abs(p[:,1])<.8))
        p=p[selected];top=np.percentile(p[:,2],95)
        return p[abs(p[:,2]-top)<.009]

    def test_real_stable_frames_keep_exact_original_rim(self):
        for step in (80,100,120):
            snap=self.snapshot(step)
            with patch('scene1.visual_cycle.rim_edges_supported',return_value=True):old=bin_rim(snap)
            new=bin_rim(snap)
            self.assertIsNotNone(new)
            for key in old:np.testing.assert_array_equal(new[key],old[key])
            self.assertTrue(.44<max(np.linalg.norm(np.roll(new['corners'],-1,axis=0)-new['corners'],axis=1))<.46)

    def test_real_isolated_point_and_stored_wrong_length_lack_four_edges(self):
        points=self.rim_points(self.snapshot(60));corners=cv2.boxPoints(cv2.minAreaRect(points[:,:2].astype(np.float32)))
        self.assertGreater(max(np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)),.70)
        self.assertFalse(rim_edges_supported(points,corners))
        stored=self.fixture['stored_failed_rim'];bad={k:np.asarray(v['array'],dtype=v['dtype']) if isinstance(v,dict) and 'array'in v else v for k,v in stored.items()}
        # The established-shape tracker intentionally preserves length under
        # occlusion. It cannot repair the false initial 641 mm rectangle.
        tracked=track_bin_rim(self.snapshot(80),bad)
        self.assertIsNotNone(tracked)
        self.assertFalse(rim_edges_supported(self.rim_points(self.snapshot(80)),tracked['corners']))
        self.assertGreater(max(np.linalg.norm(np.roll(tracked['corners'],-1,axis=0)-tracked['corners'],axis=1)),.63)

    def test_disconnected_outlier_cannot_stretch_initialized_bin(self):
        snap=self.snapshot(80);points=self.rim_points(snap)
        # Synthetic counterexample: one disconnected point in the original
        # admissible size range. The remaining points are actual raw evidence.
        outlier=np.array([points[:,0].min()-.15,points[:,1].mean(),np.median(points[:,2])])
        cloud_points=np.vstack((points,outlier));hsv=np.tile([60,200,200],(len(cloud_points),1)).astype(np.uint8)
        with patch('scene1.visual_cycle.cloud',return_value=(cloud_points,hsv)):
            result=bin_rim(snap)
            self.assertIsNotNone(result)
            self.assertTrue(rim_edges_supported(points,result['corners']))
            self.assertLess(max(np.linalg.norm(np.roll(result['corners'],-1,axis=0)-result['corners'],axis=1)),.46)
            self.assertGreater(result['corners'][:,0].min()-outlier[0],.14)

    def test_missing_or_short_side_is_not_filled_from_a_rectangle(self):
        corners=np.array([[0.,0.],[.4,0.],[.4,.3],[0.,.3]])
        edges=[np.column_stack((np.linspace(a,b,40),np.zeros(40))) for a,b in zip(corners,np.roll(corners,-1,axis=0))]
        self.assertTrue(rim_edges_supported(np.vstack(edges),corners))
        self.assertFalse(rim_edges_supported(np.vstack(edges[:3]),corners))
        short=edges[3][15:25]
        self.assertFalse(rim_edges_supported(np.vstack(edges[:3]+[short]),corners))

    def test_two_disconnected_full_rims_are_ambiguous_and_rejected(self):
        t=np.linspace(0.,1.,100)
        points=np.concatenate([np.column_stack((.45+.4*t,np.full(100,y),np.full(100,.22))) for y in (.05,.45)] + [np.column_stack((np.full(100,x),.05+.4*t,np.full(100,.22))) for x in (.45,.85)])
        both=np.vstack([points,points+[-.25,-.6,0.]])
        with patch('scene1.visual_cycle.cloud',return_value=(both,np.tile([60,150,150],(len(both),1)))):
            self.assertIsNone(bin_rim(dict(table_z_base_m=-.08,stamp=1.)))

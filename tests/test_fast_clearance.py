import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scene1.clearance import boxes_overlap


def original(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    axes = [*rotation_a.T, *rotation_b.T]
    axes += [np.cross(a,b) for a in rotation_a.T for b in rotation_b.T]
    delta = np.asarray(center_b)-center_a
    for axis in axes:
        if np.linalg.norm(axis)<1e-8:
            continue
        if abs(axis@delta)>np.abs(axis@rotation_a)@half_a+np.abs(axis@rotation_b)@half_b:
            return False
    return True


class FastClearanceTest(unittest.TestCase):
    def test_equivalence_with_original_over_rotations_and_scales(self):
        rng=np.random.default_rng(7461)
        for _ in range(2000):
            args=(rng.uniform(-.1,.1,3),Rotation.random(random_state=rng).as_matrix(),
                  rng.uniform(.005,.08,3),rng.uniform(-.1,.1,3),
                  Rotation.random(random_state=rng).as_matrix(),rng.uniform(.005,.08,3))
            self.assertEqual(boxes_overlap(*args),original(*args))

    def test_contact_and_nearly_parallel_axes_keep_original_semantics(self):
        for angle in [0.,1e-10,1e-8,1e-6]:
            rot=Rotation.from_rotvec([0,0,angle]).as_matrix()
            for offset in [-1e-10,-1e-14,0.,1e-14,1e-10]:
                args=(np.zeros(3),np.eye(3),np.full(3,.02),
                      np.array([.04+offset,0,0]),rot,np.full(3,.02))
                self.assertEqual(boxes_overlap(*args),original(*args))

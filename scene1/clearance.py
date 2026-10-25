"""Conservative open-finger/parcel overlap checks from public robot geometry."""
import numpy as np


def boxes_overlap(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    axes = np.vstack((rotation_a.T, rotation_b.T,
        np.cross(rotation_a.T[:,None,:],rotation_b.T[None,:,:]).reshape(-1,3)))
    axes = axes[np.linalg.norm(axes,axis=1) >= 1e-8]
    delta = np.asarray(center_b)-center_a
    radii = np.abs(axes@rotation_a)@half_a + np.abs(axes@rotation_b)@half_b
    separation = np.abs(axes@delta)-radii
    if np.any(np.abs(separation) <= 1e-12):
        # Preserve the original scalar operation order near exact contact;
        # this is a numerical fallback, not a different collision margin.
        return not any(abs(axis@delta) > np.abs(axis@rotation_a)@half_a
                       + np.abs(axis@rotation_b)@half_b for axis in axes)
    return not bool(np.any(separation > 0.))


def open_fingers_clear(tool_matrix, target, plane, margin=.003):
    """Check the two open inner fingers against the observed parcel volume.

    Finger bounds are derived from the public 2F85 inner_finger mesh and its
    open URDF chain, relative to the existing 148.8 mm pinch TCP. This checks
    the near-parcel fingers, not whole-robot collision freedom.
    """
    major = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
    major /= np.linalg.norm(major)
    rotation = np.column_stack(([*major,0.],[-major[1],major[0],0.],[0.,0.,1.]))
    center = np.asarray(target.get('oriented_center_base_m', target['position_base_m']),float).copy()
    height = target['top_z_base_m']-plane
    center[2] = plane+height/2
    half = np.r_[target.get('oriented_extent_m',target['extent_xy_m']),height]/2+margin
    finger_half = np.array([.015852,.011,.028258])
    for sign in (-1,1):
        finger_center = tool_matrix[:3,3]+tool_matrix[:3,:3]@np.array([sign*.056252,0.,.025857])
        if boxes_overlap(finger_center,tool_matrix[:3,:3],finger_half,center,rotation,half):
            return False
    return True

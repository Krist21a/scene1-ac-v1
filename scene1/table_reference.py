"""Small rigid reference changes measured from four visible tabletop edges."""
from .rgbd import snapshot_hsv
import copy
import cv2
import numpy as np
from scipy.spatial import cKDTree


def table_boundary(snapshot, *, fit_tilt=False):
    """Actual tabletop-to-lower-surface RGB-D edges, before hull simplification.

    The image border and higher occluding hands provide no exterior edge
    evidence. Hull vertices alone may discard long, nearly collinear edges.
    """
    depth = snapshot['depth_m']
    intrinsic, matrix = snapshot['intrinsic'], snapshot['camera_to_base']
    v, u = np.indices(depth.shape)
    x = (u-intrinsic[0, 2])/intrinsic[0, 0]
    y = (v-intrinsic[1, 2])/intrinsic[1, 1]
    z = (matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2])*depth+matrix[2, 3]
    plane = snapshot['table_z_base_m']
    hsv = snapshot_hsv(snapshot)
    valid = (depth > .1) & np.isfinite(depth) & np.isfinite(z)
    green = ((hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85)
             & (hsv[:, :, 1] >= 100))
    surface = plane
    if fit_tilt:
        # A small base tilt makes a real far tabletop edge fall outside the
        # horizontal height band. Fit only already admitted green tabletop
        # samples, then apply the same bands relative to that observed plane.
        # Neither the grasp plane nor the registration acceptance gates change.
        bx = (matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])*depth+matrix[0, 3]
        by = (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])*depth+matrix[1, 3]
        seed = (valid & green & (abs(z-plane) < .012))[::4, ::4]
        xy = np.column_stack((bx[::4, ::4][seed], by[::4, ::4][seed]))
        height = z[::4, ::4][seed]
        if len(height) >= 100:
            origin = np.median(xy, axis=0)
            design = np.column_stack((xy-origin, np.ones(len(xy))))
            singular = np.linalg.svd(design, compute_uv=False)
            if singular[-1] >= .03*singular[0]:
                coefficients = np.linalg.lstsq(design, height, rcond=None)[0]
                inliers = abs(design@coefficients-height) <= .003
                inlier_singular = np.linalg.svd(design[inliers], compute_uv=False)
                if (np.count_nonzero(inliers) >= max(100, .7*len(height))
                        and len(inlier_singular) == 3
                        and inlier_singular[-1] >= .03*inlier_singular[0]):
                    coefficients = np.linalg.lstsq(design[inliers], height[inliers], rcond=None)[0]
                    residual = abs(design@coefficients-height)
                    if (np.linalg.norm(coefficients[:2]) <= np.tan(np.deg2rad(2.))
                            and abs(coefficients[2]-plane) <= .012
                            and np.count_nonzero(residual <= .003) >= max(100, .7*len(height))):
                        surface = (coefficients[0]*(bx-origin[0])
                                   + coefficients[1]*(by-origin[1])+coefficients[2])
                        snapshot['table_boundary_plane_fit'] = dict(
                            tilt_degrees=float(np.rad2deg(np.arctan(np.linalg.norm(coefficients[:2])))),
                            samples=len(height), inliers=int(np.count_nonzero(residual <= .003)),
                            residual_p90_m=float(np.percentile(residual[residual <= .003], 90)))
    top = valid & green & (abs(z-surface) < .012)
    lower = valid & (z < surface-.015)
    boundary = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        for shift in (-2, 2):
            boundary |= top & np.roll(lower, shift, axis=axis)
    boundary[:2] = boundary[-2:] = False
    boundary[:, :2] = boundary[:, -2:] = False
    camera = np.column_stack((x[boundary]*depth[boundary],
                              y[boundary]*depth[boundary], depth[boundary]))
    return (camera @ matrix[:3, :3].T+matrix[:3, 3])[:, :2]


def _rectangle_frame(hull, plane, boundary=None):
    if hull is None:
        return None
    points=np.asarray(hull,np.float32).reshape(-1,2)
    if len(points)<4 or not np.isfinite(points).all() or not np.isfinite(plane):
        return None
    rectangle=cv2.minAreaRect(points);corners=cv2.boxPoints(rectangle)
    edges=np.roll(corners,-1,axis=0)-corners
    lengths=np.linalg.norm(edges,axis=1)
    if min(lengths)<.3 or max(lengths)>2.:
        return None
    if cv2.contourArea(points)<.7*np.prod(rectangle[1]):
        return None
    supporting = points if boundary is None else np.asarray(boundary).reshape(-1, 2)
    if not np.isfinite(supporting).all():
        return None
    support=[]
    for i,edge in enumerate(edges):
        normal=np.array([-edge[1],edge[0]])/lengths[i]
        selected=supporting[abs((supporting-corners[i])@normal)<.008]
        if boundary is not None and len(selected) < 6:
            return None
        span=float(np.ptp(selected@(edge/lengths[i]))) if len(selected)>1 else 0.
        if span<max(.075,.08*lengths[i]):
            return None
        support.append(span)
    index=int(np.argmax(abs(edges[:,0])/lengths))
    xaxis=edges[index]/lengths[index]
    if xaxis[0]<0:xaxis=-xaxis
    return dict(center=np.r_[rectangle[0],plane],
        rotation=np.column_stack((xaxis,[-xaxis[1],xaxis[0]])),
        size=np.array([lengths[index],lengths[(index+1)%4]]),support=support)


def _edge_frame(boundary, plane):
    """Retain actual straight depth edges; image/crop borders never enter here."""
    if boundary is None or not np.isfinite(plane):
        return None
    points = np.asarray(boundary, float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 30:
        return None
    _, indices = np.unique(np.floor(points/.004), axis=0, return_index=True)
    points = points[indices]
    if len(points) < 30:
        return None
    distances, neighbors = cKDTree(points).query(points, k=25)
    local = points[neighbors]
    centered = local-local.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered)/25)
    straight = ((distances[:, -1] <= .10) & (distances[:, -1] >= .025)
                & (values[:, 0] <= .02*values[:, 1]) & (values[:, 1] >= .0001))
    points, normals = points[straight], vectors[straight, :, 0]
    if len(points) < 30:
        return None
    strength = np.linalg.eigvalsh(normals.T@normals)
    if strength[0] < .1*strength[1]:
        return None  # One straight edge cannot determine planar translation.
    return dict(points=points, normals=normals, plane=float(plane))


def table_frame(hull, plane, boundary=None):
    rectangle = _rectangle_frame(hull, plane, boundary)
    edges = _edge_frame(boundary, plane)
    if rectangle is not None:
        rectangle['edges'] = edges
        return rectangle
    return None if edges is None else dict(edges=edges, partial=True)


def _edge_motion(reference, current, initial_transform=None):
    """Register independently observed nonparallel table edges, not parcels."""
    a, b = reference.get('edges'), current.get('edges')
    if a is None or b is None or abs(b['plane']-a['plane']) > .015:
        return None
    origin = np.median(a['points'], axis=0)
    local = a['points']-origin
    tree = cKDTree(b['points'])
    state = np.zeros(3)
    if initial_transform is not None:
        initial = np.asarray(initial_transform, float)
        if initial.shape != (4, 4) or not np.isfinite(initial).all():
            return None
        # Initialize correspondence search from a previously VERIFIED table
        # transform. Never use a robot goal or return this seed as evidence.
        # All original correspondence, fit and absolute-motion gates below
        # still apply against the original anchored table, not the last frame.
        state[:2] = initial[:2, :2]@origin+initial[:2, 3]-origin
        state[2] = np.arctan2(initial[1, 0], initial[0, 0])
        if np.linalg.norm(state[:2]) > .03 or abs(state[2]) > np.deg2rad(2.):
            return None
    evidence = None
    for iteration in range(10):
        angle = state[2]
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = local@rotation.T
        moved = rotated+origin+state[:2]
        distances, indices = tree.query(moved)
        normals = b['normals'][indices]
        aligned = abs(np.sum((a['normals']@rotation.T)*normals, axis=1)) >= .95
        valid = (distances <= .02) & aligned
        if np.count_nonzero(valid) < max(30, .4*len(local)):
            return None
        n, p = normals[valid], rotated[valid]
        strength = np.linalg.eigvalsh(n.T@n)
        if strength[0] < .1*strength[1]:
            return None
        residual = np.sum(n*(moved[valid]-b['points'][indices[valid]]), axis=1)
        jacobian = np.column_stack((n, np.sum(n*np.column_stack((-p[:,1],p[:,0])), axis=1)))
        singular = np.linalg.svd(jacobian, compute_uv=False)
        if singular[-1] < .03*singular[0]:
            return None
        weights = np.minimum(1., .003/np.maximum(abs(residual), 1e-9))**.5
        delta = np.linalg.lstsq(jacobian*weights[:,None], -residual*weights, rcond=None)[0]
        evidence = dict(edge_matches=int(np.count_nonzero(valid)),
                        edge_residual_p90_m=float(np.percentile(abs(residual),90)))
        # Quantized depth changes nearest neighbors at tens of micrometers.
        # Stop by projected geometric update, well below the 3 mm fit gate.
        if np.max(abs(jacobian@delta)) < .0001:
            break
        state += delta
        if np.linalg.norm(state[:2]) > .03 or abs(state[2]) > np.deg2rad(2.):
            return None
    else:
        return None
    # Partial occlusion can leave nearest neighbors on an unrelated depth
    # boundary. Certify a large, independently constrained inlier population
    # rather than allowing those correspondences to move the target frame.
    inliers = abs(residual) <= .003
    accepted = int(np.count_nonzero(inliers))
    if accepted < max(30, .7*len(residual), .4*len(local)):
        return None
    normal_strength = np.linalg.eigvalsh(n[inliers].T@n[inliers])
    singular = np.linalg.svd(jacobian[inliers], compute_uv=False)
    if normal_strength[0] < .1*normal_strength[1] or singular[-1] < .03*singular[0]:
        return None
    refined = np.linalg.lstsq(jacobian[inliers], -residual[inliers], rcond=None)[0]
    if np.max(abs(jacobian[inliers]@refined)) > .003:
        return None
    state += refined
    if np.linalg.norm(state[:2]) > .03 or abs(state[2]) > np.deg2rad(2.):
        return None
    rotation = np.array([[np.cos(state[2]), -np.sin(state[2])],
                         [np.sin(state[2]), np.cos(state[2])]])
    moved = local@rotation.T+origin+state[:2]
    residual = np.sum(n*(moved[valid]-b['points'][indices[valid]]), axis=1)
    inliers = abs(residual) <= .003
    accepted = int(np.count_nonzero(inliers))
    if accepted < max(30, .7*len(residual), .4*len(local)):
        return None
    normal_strength = np.linalg.eigvalsh(n[inliers].T@n[inliers])
    singular = np.linalg.svd(jacobian[inliers], compute_uv=False)
    if normal_strength[0] < .1*normal_strength[1] or singular[-1] < .03*singular[0]:
        return None
    evidence.update(edge_inliers=accepted, edge_outliers=int(len(residual)-accepted),
                    edge_residual_p90_m=float(np.percentile(abs(residual[inliers]),90)),
                    edge_all_residual_p90_m=float(np.percentile(abs(residual),90)))
    if initial_transform is not None:
        evidence['edge_initialization'] = 'previous_verified_table_transform'
    transform = np.eye(4)
    transform[:2,:2] = rotation
    transform[:2,3] = origin+state[:2]-rotation@origin
    transform[2,3] = b['plane']-a['plane']
    return dict(transform=transform, center_shift_m=[*state[:2], transform[2,3]],
                angle_degrees=float(np.rad2deg(state[2])), reference_mode='visible_depth_edges', **evidence)


def table_motion(reference,current,initial_transform=None):
    if reference is None or current is None:
        return None
    if reference.get('partial') or current.get('partial'):
        return _edge_motion(reference, current, initial_transform)
    delta=current['center']-reference['center']
    rotation=current['rotation']@reference['rotation'].T
    angle=float(np.arctan2(rotation[1,0],rotation[0,0]))
    if (np.max(abs(current['size']-reference['size']))>.008
            or np.linalg.norm(delta[:2])>.03 or abs(delta[2])>.015
            or abs(angle)>np.deg2rad(2.)):
        return None
    transform=np.eye(4);transform[:2,:2]=rotation
    transform[:2,3]=current['center'][:2]-rotation@reference['center'][:2]
    transform[2,3]=delta[2]
    return dict(transform=transform,center_shift_m=delta.tolist(),angle_degrees=float(np.rad2deg(angle)))


def move_target(target,transform):
    result=copy.deepcopy(target)
    point=np.r_[target['position_base_m'],1.]
    result['position_base_m']=(transform@point)[:3].tolist()
    if 'oriented_center_base_m' in target:
        oriented_point=np.r_[target['oriented_center_base_m'],1.]
        result['oriented_center_base_m']=(transform@oriented_point)[:3].tolist()
    result['major_axis_xy']=(transform[:2,:2]@np.asarray(target['major_axis_xy'])).tolist()
    result['top_z_base_m']=target['top_z_base_m']+transform[2,3]
    if 'grasp_goal_xy' in target:
        result['grasp_goal_xy']=(transform[:2,:2]@np.asarray(target['grasp_goal_xy'])+transform[:2,3]).tolist()
    # The table already accounts for common motion. Do not subtract it twice
    # using the legacy parcel references inside the unchanged movement guard.
    result['_reference_centers_xy']=[]
    return result

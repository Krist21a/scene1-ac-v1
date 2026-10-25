"""Find a nearby bounded reference outside the unchanged finger/parcel boxes."""
import numpy as np
from scipy.optimize import minimize
from .command_sweep import sampled_sweep
from .motion import MotionFilter

FAST_CONSTRAINTS = False


class FingerSeparation:
    """Signed SAT separating gap; positive matches the original box clearance."""
    def __init__(self, target, plane):
        major = np.array(target.get('major_axis_xy', [1., 0.]), dtype=float)
        major /= np.linalg.norm(major)
        self.rotation = np.column_stack(([*major, 0.], [-major[1], major[0], 0.], [0., 0., 1.]))
        height = target['top_z_base_m']-plane
        self.center = np.array(target.get('oriented_center_base_m', target['position_base_m']), dtype=float)
        self.center[2] = plane+height/2
        self.half = np.r_[target.get('oriented_extent_m', target['extent_xy_m']), height]/2+.003
        self.finger_half = np.array([.015852, .011, .028258])

    def gaps(self, pose):
        rotation = pose[:3, :3]
        axes = np.vstack((rotation.T, self.rotation.T,
            np.cross(rotation.T[:, None, :], self.rotation.T[None, :, :]).reshape(-1, 3)))
        norm = np.linalg.norm(axes, axis=1)
        axes = axes[norm >= 1e-8]/norm[norm >= 1e-8, None]
        radius = abs(axes@rotation)@self.finger_half + abs(axes@self.rotation)@self.half
        centers = np.array([pose[:3, 3]+rotation@np.array([sign*.056252, 0., .025857])
                            for sign in (-1, 1)])
        return np.max(abs((self.center-centers)@axes.T)-radius, axis=1)


def project_retreat_reference(kinematics, measured, proposed, reference, previous,
                              lower, upper, step, target, plane, motion_config):
    """Change only a blocked proposal; all accepted outputs are rechecked.

    The signed distance is only an optimization constraint. The independent
    original boolean sweep remains the acceptance test, including filtering.
    """
    measured, proposed, reference, previous = map(lambda x: np.asarray(x, float),
                                                (measured, proposed, reference, previous))
    q = measured[8:15]
    motion = MotionFilter(np.asarray(lower), np.asarray(upper),
                          motion_config['alpha'], motion_config['max_velocity'])

    def filtered(action):
        motion.reset(previous)
        return motion.apply(action, motion_config['period'])

    def checks(action):
        sent, limits = filtered(action)
        proposed = sampled_sweep(kinematics, q, action[8:15], target, plane)
        filtered_sweep = (dict(proposed) if np.array_equal(action[8:15],sent[8:15])
                          else sampled_sweep(kinematics, q, sent[8:15], target, plane))
        return dict(proposed=proposed, filtered=filtered_sweep,
                    bounds_clipped=limits['bounds_clipped'])

    original = checks(proposed)
    if original['proposed']['clear'] and original['filtered']['clear'] and not original['bounds_clipped']:
        return proposed.copy(), dict(applied=False, original=original)
    if not sampled_sweep(kinematics, q, q, target, plane)['clear']:
        raise RuntimeError('Retreat projection cannot repair an already blocked measured pose')
    low = np.maximum(lower[8:15], reference[8:15]-step)
    high = np.minimum(upper[8:15], reference[8:15]+step)
    if np.any(low > high):
        raise RuntimeError('Retreat projection has no physically bounded step')
    separation = FingerSeparation(target, plane)
    if FAST_CONSTRAINTS and hasattr(kinematics, 'constraint_gaps'):
        geometry = np.r_[separation.rotation.ravel(), separation.center,
                         separation.half, separation.finger_half]
        gap_at = lambda joints: kinematics.constraint_gaps(joints, 1, geometry)
    else:
        gap_at = lambda joints: separation.gaps(kinematics.matrix(joints, 1))
    intervals = max(2, int(np.ceil(np.max(np.abs(np.r_[low-q, high-q]))/.012)))
    intervals += intervals % 2
    fractions = np.linspace(0., 1., intervals+1)

    def action_from(x):
        action = proposed.copy()
        action[8:15] = reference[8:15]+step*x
        return action

    def constraints(x):
        action = action_from(x)
        sent, _ = filtered(action)
        # The fixed measured pose is checked above and by final full sweeps.
        # Do not optimize its constant constraint: sub-picometre roundoff
        # near the extra 1um solver buffer can make that constraint infeasible.
        # Positive numerical separation is stricter than the original margin.
        return np.concatenate([gap_at(q+a*(endpoint-q))
            for endpoint in [action[8:15], sent[8:15]] for a in fractions[1:]])/.01-1e-4

    desired = (proposed[8:15]-reference[8:15])/step
    bounds = list(zip((low-reference[8:15])/step, (high-reference[8:15])/step))
    solution = minimize(lambda x: float(np.sum((x-desired)**2)), np.zeros(7),
        jac=lambda x: 2*(x-desired), bounds=bounds,
        constraints=[dict(type='ineq', fun=constraints)], method='SLSQP',
        options=dict(maxiter=40, ftol=1e-9))
    chosen = action_from(solution.x)
    final = checks(chosen)
    if (not np.isfinite(chosen).all() or np.any(chosen[8:15] < low-1e-10)
            or np.any(chosen[8:15] > high+1e-10)
            or not final['proposed']['clear'] or not final['filtered']['clear']
            or final['bounds_clipped']):
        raise RuntimeError('No swept-clear retreat correction within original step and physical bounds')
    return chosen, dict(applied=True, original=original, final=final,
        solver_success=bool(solution.success), solver_message=str(solution.message),
        iterations=int(solution.nit), max_step_rad=float(np.max(abs(chosen[8:15]-reference[8:15]))),
        proposal_adjustment_rad=(chosen[8:15]-proposed[8:15]).tolist(), margin_m=.003)

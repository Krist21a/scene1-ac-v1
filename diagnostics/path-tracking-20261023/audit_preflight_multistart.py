"""Offline numerical audit of rejected goals; sends no robot commands.

Every initial guess uses the exact logged box, goal and acceptance tolerances.
Failure to find a solution is not a proof of global infeasibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--urdf', type=Path, required=True)
    ap.add_argument('--fixtures', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.policy))
    from scene1.kinematics import UrdfArmKinematics
    k = UrdfArmKinematics(description=args.urdf.read_text())
    cases = []
    for path in args.fixtures:
        data = json.loads(path.read_text())
        audit = data['retreat_preflight_audit']
        measured = np.asarray(data['observation_state'])[8:15]
        pose = k.matrix(measured, 1)
        rotation = Rotation.from_matrix(pose[:3, :3])
        goal = np.asarray(audit['goal_base_m'])
        lower, upper = np.asarray(audit['lower_joint_rad']), np.asarray(audit['upper_joint_rad'])
        angle_limit = np.deg2rad(audit['orientation_tolerance_degrees'])
        tolerance = audit['position_tolerance_m']
        assert np.all(upper > lower)
        direction = pose[:2, 2] / np.linalg.norm(pose[:2, 2])
        assert np.allclose(goal, pose[:3, 3] + np.r_[.06 * direction, .07], atol=1e-10)

        def errors(joints):
            mat = k.matrix(joints, 1)
            return (mat[:3, 3] - goal,
                    (rotation.inv() * Rotation.from_matrix(mat[:3, :3])).as_rotvec())

        def objective(joints):
            position, _ = errors(joints)
            return float(position @ position)

        def slack(joints):
            _, angle = errors(joints)
            return float((angle_limit - 1e-6)**2 - angle @ angle)

        rng = np.random.default_rng(20261023)
        starts = [('retained_solution', audit['feasibility_joint_rad']),
                  ('measured', measured), ('original_reference', audit['original_reference_rad']),
                  ('box_midpoint', (lower + upper) / 2)]
        starts.extend((f'fixed_random_{i}', rng.uniform(lower, upper)) for i in range(12))
        results = []
        for name, start in starts:
            solution = minimize(objective, np.clip(start, lower, upper), method='SLSQP',
                                bounds=list(zip(lower, upper)),
                                constraints=[dict(type='ineq', fun=slack)],
                                options=dict(maxiter=180, ftol=1e-12))
            joints = np.clip(solution.x, lower, upper)
            position, angle = errors(joints)
            pe, ae = float(np.linalg.norm(position)), float(np.linalg.norm(angle))
            certified = (np.isfinite(joints).all() and np.isfinite(pe) and np.isfinite(ae)
                         and pe <= tolerance and ae <= angle_limit
                         and np.all(joints >= lower) and np.all(joints <= upper))
            results.append(dict(start=name, solver_success=bool(solution.success),
                                solver_message=str(solution.message), iterations=int(solution.nit),
                                position_residual_m=pe, orientation_residual_degrees=float(np.rad2deg(ae)),
                                joints=joints.tolist(), certified=bool(certified)))
        feasible_orientation = [r for r in results if r['orientation_residual_degrees'] <= np.rad2deg(angle_limit)]
        cases.append(dict(fixture=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          failed_step=data['step'], goal_base_m=goal.tolist(),
                          position_tolerance_m=tolerance,
                          orientation_tolerance_degrees=audit['orientation_tolerance_degrees'],
                          starts=len(starts), certified_solutions=sum(r['certified'] for r in results),
                          minimum_position_residual_within_orientation_m=min(
                              (r['position_residual_m'] for r in feasible_orientation), default=None),
                          results=results))
    report = dict(cases=cases, commands_sent=False, policy_changed=False,
                  limitations=['Offline numerical evidence, not a global infeasibility proof',
                               'The 8-degree orientation gate was introduced by this path candidate',
                               'All initial guesses retained; no task retry or tolerance relaxation'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: c[k] for k in ['fixture', 'starts', 'certified_solutions',
                                       'minimum_position_residual_within_orientation_m']} for c in cases]))


if __name__ == '__main__':
    main()

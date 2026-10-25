"""Compare retained semantic stages without treating failed-run time as a gain."""
import argparse
import datetime
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = [
    'steps', 'elapsed_s', 'elapsed_sim_s', 'tracking_samples',
    'proximal_wbc_rmse_rad', 'wrist_wbc_rmse_rad',
    'command_tracking_rmse_rad', 'wbc_tracking_rmse_rad',
    'plateau_samples', 'plateau_bias_rad', 'per_joint_wbc_rmse_rad',
    'command_fk_error_m', 'wbc_fk_error_m', 'explicit_hold_steps',
    'bounds_clipped_steps', 'velocity_limited_steps',
    'within_phase_command_variation_rad', 'within_phase_outer_residual_variation_rad',
    'actuator_limit_near_samples', 'requested_torque_above_actuator_range_samples',
]


def read(path):
    return json.loads(path.read_text())


def load_run(label):
    run = HERE / label
    derived = HERE / 'analysis-full' / label
    state = read(run / 'experiment.json')
    assert state.get('ended_at'), 'Only completed attempts may be compared'
    metrics = read(derived / 'metrics-full.json')
    audit = read(derived / 'audit.json')
    phases = metrics['phases']
    def rms(field):
        available = [x for x in phases.values()
                     if x.get(field) is not None and x.get('tracking_samples', 0)]
        n = sum(x['tracking_samples'] for x in available)
        return math.sqrt(sum(x['tracking_samples'] * x[field] ** 2
                             for x in available) / n) if n else None
    return dict(
        label=label, version=state['version'], seed=state['seed'],
        requested_parcels=state['parcels'],
        visual_completion=state.get('visual_completion_checks_passed', False),
        four_parcel_visual_complete=state.get('independent_four_parcel_visual_completion', False),
        official_task_success=audit['task_success'],
        policy_sha256=state['policy_manifest_sha256'],
        controller_sha256=state['runtime_controller_sha256'],
        vr_sha256=state.get('runtime_vr_sha256'),
        task_sha256=state.get('task_sha256'), dynamic_qr_sha256=state.get('dynamic_qr_sha256'),
        stop_event=state.get('stop_event'), errors=state.get('errors'),
        summary=audit['summary'], duration_s=metrics['duration_s'],
        elapsed_sim_s=metrics['policy_sim_elapsed_s'],
        proximal_wbc_rmse_rad=rms('proximal_wbc_rmse_rad'),
        wrist_wbc_rmse_rad=rms('wrist_wbc_rmse_rad'),
        phases={name: {k: x[k] for k in FIELDS if k in x} for name, x in phases.items()},
        per_parcel=metrics['per_parcel'],
        checks={key: audit.get(key) for key in [
            'target_clips', 'velocity_limited_steps', 'pd_limits_50hz',
            'complete_events', 'bin_confirmations', 'retreat_projections',
            'regrasp_sweep_block_steps', 'regrasp_forward_gate_hold_steps']},
        reference=read(derived / 'reference-analysis.json'),
        joint_bounds=read(derived / 'joint-bound-audit.json'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference')
    ap.add_argument('candidate')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    left, right = load_run(a.reference), load_run(a.candidate)
    same_conditions = all(left[k] == right[k] for k in ['seed', 'requested_parcels', 'policy_sha256'])
    comparisons = {}
    for name in sorted(left['phases'].keys() & right['phases'].keys()):
        comparisons[name] = {}
        for key in FIELDS:
            x, y = left['phases'][name].get(key), right['phases'][name].get(key)
            if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                comparisons[name][key] = dict(reference=x, candidate=y, difference=y-x,
                                               relative_change=(y/x-1) if x else None)
    parcel_comparisons = {}
    for parcel in sorted(left['per_parcel'].keys() & right['per_parcel'].keys()):
        lp, rp = left['per_parcel'][parcel], right['per_parcel'][parcel]
        phases = {}
        for name in sorted(lp['phases'].keys() & rp['phases'].keys()):
            values = {}
            for key in FIELDS:
                x, y = lp['phases'][name].get(key), rp['phases'][name].get(key)
                if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                    values[key] = dict(reference=x, candidate=y, difference=y-x,
                                       relative_change=(y/x-1) if x else None)
            phases[name] = values
        parcel_comparisons[parcel] = dict(
            reference_visually_complete=lp['visually_complete'],
            candidate_visually_complete=rp['visually_complete'], phases=phases)
    eligible = same_conditions and left['four_parcel_visual_complete'] and right['four_parcel_visual_complete']
    report = dict(
        generated_at=datetime.datetime.now().astimezone().isoformat(),
        runs=[left, right], same_seed_parcel_count_and_policy=same_conditions,
        phase_comparisons=comparisons,
        per_parcel_phase_comparisons=parcel_comparisons,
        full_task_timing_eligible=eligible,
        full_task_wall_speedup_fraction=(1-right['duration_s']/left['duration_s']) if eligible else None,
        full_task_sim_speedup_fraction=(1-right['elapsed_sim_s']/left['elapsed_sim_s']) if eligible else None,
        limitations=[
            'Closed-loop upper references can differ despite identical code and initial seed.',
            'All-parcel phase aggregates can have unequal parcel coverage; use the per-parcel comparison for matched stages.',
            'A shared phase name does not certify stage completion in a failed trial; inspect its terminal phase and raw rows.',
            'Phase mean bias is not a static bias; static comparisons use qualifying reference plateaus.',
            'A single matched pair is not a success-rate or statistical-significance estimate.',
            'FK differences are derived from the robot model, not independent optical end-effector truth.',
            'Failed and partial trials remain in the evidence; their short time is not a task speedup.',
            'Null task_success is not official success.',
        ],
    )
    destination = Path(a.output)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(destination), same_conditions=same_conditions,
                          full_task_timing_eligible=eligible,
                          runs=[{k: r[k] for k in ['label', 'visual_completion', 'duration_s',
                                'proximal_wbc_rmse_rad', 'wrist_wbc_rmse_rad']} for r in [left, right]])))


if __name__ == '__main__':
    main()

"""Summarize retained evidence; no controller, ROS, or scene access."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    rows = []
    for label, name in [('baseline', 'baseline-empty-v2'), ('reference_index', 'candidate-empty'), ('wrist_gains', 'gains-empty'), ('wrist_torque_pd', 'pd-empty-v2')]:
        metrics = read(name + '/metrics.json')
        for segment in metrics['segments']:
            for field in ['ramp_wbc_rmse_rad', 'static_abs_bias_mean_rad', 'static_position_std_rad', 'overshoot_rad', 'settle_within_1deg_s']:
                rows.append(dict(scope='empty', variant=label, phase=segment['name'], metric=field, value=segment[field]))
    task = {}
    for label, name in [('baseline', 'baseline-first-parcel'), ('wrist_gains', 'gains-first-parcel')]:
        metrics = read(name + '/metrics.json')
        summary = read(name + '/output/summary.json')
        steps = [json.loads(line) for line in (HERE / name / 'output/steps.jsonl').open()]
        runs, current = [], []
        for s in steps:
            if s['motion']['bounds_clipped']:
                current.append(s['step'])
            elif current:
                runs.append(current); current = []
        if current:
            runs.append(current)
        first = next(s for s in steps if s['motion']['bounds_clipped'])
        task[label] = dict(run=name, steps=len(steps), reason=summary['reason'],
                           visual_confirmed_parcels=summary['visual_confirmed_parcels'],
                           official_task_success=summary['task_success'],
                           independent_four_parcel_success=False,
                           active_duration_s=metrics['duration_s'],
                           bounds_clipped_steps=sum(map(len, runs)),
                           clip_runs=[dict(first=r[0], last=r[-1], length=len(r)) for r in runs],
                           first_clipped_step=first['step'],
                           sampled_actuator_limit_near=sum(p['actuator_limit_near_samples'] for p in metrics['phases'].values()),
                           sampled_requested_torque_above_range=sum(p['requested_torque_above_actuator_range_samples'] for p in metrics['phases'].values()),
                           frozen_policy_steps=sum(p['frozen_policy_steps'] for p in metrics['phases'].values()),
                           explicit_hold_steps=sum(p['explicit_hold_steps'] for p in metrics['phases'].values()),
                           velocity_limited_steps=sum(p['velocity_limited_steps'] for p in metrics['phases'].values()))
        vm = read(name + '/visual-metrics.json')['phases']
        for phase, vals in metrics['phases'].items():
            for field in ['steps', 'elapsed_s', 'command_tracking_rmse_rad', 'wbc_tracking_rmse_rad', 'wrist_wbc_rmse_rad',
                          'bounds_clipped_steps', 'frozen_policy_steps', 'explicit_hold_steps', 'within_phase_outer_residual_variation_rad']:
                rows.append(dict(scope='task_phase', variant=label, phase=phase, metric=field, value=vals[field]))
            rows.append(dict(scope='task_phase', variant=label, phase=phase, metric='mean_fk_wbc_error_m',
                             value=vals['wbc_fk_error_m']['mean'] if vals['wbc_fk_error_m'] else None))
            rows.append(dict(scope='task_phase', variant=label, phase=phase, metric='visual_offset_variation_rad', value=vm[phase]['offset_variation_rad']))
    configs = [read(n + '/output/run.json') for n in ['baseline-first-parcel', 'gains-first-parcel']]
    shared_keys = ['mode', 'task', 'config', 'stop_after_parcels', 'original_model', 'base_config']
    report = dict(seed=0, seed_role='development', repeats_per_variant=1,
                  same_outer_run_settings={key: configs[0][key] == configs[1][key] for key in shared_keys},
                  prefix=read('matched-prefix.json'), prefix_clocks=read('prefix-clock-comparison.json'),
                  task=task, measurements=rows,
                  caveats=['asymmetric stopping: baseline clipping was discovered retrospectively',
                           'different stage coverage; full task durations and success rates are not comparable',
                           'FK is derived robot kinematics, not independent TCP truth',
                           'visual offset variation is not vision computation cost',
                           'one development seed; no new seed or full four-parcel test'])
    (HERE / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (HERE / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scope', 'variant', 'phase', 'metric', 'value'])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'measurements'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

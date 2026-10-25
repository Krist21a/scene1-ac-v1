"""Keep every outcome and export comparable phase metrics without success selection."""
import csv
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    empty, tasks, phases = [], [], []
    for p in sorted(HERE.glob('*/experiment.json')):
        d = p.parent
        state = read(p)
        if state.get('kind') == 'empty':
            metrics = read(d/'metrics.json')
            empty.append(dict(label=d.name, status=state['status'],
                              controller_sha256=state.get('runtime_controller_sha256'),
                              stop_event=state.get('stop_event'), error=state.get('error'),
                              segments=metrics.get('segments') if metrics else None,
                              reference=read(d/'reference-analysis.json'),
                              velocity_reference=read(d/'velocity-audit.json')))
            continue
        if 'parcels' not in state:
            continue
        derived = HERE/'analysis-full'/d.name
        metrics, audit = read(derived/'metrics-full.json'), read(derived/'audit.json')
        summary = state.get('policy_summary') or read(d/'output/summary.json') or {}
        record = dict(label=d.name, version=state.get('version'), seed=state.get('seed'),
                      requested_parcels=state['parcels'], status=state['status'],
                      controller_sha256=state.get('runtime_controller_sha256'),
                      dynamic_qr_sha256=state.get('dynamic_qr_sha256'),
                      policy_sha256=state.get('policy_manifest_sha256'),
                      official_task_success=summary.get('task_success'),
                      visual_confirmed_parcels=summary.get('visual_confirmed_parcels'),
                      four_parcel_visual_complete=state.get('independent_four_parcel_visual_completion',False),
                      reason=summary.get('reason'), error=summary.get('error') or state.get('error'),
                      stop_event=state.get('stop_event'),
                      policy_duration_wall_s=metrics.get('duration_s') if metrics else None,
                      policy_duration_sim_s=metrics.get('policy_sim_elapsed_s') if metrics else None,
                      steps=summary.get('steps'),
                      per_parcel=metrics.get('per_parcel') if metrics else None)
        record['chain_targets']=read(derived/'chain-targets.json') or read(HERE/(d.name+'-chain-targets.json'))
        record['joint_bound_audit']=read(derived/'joint-bound-audit.json')
        record['reset_to_cleanup_wall_s']=(
            (datetime.datetime.fromisoformat(state['ended_at'])-
             datetime.datetime.fromisoformat(state['started_at'])).total_seconds()
            if state.get('ended_at') else None)
        if audit:
            for key in ['gripper_event_count','target_clips','velocity_limited_steps',
                        'regrasp_sweep_block_steps','regrasp_forward_gate_hold_steps','pd_limits_50hz']:
                record[key]=audit.get(key)
        tasks.append(record)
        if metrics:
            for phase,x in metrics['phases'].items():
                item=dict(label=d.name, version=state.get('version'), seed=state.get('seed'), phase=phase)
                for key in ['steps','elapsed_s','elapsed_sim_s','proximal_wbc_rmse_rad','wrist_wbc_rmse_rad',
                            'command_tracking_rmse_rad','wbc_tracking_rmse_rad','plateau_samples',
                            'bounds_clipped_steps','velocity_limited_steps','frozen_policy_steps',
                            'explicit_hold_steps','within_phase_command_variation_rad',
                            'within_phase_outer_residual_variation_rad','actuator_limit_near_samples',
                            'requested_torque_above_actuator_range_samples']:
                    item[key]=x.get(key)
                for key in ['command_fk_error_m','wbc_fk_error_m']:
                    for statistic in ['mean','p95','max']:
                        item[key+'_'+statistic]=(x.get(key) or {}).get(statistic)
                phases.append(item)
    report=dict(generated_at=datetime.datetime.now().astimezone().isoformat(),empty=empty,tasks=tasks,
                limitations=['Every attempt retained, including failures and instrumentation errors',
                             'Only matched four-parcel visual completions may enter task timing comparison',
                             'Null official_task_success is not official success',
                             'Reference and sensor alignment is approximate; FK is not independent optical TCP measurement',
                             'Outer residual variation includes intended trajectory movement, not pure corrective distance',
                             'QP internal active constraints and exact actuator application timestamps unavailable'])
    (HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if phases:
        with (HERE/'phase-metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(phases[0]));w.writeheader();w.writerows(phases)
    empty_rows=[]
    for experiment in empty:
        for s in experiment['segments'] or []:
            empty_rows.append(dict(label=experiment['label'],**{k:v for k,v in s.items() if not isinstance(v,list)}))
    if empty_rows:
        with (HERE/'empty-metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(empty_rows[0]));w.writeheader();w.writerows(empty_rows)
    print(json.dumps(dict(empty=len(empty),tasks=len(tasks),phase_rows=len(phases))))


if __name__ == '__main__':
    main()

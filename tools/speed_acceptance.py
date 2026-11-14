"""Require at least ten frozen four-parcel runs below an explicit wall-time limit."""
import argparse
import hashlib
import json
from pathlib import Path
from report_performance import summarize


def inspect(label, root, store, max_wall_s=600.):
    if max_wall_s not in (600., 650.):
        raise ValueError('Supported acceptance limits are 600 or 650 wall seconds')
    d = root/'diagnostics/path-tracking-continuation-20261023'
    run = d/label
    derived = d/'analysis-full'/label
    state = json.loads((run/'experiment.json').read_text())
    pipeline = json.loads((store/(label+'-pipeline.json')).read_text())
    audit = json.loads((derived/'audit.json').read_text())
    metrics = json.loads((derived/'metrics-full.json').read_text())
    timing = summarize(json.loads(line) for line in (run/'output/performance.jsonl').open())
    pd = audit['pd_limits_50hz']['formal_policy']
    stages = pipeline['stages']
    cpu_layout = pipeline.get('cpu_layout', 'original')
    cpu_ok = True
    if cpu_layout != 'original':
        from cpu_allocation import ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, layout_for, VISION_PROFILE, PROFILES, MPC_PROFILE, CRITICAL_PROFILE, RENDER_PROFILE, PERFORMANCE_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE
        allocation = json.loads((store/(label+'-cpu-allocation.json')).read_text())
        cpu_ok = (cpu_layout in PROFILES and state.get('cpu_layout') == cpu_layout
                  and allocation['profile'] == cpu_layout
                  and allocation.get('applied') == layout_for(cpu_layout) and allocation.get('restored') is True
                  and not allocation.get('restore_errors'))
        if cpu_layout == MPC_PROFILE:
            cpu_ok = cpu_ok and state.get('mpc_main_affinity', {}).get('applied') == list(range(4, 12))
        if cpu_layout in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE):
            bdi = allocation.get('writeback_bdi', {})
            cpu_ok = (cpu_ok and allocation.get('writeback_restored') is True
                      and bdi.get('restored') is True and bdi.get('applied') == '80' and bdi.get('at_end') == '80'
                      and bool(allocation.get('writeback_original'))
                      and allocation.get('writeback_applied') == {'dirty_ratio':'40','dirty_bytes':'0'}
                      and allocation.get('writeback_at_end') == allocation.get('writeback_applied'))
        if cpu_layout == ALGO_POWER_PROFILE:
            original = allocation.get('power_original', {})
            initial = {k:v['governor'] for k,v in original.items()}
            expected = {k:('performance' if int(k.removeprefix('policy')) < 4 else v) for k,v in initial.items()}
            cpu_ok = (cpu_ok and len(original) == 20 and allocation.get('power_restored') is True
                and allocation.get('power_initial') == initial and set(initial.values()) == {'powersave'}
                and allocation.get('power_applied') == expected and allocation.get('power_at_end') == expected)
        if cpu_layout == PHASE_POWER_PROFILE:
            transition = state.get('phase_power_transition', {})
            original = allocation.get('power_original', {})
            initial = {k:v['governor'] for k,v in original.items()}
            cpu_ok = (cpu_ok and bool(original) and allocation.get('power_restored') is True
                and allocation.get('power_initial') == initial and transition.get('before') == initial
                and set(initial.values()) == {'powersave'}
                and transition.get('trigger') == 'VISUAL_CYCLE_AFTER_REGRASP step='+str(transition.get('step'))+' parcel=1'
                and transition.get('applied', {}).keys() == original.keys()
                and set(transition.get('applied', {}).values()) == {'performance'}
                and allocation.get('power_at_end') == transition.get('applied')
                and transition.get('applied_monotonic_s',0) >= transition.get('observed_monotonic_s',1))
        if cpu_layout in (PERFORMANCE_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE):
            cpu_ok = cpu_ok and bool(allocation.get('power_original')) and allocation.get('power_restored') is True
            for key in ('power_applied', 'power_at_end'):
                settings = allocation.get(key, {})
                cpu_ok = cpu_ok and settings.keys() == allocation.get('power_original', {}).keys() and set(settings.values()) == {'performance'}
    if cpu_layout != 'original' and cpu_layout in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, CRITICAL_PROFILE, RENDER_PROFILE, PERFORMANCE_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE):
        affinity = state.get('critical_affinity', {})
        if cpu_layout == VISION_PROFILE:
            workers = [r for r in affinity.get('changes', []) if r['role'] == 'worker']
            cpu_ok = cpu_ok and bool(workers) and all(r['applied'] == [10,11,16,17,18,19] for r in workers)
        required = {'mpc': [4, 5], 'wbc': [6, 7], 'physics': [8, 9]}
        if cpu_layout in (RENDER_PROFILE, PERFORMANCE_PROFILE, PERFORMANCE_WRITEBACK_PROFILE):
            required['render'] = [10, 11]
        cpu_ok = cpu_ok and affinity.get('requested') == required
        for role, target in required.items():
            records = [r for r in affinity.get('changes', []) if r['role'] == role]
            cpu_ok = cpu_ok and len(records) == 1 and records[0]['applied'] == target and records[0]['tid'] == affinity.get('roles', {}).get(role)
    entry_bins = [b for b in audit['bin_confirmations'] if b.get('confirmation_mode') == 'independent_entry']
    entry_ok = not entry_bins or (len(entry_bins) == 4 and
        all(b.get('independent_entry',{}).get('passed') is True for b in entry_bins) and
        [b['independent_entry']['parcel_index'] for b in entry_bins] == [0,1,2,3])
    flat_bins = [b for b in audit['bin_confirmations'] if b.get('layout_mode') == 'flat_two_stacks'
                 and b.get('confirmation_mode','stack_height') != 'independent_entry']
    grid_bins = [b for b in audit['bin_confirmations'] if b.get('layout_mode') == 'flat_grid']
    grid_ok = not grid_bins or (len(grid_bins) == 4 and [b['slot'] for b in grid_bins] == [0,1,2,3]
                                and all(b.get('new_points',0) >= 80 for b in grid_bins))
    flat_ok = True
    if flat_bins:
        flat_ok = len(flat_bins) == 4 and [b['slot'] for b in flat_bins] == [0, 1, 2, 3]
        for b in flat_bins:
            if b['slot'] >= 2:
                height = b.get('stack_height_verification') or {}
                flat_ok = (flat_ok and height.get('passed') is True and height.get('required_increase_m') == .02
                    and height.get('new_top_m', -1e9)-height.get('previous_top_m', 1e9) >= .02)
    checks = dict(
        full_video=(not pipeline.get('record_video',False) or
            (state.get('video',{}).get('returncode') in (0,255)
             and state['video'].get('duration_s',0) >= (timing['formal_wall_s'] or float('inf'))
             and state['video'].get('bytes',0)>0
             and state['video'].get('stopped_monotonic_s',0) >= state.get('policy_exited_monotonic_s',float('inf')))),
        flat_grid_regions=grid_ok,
        flat_stack_height=flat_ok,
        independent_entries=entry_ok,
        cpu_allocation=cpu_ok,
        wbc_warmstart=(state['version'] not in ('wbc_warmstart','wbc_reduced') or
            (audit.get('wbc_warmstart', {}).get('hot_samples', 0)>0
             and audit['wbc_warmstart']['invalid_outputs']==0
             and audit['wbc_warmstart']['equality_peak']<=1e-5
             and audit['wbc_warmstart']['inequality_peak']<=1e-5)),
        wbc_projection=(state['version']!='wbc_reduced' or audit.get('wbc_warmstart',{}).get('projection_samples',0)>0),
        complete=state.get('independent_four_parcel_visual_completion') is True,
        guarded='stop_event' not in state and not state.get('errors'),
        pipeline=all(stages.get(k) == 0 for k in ('driver', 'analysis', 'audit', 'archive', 'restore')),
        timing_coverage=timing['steps'] == state['policy_summary']['steps'],
        under_wall_limit=timing['formal_wall_s'] is not None and timing['formal_wall_s'] < max_wall_s,
        old_timing_under_wall_limit=metrics['duration_s'] < max_wall_s,
        contiguous=audit['step_sequence_contiguous'],
        four_bins=len(audit['bin_confirmations']) == 4,
        no_target_clips=audit['target_clips'] == 0,
        pd_available=pd['available'],
        no_pd_clips=pd['raw_clipped_samples'] == 0 and pd['total_clipped_samples'] == 0,
    )
    return dict(label=label, seed=state['seed'], max_wall_s=max_wall_s, checks=checks, passed=all(checks.values()),
                formal_wall_s=timing['formal_wall_s'], previous_timing_s=metrics['duration_s'],
                sim_s=metrics['policy_sim_elapsed_s'], steps=timing['steps'],
                manifest=state['policy_manifest_sha256'], profile=state.get('speed_profile'), cpu_layout=cpu_layout,
                controller=state['version'], record_video=pipeline.get('record_video',False),
                runtime={**{k: state.get(k) for k in ('library_sha256', 'dynamic_qr_sha256', 'task_sha256', 'vr_sha256', 'orchestrator_sha256', 'wbc_library_sha256')},
                         'orchestration_sha256': hashlib.sha256(json.dumps(state['orchestration_files'], sort_keys=True).encode()).hexdigest()},
                task_success=state.get('task_success'))


def agreement(runs):
    return dict(
        same_wall_limit=len({r.get('max_wall_s',600.) for r in runs}) == 1,
        same_recording=len({r.get('record_video',False) for r in runs}) == 1,
        same_manifest=len({r['manifest'] for r in runs}) == 1,
        same_profile=len({r['profile'] for r in runs}) == 1,
        same_cpu_layout=len({r['cpu_layout'] for r in runs}) == 1,
        same_controller=len({r['controller'] for r in runs}) == 1,
        runtime_available=all(all(r['runtime'].values()) for r in runs),
        same_runtime=len({json.dumps(r['runtime'], sort_keys=True) for r in runs}) == 1,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--seed-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-wall-s', type=float, choices=(600.,650.), default=600.)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    from evidence_storage import STORE
    store = STORE
    batch = json.loads(a.seed_manifest.read_text())
    requested = batch['requested_seeds']
    runs = [inspect(label, root, store, a.max_wall_s) for label in a.labels]
    checks = dict(
        at_least_ten=len(requested) >= 10,
        randomized=batch.get('sampling', {}).get('method') == 'uniform rejection without replacement',
        batch_passed=batch.get('status') == 'passed',
        declared_wall_limit=batch.get('frozen_source',{}).get('max_wall_s',600.) == a.max_wall_s,
        exact_labels={r['label'] for r in batch.get('runs', [])} == set(a.labels),
        distinct_seeds=len(set(requested)) == len(requested),
        exact_seed_set=len(runs) == len(requested) and {r['seed'] for r in runs} == set(requested),
        all_runs_passed=all(r['passed'] for r in runs),
        **agreement(runs))
    times = sorted(r['formal_wall_s'] for r in runs if r['formal_wall_s'] is not None)
    import math
    report = dict(passed=all(checks.values()), checks=checks, runs=runs,
                  mean_s=sum(times)/len(times) if times else None,
                  p95_s=times[math.ceil(.95*len(times))-1] if times else None,
                  max_s=max(times) if times else None,
                  note='Visual acceptance only. Null task_success is not an official success verdict.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)

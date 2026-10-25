"""Require six frozen, complete four-parcel runs below 900 wall-clock seconds."""
import argparse
import json
from pathlib import Path
from report_performance import summarize


def inspect(label, root, store):
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
    checks = dict(
        complete=state.get('independent_four_parcel_visual_completion') is True,
        guarded='stop_event' not in state and not state.get('errors'),
        pipeline=all(stages.get(k) == 0 for k in ('driver', 'analysis', 'audit', 'archive', 'restore')),
        timing_coverage=timing['steps'] == state['policy_summary']['steps'],
        under_900=timing['formal_wall_s'] is not None and timing['formal_wall_s'] < 900,
        old_timing_under_900=metrics['duration_s'] < 900,
        contiguous=audit['step_sequence_contiguous'],
        four_bins=len(audit['bin_confirmations']) == 4,
        no_target_clips=audit['target_clips'] == 0,
        pd_available=pd['available'],
        no_pd_clips=pd['raw_clipped_samples'] == 0 and pd['total_clipped_samples'] == 0,
    )
    return dict(label=label, seed=state['seed'], checks=checks, passed=all(checks.values()),
                formal_wall_s=timing['formal_wall_s'], previous_timing_s=metrics['duration_s'],
                sim_s=metrics['policy_sim_elapsed_s'], steps=timing['steps'],
                manifest=state['policy_manifest_sha256'], profile=state.get('speed_profile'),
                tests_passed=stages.get('tests') == 0, task_success=state.get('task_success'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs=6)
    ap.add_argument('--holdout-seeds', nargs=4, type=int, default=[6, 7, 8, 9])
    ap.add_argument('--tested-source', help='Fully tested regression run with the same frozen policy')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = Path('/media/krist/KINGSTON/scene1-speed-20261025')
    runs = [inspect(label, root, store) for label in a.labels]
    tested_source = inspect(a.tested_source, root, store) if a.tested_source else None
    source_tests = (tested_source is not None and tested_source['tests_passed']
                    and tested_source['passed']
                    and all(r['manifest'] == tested_source['manifest']
                            and r['profile'] == tested_source['profile'] for r in runs))
    expected = {0, 2, *a.holdout_seeds}
    checks = dict(six_distinct_seeds=len(expected) == 6 and {r['seed'] for r in runs} == expected,
                  same_manifest=len({r['manifest'] for r in runs}) == 1,
                  same_profile=len({r['profile'] for r in runs}) == 1,
                  regression_passed=any(r['tests_passed'] for r in runs) or source_tests,
                  all_runs_passed=all(r['passed'] for r in runs))
    report = dict(passed=all(checks.values()), checks=checks, runs=runs, tested_source=tested_source,
                  note='Visual acceptance only. Null task_success is not an official success verdict.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)

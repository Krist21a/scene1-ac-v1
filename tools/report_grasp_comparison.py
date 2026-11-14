"""Report the preregistered ten-seed final traditional/local-ACT comparison.

Incomplete and failed rounds remain in the denominator. Official scoring is
reported separately from the existing visual and runtime acceptance checks.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from evidence_storage import ARCHIVE_ROOT, EVIDENCE_ROOT, STORE
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_round(label, learned, frozen_at, model_digest):
    pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
    root = ARCHIVE_ROOT/label
    row = dict(label=label, seed=pipeline['seed'], passed=False, complete=False,
               formal_wall_s=None, first_grasp_successes=0, grasp_recoveries=0,
               support_recoveries=0, protection_triggers={}, official_task_success=None)
    try:
        state = json.loads((root/'experiment.json').read_text())
        row['after_freeze'] = datetime.fromisoformat(state['started_at']) > frozen_at
        result = inspect(label, ROOT, STORE, 650.)
        row.update(acceptance=result, passed=result['passed'],
                   complete=result['checks']['complete'], formal_wall_s=result['formal_wall_s'],
                   official_task_success=result['task_success'])
        audit = json.loads((ROOT/'diagnostics/path-tracking-continuation-20261023/analysis-full'/label/'audit.json').read_text())
        summary = audit['summary']
        row['protection_triggers'] = dict(
            target_clips=audit['target_clips'],
            regrasp_sweep_block_steps=audit['regrasp_sweep_block_steps'],
            pd_clipped_samples=audit['pd_limits_50hz']['formal_policy']['total_clipped_samples'],
            episode_errors=int(bool(summary.get('error'))),
            handover_contract_stops=int(bool(summary.get('handover_contract_stop'))))
        row['error'] = summary.get('error')
        lifts = []
        with (root/'output/steps.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                info = record.get('cartesian_alignment', {})
                if (info.get('stage') == 'lift' and info.get('transition_to') == 'observe_pad'
                        and info.get('lift_evidence') is not None):
                    lifts.append(record['step'])
                if info.get('support_recovery') is not None:
                    row['support_recoveries'] += 1
        row['externally_verified_pickups'] = len(lifts)
        if learned:
            local = json.loads((root/'output/local-act-grasps.json').read_text())
            attempts = local['attempts']
            row['model_matches_freeze'] = local['model_manifest_sha256'] == model_digest
            indices = [a['parcel_index'] for a in attempts]
            row['attempt_accounting_valid'] = (len(indices) == len(set(indices))
                and all(type(i) is int and 0 <= i < 4 for i in indices)
                and all(a.get('end_step') in lifts for a in attempts
                        if a.get('success_before_recovery') is True or a.get('success_after_recovery') is True))
            row['first_grasp_successes'] = sum(a.get('success_before_recovery') is True for a in attempts)
            row['grasp_recoveries'] = sum(a.get('recovery_started_step') is not None for a in attempts)
            row['recovered_grasp_successes'] = sum(a.get('success_after_recovery') is True for a in attempts)
            row['attempts'] = attempts
            guard_reasons = ('outside physical', 'offset budget', 'open-finger sweep')
            row['protection_triggers']['learned_guard_rejections'] = sum(
                any(reason in (a.get('failure_reason') or '') for reason in guard_reasons)
                for a in attempts)
        else:
            # The current traditional pickup has no retry loop after a failed
            # grasp: external trial-lift completion establishes first success.
            row['first_grasp_successes'] = len(lifts)
            row['attempt_accounting_valid'] = len(lifts) <= 4
    except (OSError, KeyError, ValueError) as exc:
        row.update(passed=False, evidence_error=str(exc))
    return row


def aggregate(rows):
    times = sorted(r['formal_wall_s'] for r in rows if r['formal_wall_s'] is not None)
    percentile = None
    if times:
        position = .95*(len(times)-1)
        lo, hi = math.floor(position), math.ceil(position)
        percentile = times[lo]+(times[hi]-times[lo])*(position-lo)
    guards = Counter()
    for row in rows:
        guards.update(row['protection_triggers'])
    return dict(rounds=len(rows), completed_rounds=sum(r['complete'] for r in rows),
        successful_rounds_under_650=sum(r['passed'] for r in rows),
        task_success_rate=sum(r['complete'] for r in rows)/len(rows),
        first_grasp_successes=sum(r['first_grasp_successes'] for r in rows),
        grasp_opportunities=4*len(rows),
        first_grasp_success_rate=sum(r['first_grasp_successes'] for r in rows)/(4*len(rows)),
        grasp_recoveries=sum(r['grasp_recoveries'] for r in rows),
        support_recoveries=sum(r['support_recoveries'] for r in rows),
        timing_samples=len(times), mean_wall_s=sum(times)/len(times) if times else None,
        p95_wall_s=percentile, max_wall_s=max(times) if times else None,
        p95_method='linear interpolation at rank .95*(n-1)',
        timing_scope='All recorded attempted rounds, including truncated failures; missing timings are not imputed',
        protection_triggers=dict(guards), official_task_success=[r['official_task_success'] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traditional', nargs=10, required=True)
    parser.add_argument('--local-act', nargs=10, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--model-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Comparison report already exists')
    registry = json.loads(args.registry.read_text())
    freeze = json.loads(args.model_freeze.read_text())
    split_hash = hashlib.sha256(json.dumps(registry['seed_splits'], sort_keys=True).encode()).hexdigest()
    seeds = registry['seed_splits']['test']
    if (registry['status'] != 'complete' or freeze.get('stage') != 'frozen_final'
            or freeze.get('training_updates') != 20000 or len(set(seeds)) != 10
            or freeze.get('seed_split_sha256') != split_hash or registry['seed_split_sha256'] != split_hash
            or freeze.get('test_seeds') != seeds):
        raise ValueError('Require the original ten final-test seeds and a final frozen model')
    relative = Path(freeze['model_relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Final model must be archived under the evidence root')
    for name, expected in freeze['files_sha256'].items():
        name = Path(name)
        if name.is_absolute() or '..' in name.parts or digest(EVIDENCE_ROOT/relative/name) != expected:
            raise ValueError('Final checkpoint or preprocessing changed')
    frozen_at = datetime.fromisoformat(freeze['frozen_at'])
    if frozen_at.tzinfo is None:
        raise ValueError('Freeze timestamp must include its timezone')
    model_digest = digest(args.model_freeze)
    groups = [args.traditional, args.local_act]
    if len(set(groups[0]+groups[1])) != 20:
        raise ValueError('Need twenty distinct full-round attempts')
    rows = [[read_round(label, learned, frozen_at, model_digest) for label in labels]
            for learned, labels in enumerate(groups)]
    if any([r['seed'] for r in group] != seeds for group in rows):
        raise ValueError('Both schemes must follow the same preregistered test-seed order')
    baseline, learned = map(aggregate, rows)
    all_rows = rows[0]+rows[1]
    checks = dict(
        frozen_evidence_complete=all('evidence_error' not in r and r.get('after_freeze') is True
            and r.get('attempt_accounting_valid') is True for r in all_rows),
        baseline_passed=baseline['successful_rounds_under_650'] == 10,
        learned_passed=learned['successful_rounds_under_650'] == 10,
        first_grasps_not_worse=learned['first_grasp_successes'] >= baseline['first_grasp_successes'],
        mean_at_least_five_percent_faster=(baseline['timing_samples'] == learned['timing_samples'] == 10
            and learned['mean_wall_s'] <= .95*baseline['mean_wall_s']),
        model_matches_freeze=all(r.get('model_matches_freeze') is True for r in rows[1]),
        no_additional_protection_triggers=all(value <= baseline['protection_triggers'].get(key, 0)
            for key, value in learned['protection_triggers'].items()))
    for field in ('manifest', 'runtime', 'cpu_layout', 'controller', 'record_video'):
        values = [r.get('acceptance', {}).get(field) for r in all_rows]
        checks['same_'+field] = all(v is not None for v in values) and len({json.dumps(v, sort_keys=True) for v in values}) == 1
    checks['policy_matches_freeze'] = all(r.get('acceptance', {}).get('manifest') == freeze.get('policy_manifest_sha256')
                                        for r in all_rows) and bool(freeze.get('policy_manifest_sha256'))
    for group, key in zip(rows, ('traditional_profile', 'local_act_profile')):
        checks[key+'_matches_freeze'] = bool(freeze.get(key)) and all(
            r.get('acceptance', {}).get('profile') == freeze[key] for r in group)
    import yaml
    profiles = []
    for label, key in zip((args.traditional[0], args.local_act[0]),
                          ('traditional_profile', 'local_act_profile')):
        profile = Path(freeze[key])
        if profile.is_absolute() or '..' in profile.parts:
            raise ValueError('Unsafe frozen profile path')
        profiles.append(yaml.safe_load((STORE/'policies'/label/profile).read_text()))
    checks['declared_modes'] = (profiles[0]['visual_cycle'].get('grasp_execution') == 'traditional'
                                and profiles[1]['visual_cycle'].get('grasp_execution') == 'local_act')
    for profile in profiles:
        for key in ('grasp_execution', 'local_act_model', 'local_act_manifest'):
            profile['visual_cycle'].pop(key, None)
    checks['same_remaining_configuration'] = profiles[0] == profiles[1]
    report = dict(traditional=baseline, local_act=learned, traditional_runs=rows[0], local_act_runs=rows[1],
        replacement_checks=checks, replace_traditional=all(checks.values()),
        model_freeze_sha256=model_digest, collection_registry_sha256=digest(args.registry),
        report_generator_sha256=digest(Path(__file__)),
        official_scoring_note='null remains unscored; visual completion is not an official verdict')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(replace_traditional=report['replace_traditional'], checks=checks)))


if __name__ == '__main__':
    main()

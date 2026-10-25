"""Audit published recovery continuity and unchanged insertion readiness gates."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
label = sys.argv[1]
rows = [json.loads(line) for line in
        (root / label / 'output/steps.jsonl').read_text().splitlines()]
recoveries = []
pitch_frames = 0
for i, row in enumerate(rows):
    info = row['cartesian_alignment']
    recovery = info.get('support_recovery')
    if recovery:
        assert i > 0
        previous = rows[i - 1]
        jump = max(abs(a - b) for a, b in
                   zip(row['sent_action'][8:15], previous['sent_action'][8:15]))
        assert jump < 1e-12, ('recovery reference discontinuity', row['step'], jump)
        assert row['sent_action'][15] >= .9
        assert recovery['original_budget_steps'] == 450
        assert abs(recovery['original_margin_m'] - .025) < 1e-12
        assert recovery['phase_steps'] < 450
        assert recovery['phase_steps'] > previous['cartesian_alignment']['phase_steps']
        recoveries.append(dict(step=row['step'], arm_jump_rad=jump, **recovery))
    pitch = info.get('insert_pitch_task')
    if pitch and pitch['relaxed']:
        assert pitch['original_orientation_gate_passed']
        assert abs(pitch['weight'] - .3) < 1e-12
        assert info['regrasp_stage'] == 'insert'
        pitch_frames += 1

out = dict(label=label, recoveries=recoveries, relaxed_pitch_frames=pitch_frames,
           recovery_exercised=bool(recoveries), all_checks_passed=True)
target = root / 'analysis-full' / label / 'support-recovery-audit.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))

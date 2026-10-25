"""Measure actual release target corrections and recorded waits, per block."""
import json
import math
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
label=sys.argv[1]
rows=[json.loads(x) for x in (P/label/'output/steps.jsonl').read_text().splitlines()]
blocks=[]
waits={}
for row in rows:
    info=row['cartesian_alignment']
    for name in ['table_reference_hold','regrasp_forward_gate_hold','waiting_for_fresh_rgbd']:
        if info.get(name):waits[name]=waits.get(name,0)+1
    prep=info.get('release_preparation')
    if prep is None:continue
    if not blocks or row['step']!=blocks[-1][-1]['step']+1:blocks.append([])
    blocks[-1].append(row)
results=[]
for block in blocks:
    adjustments=[r['cartesian_alignment']['release_preparation']['settled_target_adjustment_rad'] for r in block]
    n=len(adjustments[0])
    cumulative=[sum(abs(a[j]) for a in adjustments) for j in range(n)]
    net=[sum(a[j] for a in adjustments) for j in range(n)]
    final=block[-1]['cartesian_alignment']['release_preparation']
    results.append(dict(first_step=block[0]['step'],last_step=block[-1]['step'],preparation_steps=len(block),
        correction_events=sum(any(abs(v)>1e-12 for v in a) for a in adjustments),
        summed_l2_correction_rad=sum(math.sqrt(sum(v*v for v in a)) for a in adjustments),
        cumulative_abs_correction_per_joint_rad=cumulative,net_correction_per_joint_rad=net,
        preparation_wall_s=sum(r['step_s'] for r in block),
        final_tracking_error_rad=final['tracking_error_rad'],final_movement_rad=final['settled_movement_rad'],
        final_stable_steps=final['stable_steps']))
result=dict(label=label,blocks=results,recorded_wait_flags=waits,
    interpretation='Actual logged release-reference adjustment increments only; not total arm motion. Preparation steps include adaptation and settling. Unlogged wait causes are not inferred.')
out=P/'analysis-full'/label/'release-adjustment-audit.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(label=label,blocks=len(results),correction_events=sum(b['correction_events'] for b in results),preparation_steps=sum(b['preparation_steps'] for b in results))))

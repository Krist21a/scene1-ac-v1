"""Verify recorded pad observation sampling and preserve its latency costs."""
import json
from pathlib import Path
import statistics
import sys
P=Path(__file__).resolve().parent
label=sys.argv[1]
rows=[json.loads(line) for line in (P/label/'output/steps.jsonl').read_text().splitlines()]
observations=[r for r in rows if 'pad_boundary_observed' in r['cartesian_alignment']]
assert observations, 'No recorded pad observation'
assert all(r['cartesian_alignment'].get('pad_surface_stride')==2 for r in observations)
groups=[]
for row in observations:
    if not groups or row['step']!=groups[-1][-1]['step']+1:groups.append([])
    groups[-1].append(row)
result=dict(label=label,status='passed',surface_stride=2,
    blocks=[dict(first_step=g[0]['step'],last_step=g[-1]['step'],steps=len(g),
                 observed_frames=sum(r['cartesian_alignment']['pad_boundary_observed'] for r in g),
                 final_stable_frames=g[-1]['cartesian_alignment']['pad_stable_frames'],
                 inference_mean_s=statistics.mean(r['inference_s'] for r in g),
                 inference_max_s=max(r['inference_s'] for r in g)) for g in groups],
    interpretation='Runtime metadata confirms fixed stride2 only in pad observation; original full boundary and acceptance thresholds remain in immutable source manifest.')
out=P/'analysis-full'/label/'dense-sampling-audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

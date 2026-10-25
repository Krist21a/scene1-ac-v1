"""Compare measured pre-release joint speeds; nearest callback is not exact-cycle truth."""
import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent
out={}
for label in sys.argv[1:]:
 rows=[json.loads(s) for s in (p/label/'output/steps.jsonl').read_text().splitlines()]
 events=[(i,r) for i,r in enumerate(rows) if r['cartesian_alignment'].get('release_reference_transfer')]
 selected={r['step']:dict(step=r['step'],time=r['send_monotonic_s'],source_stamp=r.get('observation_receipts',{}).get('/sensors_data_raw',{}).get('source_stamp_s'),observation=r['observation_state'][8:15],sent=r['sent_action'][8:15],prep=r['cartesian_alignment'].get('release_preparation')) for i,_ in events for r in rows[max(0,i-3):i+1]}
 for line in (p/label/'chain.jsonl').open():
  a=json.loads(line)
  if a.get('topic')!='/sensors_data_raw':continue
  t=a['callback_monotonic_ns']/1e9
  for r in selected.values():
   distance=abs(t-r['time'])
   if distance>.025 or distance>=r.get('distance',1.):continue
   r.update(distance=distance,q=a['message']['joint_data']['joint_q'][20:27],v=a['message']['joint_data']['joint_v'][20:27])
 report=[]
 for i,row in events:
  window=[selected[r['step']] for r in rows[max(0,i-3):i+1]]
  report.append(dict(release_step=row['step'],samples=window,max_speed_before_release_rad_s=max(abs(v) for r in window[:-1] for v in r.get('v',[])),release_speed_rad_s=window[-1].get('v')))
 out[label]=report
 print(json.dumps({label:[{k:v for k,v in r.items() if k!='samples'} for r in report]}),flush=True)
(p/'release-speed-comparison.json').write_text(json.dumps(out,indent=2)+'\n')

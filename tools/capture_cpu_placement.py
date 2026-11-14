"""Read process/thread CPU affinity and frequency without changing runtime state."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from cpu_allocation import LAYOUT, STORE

p = argparse.ArgumentParser(); p.add_argument('label'); a = p.parse_args()
import re
if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', a.label): raise ValueError('Invalid label')
result = dict(at=datetime.datetime.now().astimezone().isoformat(), containers={}, frequency_khz={})
for container, expected in LAYOUT.items():
    init = int(subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',container],text=True))
    top = subprocess.check_output(['docker','top',container,'-eo','pid,comm'],text=True)
    processes = []
    for line in top.splitlines()[1:]:
        pid, name = line.split(maxsplit=1);pid=int(pid)
        if pid == init: continue
        try:
            command = (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            masks = {}
            for task in (Path('/proc')/str(pid)/'task').iterdir():
                try:
                    mask=next(l.split(':',1)[1].strip() for l in (task/'status').read_text().splitlines() if l.startswith('Cpus_allowed_list:'))
                    masks[mask]=masks.get(mask,0)+1
                except FileNotFoundError: pass
            processes.append(dict(pid=pid,name=name,command=command,masks=masks))
        except FileNotFoundError: pass
    result['containers'][container]=dict(expected=expected,processes=processes)
for f in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'):
    result['frequency_khz'][f.parent.name]=int(f.read_text())
output=STORE/(a.label+'-cpu-samples.jsonl')
with output.open('a') as f:f.write(json.dumps(result)+'\n')
print(json.dumps({c:[dict(name=p['name'],masks=p['masks']) for p in d['processes'] if 'run_scene1.py' in p['command'] or p['name'].startswith(('humanoid_sqp','nodelet'))] for c,d in result['containers'].items()}))

import json
from pathlib import Path
import subprocess
import sys
import time
time.sleep(float(sys.argv[1]) if len(sys.argv)>1 else 0)
p=Path(__file__).resolve().parent
s=json.loads((p/'successor-validation-20261025.json').read_text())
if s.get('phase')=='frozen_acceptance' and (p/'frozen-pad-acceptance-20261025.json').exists():
    s=json.loads((p/'frozen-pad-acceptance-20261025.json').read_text())
print(json.dumps({k:s.get(k) for k in ['status','current_trial','phase','error']}),flush=True)
label=s['current_trial']
if (p/label/'experiment.json').exists():
    subprocess.run([sys.executable,str(p/'status_brief.py'),label,'0'],check=True)

import json
from pathlib import Path
import subprocess
import sys
import time

time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
p = Path(__file__).resolve().parent
state = json.loads((p / 'frozen-acceptance-20261025.json').read_text())
print(json.dumps({k:state.get(k) for k in ['status','current_trial','error']}, ensure_ascii=False), flush=True)
print(json.dumps(dict(finished=len(state['trials']), visual_completed=sum(bool(x['visual_complete']) for x in state['trials'])), ensure_ascii=False), flush=True)
label = state.get('current_trial')
if label and (p / label / 'experiment.json').exists():
    subprocess.run([sys.executable, str(p / 'status_brief.py'), label, '0'], check=True)

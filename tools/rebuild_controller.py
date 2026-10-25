"""Run inside the original simulator build environment; isolated output only."""
import argparse,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('component',choices=['A','C']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
name='candidate-reference-origin' if a.component=='A' else 'candidate-wrist-pd'
src=root/'controllers'/name;m=json.loads((src/'build-manifest.json').read_text())
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
old=Path(m['compile'][m['compile'].index('-c')+1]).parent
source=Path(m['compile'][m['compile'].index('-c')+1]).name
assert hashlib.sha256((src/source).read_bytes()).hexdigest()==m['source_sha256']
(a.output/source).write_bytes((src/source).read_bytes())
for key in ['compile','link']:
 cmd=[str(a.output)+x[len(str(old)):] if x.startswith(str(old)+'/') else x for x in m[key]]
 # All other original build dependencies remain required in the simulator image.
 target=Path(cmd[cmd.index('-o')+1]);assert target.parent==a.output
 subprocess.run(cmd,cwd=m['cwd'],check=True)
print('Built isolated artifact:',target,'sha256',hashlib.sha256(target.read_bytes()).hexdigest())

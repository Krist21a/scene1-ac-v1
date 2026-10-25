"""Exercise cleanup with a non-ROS process that ignores INT and TERM."""
import json,subprocess,time
from pathlib import Path
import environment as e

e.wait_idle(e.ALGO)
code="import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"
p=subprocess.Popen(['docker','exec',e.ALGO,'python3','-c',code,'run_scene1.py'],stdout=subprocess.PIPE,text=True)
assert p.stdout.readline().strip()=='ready'
start=time.monotonic();receipt=json.loads(e.stop(e.ALGO,'/run_scene1.py'));result=p.wait(timeout=5)
assert result!=0 and [r['signal'] for r in receipt['events']]==[2,15,9]
assert time.monotonic()-start<8
e.wait_idle(e.ALGO)
out=dict(non_ros_fixture=True,events=receipt['events'],returncode=result,elapsed_s=time.monotonic()-start,idle_after=True)
(Path(__file__).parent/'bounded-shutdown-test.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))

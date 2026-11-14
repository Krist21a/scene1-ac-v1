"""Host-side JSON client; archives every response and pulls camera artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--socket', default='/tmp/scene1-model.sock')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--container', default='kuavo-ros-noetic-s12-ready')
    ap.add_argument('--root', default='/tmp/scene1-model-experiment')
    args = ap.parse_args()
    request = json.load(sys.stdin)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if request.get('op') in ('move_ee', 'set_gripper', 'set_head'):
        request.setdefault('command_id', uuid.uuid4().hex)
    started_wall_s = time.time()
    started = time.monotonic()
    result = subprocess.run(['docker', 'exec', '-i', args.container, 'python3', args.root+'/model_control/client.py', '--socket', args.socket],
                            input=json.dumps(request), text=True, capture_output=True, timeout=70)
    roundtrip_s = time.monotonic()-started
    if not result.stdout:
        raise RuntimeError(result.stderr)
    response = json.loads(result.stdout)
    rid = uuid.uuid4().hex
    (args.run_dir/(rid+'.json')).write_text(json.dumps(dict(request=request,response=response,started_wall_s=started_wall_s,roundtrip_s=roundtrip_s),indent=2)+'\n')
    observation = response.get('observation', response)
    for key in ('image', 'depth'):
        source = observation.get(key)
        if source:
            dest = args.run_dir/Path(source).name
            with dest.open('wb') as stream:
                subprocess.run(['docker','exec',args.container,'cat',source],stdout=stream,check=True)
            observation['host_'+key] = str(dest)
    (args.run_dir/'last_response.json').write_text(json.dumps(response,indent=2)+'\n')
    if observation.get('observation_id'):
        (args.run_dir/'latest.json').write_text(json.dumps(response,indent=2)+'\n')
    if args.summary:
        output = {k:v for k,v in response.items() if k != 'observation'} if 'observation' in response else {k:response[k] for k in ('session_id','status','command_id','error','cursor','active','fault','observation_status') if k in response}
        output['observation'] = {k:observation.get(k) for k in ('observation_id','left','right','host_image','state')}
        if response.get('error'):
            output['error'] = response['error']
        print(json.dumps(output,indent=2))
    else:
        print(json.dumps(response,indent=2))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

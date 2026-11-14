"""Watch compact action events without blocking on observation persistence."""
import argparse
import atexit
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/tmp/scene1-model.sock')
    parser.add_argument('--session', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--timeout', type=float, default=75.)
    parser.add_argument('--output', help='Optional new JSONL receipt file (never overwrites)')
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error('timeout must be in (0,3600] seconds')
    deadline = time.monotonic()+args.timeout
    receipt = None if args.output is None else open(args.output, 'x', encoding='utf-8', buffering=1)
    if receipt is not None:
        atexit.register(receipt.close)

    def request(op, **fields):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Watch deadline expired; action has NOT been cancelled')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(min(5., remaining))
            connection.connect(args.socket)
            payload = dict(op=op, session_id=args.session, **fields)
            connection.sendall((json.dumps(payload)+'\n').encode())
            with connection.makefile('r') as stream:
                response = json.loads(stream.readline())
        if response.get('status') == 'error':
            raise RuntimeError(response.get('error', response))
        return response

    def emit(value):
        line = json.dumps(dict(value, watch_received_wall_s=time.time()), ensure_ascii=False)
        if receipt is not None:
            receipt.write(line+'\n')
        print(line, flush=True)

    # Establish a cursor before querying the result, so a terminal event cannot
    # fall into a race between the initial query and subscription.
    cursor = request('events', after=0)['cursor']
    result = request('result', command_id=args.command)
    if result['status'] != 'running':
        emit(dict(kind='result_query', result=result))
        return
    emit(dict(kind='running', command_id=args.command, phase=result.get('phase', 'executing')))
    while True:
        batch = request('events', after=cursor)
        cursor = batch['cursor']
        for event in batch['events']:
            cid = event.get('command_id', event.get('result', {}).get('command_id'))
            if cid != args.command or event['kind'] not in ('progress', 'result'):
                continue
            emit(event)
            if event['kind'] == 'result':
                return
        # A result query also handles a feedback-window gap without assuming
        # that a missing event means an action failed or should be repeated.
        result = request('result', command_id=args.command)
        if result['status'] != 'running':
            emit(dict(kind='result_query', result=result))
            return
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Watch deadline expired; action has NOT been cancelled')
        time.sleep(min(.2, remaining))


if __name__ == '__main__':
    main()

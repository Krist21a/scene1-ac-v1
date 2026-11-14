"""Send one JSON request from stdin to the local scene1 bridge."""
import argparse
import json
import socket
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', default='/tmp/scene1-model.sock')
    args = parser.parse_args()
    request = json.load(sys.stdin)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(70)
        connection.connect(args.socket)
        connection.sendall((json.dumps(request, allow_nan=False)+'\n').encode())
        with connection.makefile('r') as stream:
            result = json.loads(stream.readline())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get('status') in ('error', 'rejected_or_failed', 'timeout'):
        sys.exit(1)


if __name__ == '__main__':
    main()

"""Record the live MuJoCo window at wall-clock speed; never label success.

Run on the host with DISPLAY/XAUTHORITY from the current desktop session.
SIGINT/SIGTERM finalize the MP4 and its timestamp/hash receipt. This recorder
reads only display pixels and never subscribes to simulator truth topics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New .mp4 file')
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('fps must be between 1 and 60')
    path = args.output.absolute()
    if path.suffix.lower() != '.mp4':
        parser.error('output must end in .mp4')
    if path.exists() or path.with_suffix('.json').exists():
        parser.error('output/receipt already exists; use a new run filename')
    path.parent.mkdir(parents=True, exist_ok=True)
    display = os.environ.get('DISPLAY', ':0')
    env = dict(os.environ, DISPLAY=display)
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], env=env, text=True)
    matches = re.findall(r'(0x[0-9a-f]+) "([^"\n]*(?:[Mm][Uu][Jj][Oo][Cc][Oo]|[Ss]imulate)[^"\n]*)"', tree)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one MuJoCo window, found {matches}')
    window, title = matches[0]
    info = subprocess.check_output(['xwininfo', '-id', window], env=env, text=True)
    width, height = [int(re.search(r'\b'+name+r':\s*(\d+)', info).group(1)) for name in ('Width', 'Height')]
    # Encode odd-sized windows without resizing or cropping scene content.
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n',
               '-f', 'x11grab', '-draw_mouse', '0', '-framerate', str(args.fps),
               '-window_id', str(int(window, 16)), '-video_size', f'{width}x{height}',
               '-i', display, '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-c:v', 'h264_nvenc', '-preset', 'p1', '-rc', 'vbr', '-cq', '26',
               '-b:v', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    receipt = dict(output=str(path), window_id=window, window_title=title,
                   frame_rate=args.fps, width=width, height=height, encoder='h264_nvenc',
                   speed=1.0, scope='live simulation display only', task_success=None,
                   started_wall_time_s=time.time(), started_monotonic_s=time.monotonic(),
                   recorder_pid=os.getpid(), status='recording')
    running = True
    def stop(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with path.with_suffix('.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt['ffmpeg_pid'] = process.pid
        path.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
        try:
            while running and process.poll() is None:
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                receipt['error'] = 'Encoder did not finalize within 30 seconds'
    receipt.update(stopped_wall_time_s=time.time(), stopped_monotonic_s=time.monotonic(),
                   returncode=process.returncode)
    if process.returncode not in (0, 255) or not path.is_file() or not path.stat().st_size:
        receipt.update(status='failed', error=receipt.get('error', 'Encoder failed; inspect .log'))
    else:
        try:
            metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format',
                                                          '-show_streams', '-of', 'json', str(path)], text=True))
            digest = hashlib.sha256()
            with path.open('rb') as video:
                for block in iter(lambda: video.read(1024*1024), b''):
                    digest.update(block)
            receipt.update(status='finalized', duration_s=float(metadata['format']['duration']),
                           bytes=path.stat().st_size, sha256=digest.hexdigest(),
                           streams=metadata['streams'])
        except Exception as exc:
            receipt.update(status='failed', error=str(exc))
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    if receipt['status'] != 'finalized':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

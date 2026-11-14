"""Record only the simulation window, continuously at original wall-clock speed."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


class WindowVideo:
    def __init__(self, directory):
        self.path = Path(directory)/'four-parcel-full.mp4'
        self.environment = dict(os.environ, DISPLAY=':0', XAUTHORITY='/run/user/1000/gdm/Xauthority')
        tree = subprocess.check_output(['xwininfo','-root','-tree'],env=self.environment,text=True)
        matches = re.findall(r'(0x[0-9a-f]+) "([^"\n]*(?:[Mm][Uu][Jj][Oo][Cc][Oo]|[Ss]imulate)[^"\n]*)"',tree)
        if len(matches) != 1:
            raise RuntimeError('Expected one MuJoCo window for recording: '+repr(matches))
        window, title = matches[0]
        # MuJoCo's Tab/Shift-Tab shortcuts only toggle the display panels.
        # Activate first: GLFW ignores synthetic SendEvent key delivery.
        for command in (['xdotool','windowactivate','--sync',window],
                        ['xdotool','key','--clearmodifiers','Tab','shift+Tab'],
                        ['xdotool','windowsize',window,'1280','900'],
                        ['xdotool','windowmove',window,'0','0'],['xdotool','windowraise',window]):
            subprocess.run(command,env=self.environment,check=True,stdout=subprocess.DEVNULL,timeout=10)
        time.sleep(1.)
        self.log = (Path(directory)/'video-recorder.log').open('w')
        self.started = time.monotonic()
        self.process = subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','warning',
            '-f','x11grab','-draw_mouse','0','-framerate','15','-window_id',str(int(window,16)),
            '-video_size','1280x900','-i',':0','-an','-c:v','h264_nvenc','-preset','p1',
            '-rc','vbr','-cq','26','-b:v','0','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],
            env=self.environment,stdout=self.log,stderr=subprocess.STDOUT)
        self.receipt = dict(window_title=title,window_id=window,frame_rate=15,
            width=1280,height=900,encoder='h264_nvenc',speed=1.,
            started_monotonic_s=self.started,scope='simulation window only',ui_panels_hidden=True)
        time.sleep(1.)
        if self.process.poll() is not None:
            self.log.close()
            raise RuntimeError('Video recorder failed before policy; inspect video-recorder.log')

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        try:self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait();raise RuntimeError('Video recorder did not finalize')
        finally:self.log.close()
        self.receipt['stopped_monotonic_s'] = time.monotonic()
        self.receipt['returncode'] = self.process.returncode
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError('Recorded video missing')
        metadata = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format',
            '-show_streams','-of','json',str(self.path)],text=True))
        self.receipt['duration_s'] = float(metadata['format']['duration'])
        self.receipt['bytes'] = self.path.stat().st_size
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        self.receipt['sha256'] = digest.hexdigest()
        self.path.with_suffix('.json').write_text(json.dumps(self.receipt,indent=2)+'\n')
        return self.receipt

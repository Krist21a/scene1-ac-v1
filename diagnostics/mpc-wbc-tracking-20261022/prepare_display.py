"""Copy the current X11 cookie without printing it; require idle simulation."""
import os
from pathlib import Path
import subprocess
import tempfile
import experiment as e

e.wait_idle(e.ALGO)
e.wait_idle(e.SIM)
candidates = [os.environ.get('XAUTHORITY'), str(Path.home() / '.Xauthority'),
              '/run/user/' + str(os.getuid()) + '/gdm/Xauthority']
records = ''
source = None
for candidate in candidates:
    if not candidate or not Path(candidate).is_file():
        continue
    p = subprocess.run(['xauth', '-f', candidate, 'nlist', ':0'], text=True, capture_output=True, check=True)
    if p.stdout.strip():
        records = '\n'.join('ffff' + line[4:] for line in p.stdout.splitlines()) + '\n'
        source = candidate
        break
if not records:
    raise SystemExit('No current :0 X11 cookie found; check the desktop DISPLAY/XAUTHORITY session.')
with tempfile.TemporaryDirectory(prefix='scene1-tracking-display-') as tmp:
    authority = Path(tmp) / 'Xauthority'
    authority.touch(mode=0o600)
    subprocess.run(['xauth', '-f', str(authority), 'nmerge', '-'], input=records, text=True, check=True)
    subprocess.run(['docker', 'cp', str(authority), e.SIM + ':/tmp/scene1-current-display.Xauthority'], check=True)
print('Current display authorization copied from ' + source + '; cookie contents not printed.')

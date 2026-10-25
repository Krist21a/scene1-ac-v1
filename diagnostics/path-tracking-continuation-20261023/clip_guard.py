"""Consecutive published-target clipping guard for future limited task trials."""


class ClipGuard:
    def __init__(self, threshold):
        assert threshold >= 1
        self.threshold = threshold
        self.last_step = None
        self.consecutive = 0
        self.trigger = None

    def consume(self, rows):
        for row in rows:
            step = row['step']
            if self.last_step is not None and step <= self.last_step:
                continue
            if self.last_step is not None and step != self.last_step + 1:
                self.consecutive = 0
            self.last_step = step
            self.consecutive = self.consecutive + 1 if row['clipped'] else 0
            if self.consecutive >= self.threshold and self.trigger is None:
                self.trigger = dict(step=step, consecutive=self.consecutive, threshold=self.threshold)
        return self.trigger


# Executed inside the algorithm container, prints only clipping flags and cursor.
READ_NEW_STEPS = '''import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);offset=int(sys.argv[2]);rows=[]
if p.exists():
 with p.open('rb') as f:
  f.seek(offset);data=f.read(1024*1024)
 last=data.rfind(b'\\n')
 if last>=0:
  for line in data[:last+1].splitlines():
   r=json.loads(line);rows.append(dict(step=r['step'],clipped=bool(r['motion']['bounds_clipped'])))
  offset+=last+1
print(json.dumps(dict(offset=offset,rows=rows)))
'''

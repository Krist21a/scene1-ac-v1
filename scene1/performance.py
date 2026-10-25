"""Wall-clock spans covering a complete policy cycle, including diagnostics."""
import json
import time
import cProfile


class CycleTimer:
    def __init__(self, path, enabled=False, clock=time.monotonic, profile_every=0):
        self.clock = clock
        self.path = path
        self.profile_every = profile_every
        self.profiler = cProfile.Profile() if enabled and profile_every else None
        self.stream = path.open('w', buffering=65536) if enabled else None
        self.row = None

    def begin(self, step, ros_time):
        if self.stream is None:
            return
        self.finish()
        self.started = self.last = self.clock()
        self.row = dict(step=step, started_monotonic_s=self.started,
                        ros_start_s=ros_time, spans_s={})
        if self.profiler is not None and step % self.profile_every == 0:
            self.profiler.enable()
            self.row['profiled'] = True

    def mark(self, name, **metadata):
        if self.row is None:
            return
        now = self.clock()
        self.row['spans_s'][name] = self.row['spans_s'].get(name, 0.) + now-self.last
        self.row.update(metadata)
        self.last = now

    def finish(self, **metadata):
        if self.row is None:
            return
        if self.profiler is not None:
            self.profiler.disable()
        self.mark('tail', **metadata)
        self.row['cycle_s'] = self.last-self.started
        self.stream.write(json.dumps(self.row, separators=(',', ':'))+'\n')
        self.row = None

    def close(self):
        self.finish(incomplete=True)
        if self.stream is not None:
            self.stream.close()
        if self.profiler is not None:
            self.profiler.dump_stats(str(self.path.with_suffix('.pstats')))

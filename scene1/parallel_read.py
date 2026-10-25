"""Overlap one current-cycle sensor read with independent image detection."""
from concurrent.futures import ThreadPoolExecutor


class CycleRead:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scene1-rgbd')
        self.pending = None

    def begin(self, function, *args):
        if self.pending is not None:
            raise RuntimeError('Previous cycle RGB-D read has not been consumed')
        self.pending = self.pool.submit(function, *args)

    def result(self):
        if self.pending is None:
            raise RuntimeError('No current-cycle RGB-D read')
        try:
            return self.pending.result()
        finally:
            self.pending = None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending = None

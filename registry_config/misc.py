import time
from registry_config.logger import logger

class Perf(object):
    def __init__(self, banner, freq = 25, logger = logger):
        self._banner = banner
        self._freq = freq
        self._logger = logger
        self._start_tic = 0
        self._duration = 0
        self._count = 0

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end()

    def start(self):
        self._start_tic = time.time()

    def end(self):
        self._duration += time.time() - self._start_tic
        self._count += 1
        if self._count % self._freq == 0 and self._count > 0:
            avg_duration = self._duration / self._count
            self._logger.info(
                f'[perf] {self._banner} 
                avg cost: {avg_duration:.3f} s'
            )
            self._duration = 0
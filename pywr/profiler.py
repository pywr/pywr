import time
import resource
import sys
from dataclasses import dataclass, asdict
import pandas


def memory_usage() -> float:
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom


@dataclass
class ProfilerEntry:
    phase: str
    class_name: str
    name: str
    perf_counter: float
    memory_usage: float


class Profiler:
    """A basic memory consumption profiler"""
    def __init__(self):
        self.entries = []

        self._last_perf_counter = None
        self._last_max_rss = None
        self.reset()

    def reset(self):
        self._last_perf_counter = time.perf_counter()
        self._last_max_rss = memory_usage()

    def checkpoint(self, phase: str, class_name: str, name: str):

        max_rss = memory_usage()
        perf_counter = time.perf_counter()

        self.entries.append(
            ProfilerEntry(
                phase=phase,
                class_name=class_name,
                name=name,
                perf_counter=perf_counter-self._last_perf_counter,
                memory_usage=max_rss-self._last_max_rss
            )
        )

        self.reset()

    def to_dataframe(self):
        return pandas.DataFrame([asdict(e) for e in self.entries]).set_index(['phase', 'class_name', 'name'])







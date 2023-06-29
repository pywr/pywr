import time
from typing import Optional

try:
    import resource
except ModuleNotFoundError:
    resource = None
import sys
from dataclasses import dataclass, asdict
import pandas


def memory_usage() -> Optional[float]:
    if resource is None:
        return None
    rusage_denom = 1024.0
    if sys.platform == "darwin":
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom


@dataclass
class ProfilerEntry:
    phase: str
    class_name: str
    name: str
    perf_counter: float
    memory_usage: Optional[float]


class Profiler:
    """A basic memory and performance counter profiler.

    This profiler can be used to record memory performance counter increases between checkpoints. It uses
    the resource module from the Python standard library for tracking MAXRSS. This module is only available
    on certain platforms (see Python documentation). If the resource module is not available no memory tracking
    is undertaken.
    """

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

        if max_rss is None or self._last_max_rss is None:
            delta_rss = None
        else:
            delta_rss = max_rss - self._last_max_rss

        self.entries.append(
            ProfilerEntry(
                phase=phase,
                class_name=class_name,
                name=name,
                perf_counter=perf_counter - self._last_perf_counter,
                memory_usage=delta_rss,
            )
        )

        self.reset()

    def to_dataframe(self) -> pandas.DataFrame:
        """Return the checkpoints as a `pandas.DataFrame`"""
        return pandas.DataFrame([asdict(e) for e in self.entries]).set_index(
            ["phase", "class_name", "name"]
        )

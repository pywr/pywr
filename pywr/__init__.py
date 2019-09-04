#!/usr/bin/env python
import os
import sys
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

if sys.platform == "win32":
    libdir = os.path.join(os.path.dirname(__file__), ".libs")
    os.environ["PATH"] = os.environ["PATH"] + ";" + libdir

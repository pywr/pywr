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
    dll_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".libs"))
    if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
        # https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
        if os.path.exists(dll_folder):
            os.add_dll_directory(dll_folder)
    else:
        os.environ["PATH"] = os.environ["PATH"] + ";" + dll_folder

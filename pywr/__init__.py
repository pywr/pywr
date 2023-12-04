#!/usr/bin/env python
import os
import sys

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

if sys.platform == "win32":
    dll_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".libs"))
    if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
        # https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew
        if os.path.exists(dll_folder):
            os.add_dll_directory(dll_folder)
    else:
        os.environ["PATH"] = os.environ["PATH"] + ";" + dll_folder

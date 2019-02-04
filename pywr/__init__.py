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


def get_git_hash():
    """Get the git hash for this build, if available"""
    try:
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, "GIT_VERSION.txt")
        with open(path, "r") as f:
            data = f.read().rstrip()
    except FileNotFoundError:
        data = None
    return data

__git_hash__ = get_git_hash()

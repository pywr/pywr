#!/usr/bin/env python
import os

__version__ = "0.5.1"


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

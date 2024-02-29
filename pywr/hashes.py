""" Generic utilities functions useful for pywr, but not specific to any of the core modules
"""

import io
import hashlib


def compute_hash(filename, algorithm="md5", chunk_size=io.DEFAULT_BUFFER_SIZE):
    """Compute the hash of a large file using hashlib"""
    h = hashlib.new(algorithm)

    with io.open(filename, mode="rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def check_hash(filename, hash, cache=None, algorithm="md5", **kwargs):
    """Check the hash for filename using the named algorithm

    If a cache provided, it is checked to see if the file hash has already been
    computed. If it has not, a hash is computed for the file using the given
    algorithm. The cached/computed cache is then compared against the provided
    hash. If the hashes do not match a HashMismatchError error is raised.

    This function is not case sensitive.
    """

    if cache is not None:
        if (cached_hash := cache.get((filename, algorithm))) is not None:
            if cached_hash.lower() == hash.lower():
                return
            else:
                raise HashMismatchError(
                    f'Hash mismatch using {algorithm} on file: "{filename}"'
                )

    actual_hash = compute_hash(filename, algorithm=algorithm, **kwargs)

    if hash.lower() != actual_hash.lower():
        raise HashMismatchError(
            'Hash mismatch using {} on file: "{}"'.format(algorithm, filename)
        )

    if cache is not None:
        cache[(filename, algorithm)] = actual_hash


class HashMismatchError(IOError):
    pass

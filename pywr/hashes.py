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


class HashMismatchError(IOError):
    pass


def check_hash(filename, hash, algorithm="md5", **kwargs):
    """Check the hash for filename using the named algorithm

    If the hashes do not match a HashMismatchError error is raised.
    """

    actual_hash = compute_hash(filename, algorithm=algorithm, **kwargs)

    if hash != actual_hash:
        raise HashMismatchError(
            'Hash mismatch using {} on file: "{}"'.format(algorithm, filename)
        )

import os
import pytest
from pywr import hashes


TEST_FOLDER = os.path.dirname(__file__)


@pytest.mark.parametrize(
    "filename, algorithm, hash, correct",
    [
        ("timeseries2.csv", "md5", "a5c4032e2d8f5205ca99dedcfa4cd18e", True),
        (
            "timeseries2.csv",
            "sha256",
            "0f75b3cee325d37112687d3d10596f44e0add374f4e40a1b6687912c05e65366",
            True,
        ),
        ("timeseries2.h5", "md5", "0f6c65a36851c89c7c4e63ab1893554b", True),
        # This next one is the sha256 hash, but is given as md5. Therefore it should fail.
        (
            "timeseries2.h5",
            "md5",
            "1272702d60694f3417b910fb158e717de4fccdbf6aa10aa37f1c95cd78f8075e",
            False,
        ),
    ],
)
def test_hash_timeseries2_(filename, algorithm, hash, correct):
    """Test the hash value of files in the models directory"""

    fullname = os.path.join(TEST_FOLDER, "models", filename)

    if correct:
        hashes.check_hash(fullname, hash, algorithm=algorithm)
    else:
        with pytest.raises(hashes.HashMismatchError):
            hashes.check_hash(fullname, hash, algorithm=algorithm)

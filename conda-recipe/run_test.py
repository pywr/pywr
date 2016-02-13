import os
import sys
import pytest

tests_folder = os.path.join(os.environ['SRC_DIR'], 'tests')
result = pytest.main(['-x', tests_folder])

if result != 0:
    # test(s) failed, raise error
    sys.exit(result)

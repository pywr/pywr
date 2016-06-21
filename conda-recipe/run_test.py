import os
import sys
import pytest

os.chdir(os.environ['SRC_DIR'])
tests_folder = os.path.join(os.environ['SRC_DIR'], 'tests')

for solver in ('glpk', 'lpsolve'):
    result = pytest.main(['-x', tests_folder, '--solver={}'.format(solver)])
    if result != 0:
        # test(s) failed, raise error
        sys.exit(result)

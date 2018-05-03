#!/bin/sh

# clean up before building
# assumes all .c files were compiled from .pyx and should be removed
rm -rf build dist
find . | grep -E "(__pycache__|\.pyc|\.pyo|\.pyd|\.so|\.c$)" | xargs rm -rf

export C_INCLUDE_PATH=${PREFIX}/include

if [ `uname` == "Darwin" ]; then
  # define rpath (runtime search path for shared libraries)
  # conda will automatically convert this to a relocatable path in the package
  export CFLAGS="-Wl,-rpath,${PREFIX}/lib"
fi

if [ "$FEATURE_OLDGLIBC" == "1" ]; then
  # use old version of memcpy
  export CFLAGS="-I. -include conda-recipe/glibc_version_fix.h"
fi

$PYTHON setup.py build_ext --with-glpk --with-lpsolve install --single-version-externally-managed --record=record.txt

# Also build the source and wheel distributions for pypi
$PYTHON setup.py sdist
$PYTHON setup.py bdist_wheel
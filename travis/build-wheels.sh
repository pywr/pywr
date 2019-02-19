#!/bin/bash
# Build script for use with manylinux1 docker image to construct Linux wheels
# Note this script does not build the wheels with lpsolve support.
set -e -x

# Compile wheels

# Install the build dependencies of the project. If some dependencies contain
# compiled extensions and are not provided as pre-built wheel packages,
# pip will build them from source using the MSVC compiler matching the
# target Python version and architecture
"${PYBIN}/pip" install cython packaging numpy jupyter pytest wheel
PYWR_BUILD_GLPK="true" "${PYBIN}/pip" wheel /io/ -w wheelhouse/

# Bundle external shared libraries into the wheels
for whl in ${WORKDIR}/wheelhouse/pywr*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test

"${PYBIN}/pip" install pywr --no-index -f /io/wheelhouse
"pip install platypus-opt inspyred pygmo"
"${PYBIN}/pytest" ${WORKDIR}/pywr/tests

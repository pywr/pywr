#!/bin/bash
# Build script for use with manylinux1 docker image to construct Linux wheels

set -e -x

cd /io
# Setup path to use PYBIN's binary folder
export PATH=${PYBIN}:${PATH}

# Install the build dependencies of the project. If some dependencies contain
# compiled extensions and are not provided as pre-built wheel packages,
# pip will build them from source matching the target Python version and architecture
pip install cython packaging numpy jupyter pytest wheel future setuptools_scm
PYWR_BUILD_GLPK="true" PYWR_BUILD_LPSOLVE="true" python setup.py build_ext bdist_wheel -d wheelhouse/

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pywr*.whl; do
    auditwheel repair "$whl" -w wheelhouse/
done
# Install packages and test
pip install platypus-opt inspyred pygmo

# Move the source package to prevent import conflicts when running the tests
mv pywr pywr.build

# List the built wheels
ls -l wheelhouse

# Only test the manylinux wheels
for whl in wheelhouse/pywr*manylinux*.whl; do
    pip install --force-reinstall --ignore-installed "$whl"
    PYWR_SOLVER=glpk pytest tests
    PYWR_SOLVER=glpk-edge pytest tests
    PYWR_SOLVER=lpsolve pytest tests
done

if [[ "${BUILD_DOC}" -eq "1" ]]; then
  echo "Building documentation!"
  pip install sphinx sphinx_rtd_theme numpydoc
  cd docs
  make html
  mkdir -p /io/pywr-docs
  cp -r build/html /io/pywr-docs/
  cd -
fi
PYWR_SOLVER=glpk py.test -v ${SRC_DIR}/tests
PYWR_SOLVER=lpsolve py.test -v ${SRC_DIR}/tests

if [ "${PY3K}" == "1" ]; then
    PY_VER=3
else
    PY_VER=2
fi
jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python${PY_VER} ${SRC_DIR}/tests/notebook.ipynb


if [ "${BUILD_DOC}" == "1" ]; then
    echo "Building documentation!"
    cd ${SRC_DIR}/docs
    make html
    mkdir -p ${HOME}/pywr/docs
    cp -r ${SRC_DIR}/docs/build/html ${HOME}/pywr/docs/
    cd -
fi
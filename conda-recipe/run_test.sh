py.test ${SRC_DIR}/tests --solver=glpk
py.test ${SRC_DIR}/tests --solver=lpsolve

if [ "${PY3K}" == "1" ]; then
    PY_VER=3
else
    PY_VER=2
fi
jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python${PY_VER} ${SRC_DIR}/tests/notebook.ipynb

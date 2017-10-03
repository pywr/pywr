py.test %SRC_DIR%\tests --verbose --solver=glpk
py.test %SRC_DIR%\tests --verbose --solver=lpsolve

if errorlevel 1 exit 1

if %PY3K% == 1 (
    set PY_VER=3
) else (
    set PY_VER=2
)
jupyter nbconvert --to html --execute --ExecutePreprocessor.kernel_name=python%PY_VER% %SRC_DIR%\tests\notebook.ipynb


echo "Running tests in: %1";

SET PYWR_SOLVER=glpk
python -m pytest "%1"\tests || goto :error

SET PYWR_SOLVER=glpk-edge
python -m pytest "%1"\tests || goto :error

SET PYWR_SOLVER=lpsolve
python -m pytest "%1"\tests || goto :error

:error
echo Tests failed with error #%errorlevel%.
exit /b %errorlevel%

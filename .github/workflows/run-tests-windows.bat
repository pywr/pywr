
echo "Running tests in: %1";

SET PYWR_SOLVER=glpk
python -m pytest "%1"\tests

SET PYWR_SOLVER=glpk-edge
python -m pytest "%1"\tests

SET PYWR_SOLVER=lpsolve
python -m pytest "%1"\tests

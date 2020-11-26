#!/usr/bin/env bash

echo "Running tests in: $1";
PYWR_SOLVER="glpk" pytest "$1"/tests;
PYWR_SOLVER="glpk-edge" pytest "$1"/tests;
PYWR_SOLVER="lpsolve" pytest "$1"/tests;

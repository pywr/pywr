#!/usr/bin/env bash

# GLPK solver
echo "Running GLPK (route-based) tests in: $1";
PYWR_SOLVER="glpk" pytest "$1"/tests  || exit 1
PYWR_SOLVER_GLPK_UNSAFE_API=true PYWR_SOLVER="glpk" pytest "$1"/tests  || exit 1
PYWR_SOLVER_GLPK_FIXED_COSTS_ONCE=true PYWR_SOLVER_GLPK_FIXED_FLOWS_ONCE=true PYWR_SOLVER_GLPK_FIXED_FACTORS_ONCE=true PYWR_SOLVER="glpk" pytest "$1"/tests  || exit 1

# GLPK Edge solver
echo "Running GLPK (edge-based) tests in: $1";
PYWR_SOLVER="glpk-edge" pytest "$1"/tests  || exit 1
PYWR_SOLVER_GLPK_UNSAFE_API=true PYWR_SOLVER="glpk-edge" pytest "$1"/tests  || exit 1
PYWR_SOLVER_GLPK_FIXED_COSTS_ONCE=true PYWR_SOLVER_GLPK_FIXED_FLOWS_ONCE=true PYWR_SOLVER_GLPK_FIXED_FACTORS_ONCE=true PYWR_SOLVER="glpk-edge" pytest "$1"/tests  || exit 1

# Lpsolve solver
echo "Running LpSolve tests in: $1";
PYWR_SOLVER="lpsolve" pytest "$1"/tests || exit 1

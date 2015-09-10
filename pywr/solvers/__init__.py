#!/usr/bin/env python
from ..core import Solver
from .cython_glpk import CythonGLPKSolver as cy_CythonGLPKSolver
from .cython_lpsolve import CythonLPSolveSolver as cy_CythonLPSolveSolver


class CythonGLPKSolver(Solver):
    """Python wrapper of Cython GLPK solver.

    This is required to subclass Solver and get the metaclass magic.
    """
    name = 'GLPK'
    def __init__(self, *args, **kwargs):
        super(CythonGLPKSolver, self).__init__(*args, **kwargs)
        self._cy_solver = cy_CythonGLPKSolver()

    def setup(self, model):
        return self._cy_solver.setup(model)

    def solve(self, model):
        return self._cy_solver.solve(model)


class CythonLPSolveSolver(Solver):
    """Python wrapper of Cython LPSolve55 solver.

    This is required to subclass Solver and get the metaclass magic.
    """
    name = 'lpsolve'
    def __init__(self, *args, **kwargs):
        super(CythonLPSolveSolver, self).__init__(*args, **kwargs)
        self._cy_solver = cy_CythonLPSolveSolver()

    def setup(self, model):
        return self._cy_solver.setup(model)        

    def solve(self, model):
        return self._cy_solver.solve(model)

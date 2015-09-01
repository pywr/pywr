#!/usr/bin/env python
from ..core import Solver
from .pywr_glpk2 import SolverGLPK
from .cython_glpk import CythonGLPKSolver as cy_CythonGLPKSolver

class CythonGLPKSolver(Solver):
    """Python wrapper of Cython GLPK solver.

    This is required to subclass Solver and get the metaclass magic.
    """
    name = 'cyGLPK'
    def __init__(self, *args, **kwargs):
        super(CythonGLPKSolver, self).__init__(*args, **kwargs)
        self._cy_solver = cy_CythonGLPKSolver()

    def solve(self, model):
        return self._cy_solver.solve(model)

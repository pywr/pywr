"""
This module contains a Solver baseclass and several implemented subclasses.

Solvers are used to with pywr.core.Model classes to solve the network
allocation problem every time step.

Currently there are only linear programme based solvers using,
    - GLPK
    - LPSolve55

"""
from six import with_metaclass


class SolverMeta(type):
    """Solver metaclass used to keep a registry of Solver classes"""
    solvers = {}
    def __new__(cls, clsname, bases, attrs):
        newclass = super(SolverMeta, cls).__new__(cls, clsname, bases, attrs)
        cls.solvers[newclass.name.lower()] = newclass
        return newclass
    @classmethod
    def get_default(cls, ):
        return cls.solvers['glpk']


class Solver(with_metaclass(SolverMeta)):
    """Solver base class from which all solvers should inherit"""
    name = 'default'
    def setup(self, model):
        raise NotImplementedError('Solver should be subclassed to provide setup()')
    def solve(self, model, timestep):
        raise NotImplementedError('Solver should be subclassed to provide solve()')


# Attempt to import solvers. These will only be successful if they are built correctly.
try:
    from .cython_glpk import CythonGLPKSolver as cy_CythonGLPKSolver
except ImportError:
    pass
else:
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


try:
    from .cython_lpsolve import CythonLPSolveSolver as cy_CythonLPSolveSolver
except ImportError:
    pass
else:
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

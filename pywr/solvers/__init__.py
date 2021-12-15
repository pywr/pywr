"""
This module contains a Solver baseclass and several implemented subclasses.

Solvers are used to with pywr.core.Model classes to solve the network
allocation problem every time step.

Currently there are only linear programme based solvers using,
    - GLPK
    - LPSolve55
"""
import os

solver_registry = []


def _parse_env_kwarg(kwargs, keyword, env_name, env_type):
    """Insert a keyword argument from an environment."""
    if keyword not in kwargs:
        # Keyword not given by user; see if an environment variable is defined with one
        env_value = os.environ.get(env_name, None)
        if env_value is not None:
            if env_type is bool:
                kwargs[keyword] = env_value == "1" or env_value.lower() == "true"
            elif env_type is str:
                kwargs[keyword] = env_value
            elif env_type is float:
                kwargs[keyword] = float(env_value)
            elif env_type is int:
                kwargs[keyword] = int(env_value)
            else:
                raise NotImplementedError(
                    f"Parsing environment variables of type {env_type} is not supported."
                )
    return kwargs


class Solver(object):
    """Solver base class from which all solvers should inherit"""

    name = "default"

    def __init__(self, *args, **kwargs):
        pass

    def setup(self, model):
        raise NotImplementedError("Solver should be subclassed to provide setup()")

    def solve(self, model):
        raise NotImplementedError("Solver should be subclassed to provide solve()")

    def reset(self):
        raise NotImplementedError("Solver should be subclassed to provide reset()")

    @property
    def stats(self):
        return {}

    @property
    def settings(self):
        return {}


# Attempt to import solvers. These will only be successful if they are built correctly.
try:
    from .cython_glpk import (
        CythonGLPKSolver as cy_CythonGLPKSolver,
        GLPKError,
        GLPKInternalError,
    )
except ImportError:
    pass
else:

    class CythonGLPKSolver(Solver):
        """Python wrapper of Cython GLPK solver.

        This is required to subclass Solver and get the metaclass magic.
        """

        name = "glpk"

        def __init__(self, *args, **kwargs):
            super(CythonGLPKSolver, self).__init__(*args, **kwargs)
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_flows_once",
                "PYWR_SOLVER_GLPK_FIXED_FLOWS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_costs_once",
                "PYWR_SOLVER_GLPK_FIXED_COSTS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_factors_once",
                "PYWR_SOLVER_GLPK_FIXED_FACTORS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "use_unsafe_api",
                "PYWR_SOLVER_GLPK_UNSAFE_API",
                bool,
            )
            self._cy_solver = cy_CythonGLPKSolver(**kwargs)

        def setup(self, model):
            return self._cy_solver.setup(model)

        def solve(self, model):
            return self._cy_solver.solve(model)

        def reset(self):
            return self._cy_solver.reset()

        def dump_mps(self, filename):
            return self._cy_solver.dump_mps(filename)

        def dump_lp(self, filename):
            return self._cy_solver.dump_lp(filename)

        def dump_glpk(self, filename):
            return self._cy_solver.dump_glpk(filename)

        def retry_solve():
            def fget(self):
                return self._cy_solver.retry_solve

            def fset(self, value):
                self._cy_solver.retry_solve = value

            return locals()

        retry_solve = property(**retry_solve())

        def save_routes_flows():
            def fget(self):
                return self._cy_solver.save_routes_flows

            def fset(self, value):
                self._cy_solver.save_routes_flows = value

            return locals()

        save_routes_flows = property(**save_routes_flows())

        @property
        def routes(self):
            return self._cy_solver.routes

        @property
        def routes_flows_array(self):
            return self._cy_solver.route_flows_arr

        @property
        def stats(self):
            return self._cy_solver.stats

        @property
        def use_presolve(self):
            return self._cy_solver.use_presolve

        @property
        def use_unsafe_api(self):
            return self._cy_solver.use_unsafe_api

        @property
        def set_fixed_flows_once(self):
            return self._cy_solver.set_fixed_flows_once

        @property
        def set_fixed_costs_once(self):
            return self._cy_solver.set_fixed_costs_once

        @property
        def set_fixed_factors_once(self):
            return self._cy_solver.set_fixed_factors_once

        @property
        def settings(self):
            return {
                "use_presolve": self.use_presolve,
                "save_routes_flows": self.save_routes_flows,
                "retry_solve": self.retry_solve,
                "set_fixed_flows_once": self.set_fixed_flows_once,
                "set_fixed_costs_once": self.set_fixed_costs_once,
                "set_fixed_factors_once": self.set_fixed_factors_once,
            }

    solver_registry.append(CythonGLPKSolver)


try:
    from .cython_glpk import CythonGLPKEdgeSolver as cy_CythonGLPKEdgeSolver
except ImportError:
    pass
else:

    class CythonGLPKEdgeSolver(Solver):
        """Python wrapper of Cython GLPK solver.

        This is required to subclass Solver and get the metaclass magic.
        """

        name = "glpk-edge"

        def __init__(self, *args, **kwargs):
            super(CythonGLPKEdgeSolver, self).__init__(*args, **kwargs)
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_flows_once",
                "PYWR_SOLVER_GLPK_FIXED_FLOWS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_costs_once",
                "PYWR_SOLVER_GLPK_FIXED_COSTS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "set_fixed_factors_once",
                "PYWR_SOLVER_GLPK_FIXED_FACTORS_ONCE",
                bool,
            )
            kwargs = _parse_env_kwarg(
                kwargs,
                "use_unsafe_api",
                "PYWR_SOLVER_GLPK_UNSAFE_API",
                bool,
            )
            self._cy_solver = cy_CythonGLPKEdgeSolver(**kwargs)

        def setup(self, model):
            return self._cy_solver.setup(model)

        def solve(self, model):
            return self._cy_solver.solve(model)

        def reset(self):
            return self._cy_solver.reset()

        def dump_mps(self, filename):
            return self._cy_solver.dump_mps(filename)

        def dump_lp(self, filename):
            return self._cy_solver.dump_lp(filename)

        def dump_glpk(self, filename):
            return self._cy_solver.dump_glpk(filename)

        def retry_solve():
            def fget(self):
                return self._cy_solver.retry_solve

            def fset(self, value):
                self._cy_solver.retry_solve = value

            return locals()

        retry_solve = property(**retry_solve())

        @property
        def stats(self):
            return self._cy_solver.stats

        @property
        def use_presolve(self):
            return self._cy_solver.use_presolve

        @property
        def use_unsafe_api(self):
            return self._cy_solver.use_unsafe_api

        @property
        def set_fixed_flows_once(self):
            return self._cy_solver.set_fixed_flows_once

        @property
        def set_fixed_costs_once(self):
            return self._cy_solver.set_fixed_costs_once

        @property
        def set_fixed_factors_once(self):
            return self._cy_solver.set_fixed_factors_once

        @property
        def settings(self):
            return {
                "use_presolve": self.use_presolve,
                "retry_solve": self.retry_solve,
                "set_fixed_flows_once": self.set_fixed_flows_once,
                "set_fixed_costs_once": self.set_fixed_costs_once,
                "set_fixed_factors_once": self.set_fixed_factors_once,
            }

    solver_registry.append(CythonGLPKEdgeSolver)


try:
    from .cython_lpsolve import CythonLPSolveSolver as cy_CythonLPSolveSolver
except ImportError:
    pass
else:

    class CythonLPSolveSolver(Solver):
        """Python wrapper of Cython LPSolve55 solver.

        This is required to subclass Solver and get the metaclass magic.
        """

        name = "lpsolve"

        def __init__(self, *args, **kwargs):
            super(CythonLPSolveSolver, self).__init__(*args, **kwargs)
            self._cy_solver = cy_CythonLPSolveSolver(**kwargs)

        def setup(self, model):
            return self._cy_solver.setup(model)

        def solve(self, model):
            return self._cy_solver.solve(model)

        def reset(self):
            pass

        @property
        def save_routes_flows(self):
            return self._cy_solver.save_routes_flows

        @save_routes_flows.setter
        def save_routes_flows(self, value):
            self._cy_solver.save_routes_flows = value

        @property
        def routes(self):
            return self._cy_solver.routes

        @property
        def routes_flows_array(self):
            return self._cy_solver.route_flows_arr

        @property
        def stats(self):
            return self._cy_solver.stats

        @property
        def settings(self):
            return {
                "save_routes_flows": self.save_routes_flows,
            }

    solver_registry.append(CythonLPSolveSolver)


class NullSolver(Solver):
    """A null solver that performs no allocation routine.

    This solver does no allocation and is mainly useful for debugging purposes.
    """

    name = "null"

    def setup(self, model):
        pass

    def solve(self, model):
        pass

    def reset(self):
        pass


solver_registry.append(NullSolver)

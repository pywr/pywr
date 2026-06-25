import numpy as np
from . import BaseOptimisationWrapper
import logging

logger = logging.getLogger(__name__)


class PymooWrapper(BaseOptimisationWrapper):
    """A helper class for running pywr optimisations with pymoo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem = self._make_pymoo_problem()

    def _make_pymoo_problem(self):
        try:
            from pymoo.core.problem import ElementwiseProblem
        except ImportError:
            raise ImportError("pymoo is required. Install with: pip install pymoo")

        lower, upper = self._get_variable_bounds()
        n_var = len(lower)
        n_obj = len(self.model_objectives)
        n_ieq = self._count_ineq_constraints()
        n_eq = len([c for c in self.model_constraints if c.is_equality_constraint])
        wrapper = self

        class _ProblemImpl(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_ieq_constr=n_ieq,
                    n_eq_constr=n_eq,
                    xl=lower,
                    xu=upper,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                wrapper._pymoo_evaluate(x, out)

        return _ProblemImpl()

    def _get_variable_bounds(self):
        lower = []
        upper = []
        for var in self.model_variables:
            if var.double_size > 0:
                lower.append(var.get_double_lower_bounds())
                upper.append(var.get_double_upper_bounds())
            if var.integer_size > 0:
                lower.append(var.get_integer_lower_bounds())
                upper.append(var.get_integer_upper_bounds())
        return np.concatenate(lower), np.concatenate(upper)

    def _count_ineq_constraints(self):
        count = 0
        for c in self.model_constraints:
            if c.is_double_bounded_constraint:
                count += 2
            elif c.is_lower_bounded_constraint or c.is_upper_bounded_constraint:
                count += 1
        return count

    def _pymoo_evaluate(self, x, out):
        logger.info("Evaluating solution ...")

        for ivar, var in enumerate(self.model_variables):
            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar + 1])
            x_slice = np.array(x[j])
            if var.double_size > 0:
                var.set_double_variables(x_slice[: var.double_size])
            if var.integer_size > 0:
                ints = np.round(x_slice[-var.integer_size :]).astype(np.int32)
                var.set_integer_variables(ints)

        self.run_stats = self.model.run()

        objectives = []
        for r in self.model_objectives:
            sign = 1.0 if r.is_objective == "minimise" else -1.0
            objectives.append(sign * r.aggregated_value())
        out["F"] = np.array(objectives)

        ineq_constraints = []
        eq_constraints = []
        for r in self.model_constraints:
            val = r.aggregated_value()
            if r.is_double_bounded_constraint:
                ineq_constraints.append(r.constraint_lower_bounds - val)
                ineq_constraints.append(val - r.constraint_upper_bounds)
            elif r.is_equality_constraint:
                eq_constraints.append(val)
            elif r.is_lower_bounded_constraint:
                ineq_constraints.append(r.constraint_lower_bounds - val)
            elif r.is_upper_bounded_constraint:
                ineq_constraints.append(val - r.constraint_upper_bounds)
            else:
                raise RuntimeError(
                    f'The bounds of constraint "{r.name}" could not be identified correctly.'
                )

        if ineq_constraints:
            out["G"] = np.array(ineq_constraints)
        if eq_constraints:
            out["H"] = np.array(eq_constraints)

        logger.info(
            f"Evaluation completed in {self.run_stats.time_taken:.2f} seconds "
            f"({self.run_stats.speed:.2f} ts/s)."
        )

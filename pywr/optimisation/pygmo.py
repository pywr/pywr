import numpy as np
from . import BaseOptimisationWrapper

import logging

logger = logging.getLogger(__name__)


class PygmoWrapper(BaseOptimisationWrapper):
    def fitness(self, solution):
        logger.info("Evaluating solution ...")

        for ivar, var in enumerate(self.model_variables):
            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar + 1])
            var.set_double_variables(np.array(solution[j]).copy())

        self.model.reset()
        self.run_stats = self.model.run()

        objectives = []
        for r in self.model_objectives:
            sign = 1.0 if r.is_objective == "minimise" else -1.0
            value = r.aggregated_value()
            objectives.append(sign * value)

        # Return separate lists for equality and inequality constraints.
        # pygmo requires that inequality constraints are all of the form g(x) <= 0
        # Therefore these are converted to this form from their respective bounds.
        eq_constraints = []
        ineq_constraints = []
        for r in self.model_constraints:
            x = r.aggregated_value()
            if r.is_double_bounded_constraint:
                # Need to create two constraints
                ineq_constraints.append(r.constraint_lower_bounds - x)
                ineq_constraints.append(x - r.constraint_upper_bounds)
            elif r.is_equality_constraint:
                eq_constraints.append(x)
            elif r.is_lower_bounded_constraint:
                ineq_constraints.append(r.constraint_lower_bounds - x)
            elif r.is_upper_bounded_constraint:
                ineq_constraints.append(x - r.constraint_upper_bounds)
            else:
                raise RuntimeError(
                    f'The bounds if constraint "{r.name}" could not be identified correctly.'
                )

        # Return values to the solution
        logger.info(
            f"Evaluation completed in {self.run_stats.time_taken:.2f} seconds "
            f"({self.run_stats.speed:.2f} ts/s)."
        )
        return objectives + eq_constraints + ineq_constraints

    def get_bounds(self):
        """Return the variable bounds."""
        lower = []
        upper = []
        for var in self.model_variables:

            if var.double_size > 0:
                lower.append(var.get_double_lower_bounds())
                upper.append(var.get_double_upper_bounds())

            if var.integer_size > 0:
                lower.append(var.get_integer_lower_bounds())
                upper.append(var.get_integer_upper_bounds())

        lower = np.concatenate(lower)
        upper = np.concatenate(upper)

        if len(lower) != len(upper):
            raise ValueError(
                "Upper and lower bounds are different lengths. Malformed bound data from Parameter:"
                ' "{}"'.format(var.name)
            )

        return lower, upper

    def get_nobj(self):
        return len(self.model_objectives)

    def get_nec(self):
        return len([c for c in self.model_constraints if c.is_equality_constraint])

    def get_nic(self):
        count = 0
        for c in self.model_constraints:
            if c.is_double_bounded_constraint:
                count += 2
            elif c.is_lower_bounded_constraint or c.is_upper_bounded_constraint:
                count += 1
        return count

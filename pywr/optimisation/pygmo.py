import numpy as np
from . import BaseOptimisationWrapper

import logging
logger = logging.getLogger(__name__)


class PygmoWrapper(BaseOptimisationWrapper):

    def fitness(self, solution):
        logger.info('Evaluating solution ...')

        for ivar, var in enumerate(self.model_variables):
            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar+1])
            var.set_double_variables(np.array(solution[j]).copy())

        self.model.reset()
        run_stats = self.model.run()

        objectives = []
        for r in self.model_objectives:
            sign = 1.0 if r.is_objective == 'minimise' else -1.0
            value = r.aggregated_value()
            objectives.append(sign*value)

        constraints = [r.aggregated_value() for r in self.model_constraints]

        # Return values to the solution
        logger.info('Evaluation complete!')
        return objectives + constraints

    def get_bounds(self):
        """ Return the variable bounds. """
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
                'Upper and lower bounds are different lengths. Malformed bound data from Parameter:'
                ' "{}"'.format(var.name))

        return lower, upper

    def get_nobj(self):
        return len(self.model_objectives)

    def get_nec(self):
        return len(self.model_constraints)
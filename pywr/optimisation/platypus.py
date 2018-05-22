import numpy as np
import platypus
from . import cache_constraints, cache_objectives, cache_variable_parameters, BaseOptimisationWrapper

import logging
logger = logging.getLogger(__name__)


class PlatypusWrapper(BaseOptimisationWrapper):
    """ A helper class for running pywr optimisations with platypus.
    """
    def __init__(self, *args, **kwargs):
        super(PlatypusWrapper, self).__init__(*args, **kwargs)

        # To determine the number of variables, etc
        m = self.model

        # Cache the variables, objectives and constraints
        variables, variable_map = cache_variable_parameters(m)
        objectives = cache_objectives(m)
        constraints = cache_constraints(m)

        self.problem = platypus.Problem(variable_map[-1], len(objectives), len(constraints))
        self.problem.function = self.evaluate

        # Setup the problem; subclasses can change this behaviour
        self._make_variables(variables)
        self._make_constraints(constraints)

    def _make_variables(self, variables):
        """Setup the variable types. """

        ix = 0
        for var in variables:

            lower = []
            upper = []
            if var.double_size > 0:
                lower.append(var.get_double_lower_bounds())
                upper.append(var.get_double_upper_bounds())

            if var.integer_size > 0:
                lower.append(var.get_integer_lower_bounds())
                upper.append(var.get_integer_upper_bounds())

            lower = np.concatenate(lower)
            upper = np.concatenate(upper)

            if len(lower) != len(upper):
                raise ValueError('Upper and lower bounds are different lengths. Malformed bound data from Parameter:'
                                 ' "{}"'.format(var.name))

            for i in range(len(lower)):
                self.problem.types[ix] = platypus.Real(lower[i], upper[i])
                ix += 1

    def _make_constraints(self, constraints):
        """ Setup the constraints. """
        # Setup the constraints
        self.problem.constraints[:] = "<=0"

    def evaluate(self, solution):
        logger.info('Evaluating solution ...')

        for ivar, var in enumerate(self.model_variables):
            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar+1])
            var.update(np.array(solution[j]))

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
        if len(constraints) > 0:
            return objectives, constraints
        else:
            return objectives

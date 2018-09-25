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

        if len(variables) < 1:
            raise ValueError('At least one variable must be defined.')

        if len(objectives) < 1:
            raise ValueError('At least one objective must be defined.')

        self.problem = platypus.Problem(variable_map[-1], len(objectives), len(constraints))
        self.problem.function = self.evaluate
        self.problem.wrapper = self

        # Setup the problem; subclasses can change this behaviour
        self._make_variables(variables)
        self._make_constraints(constraints)

    def _make_variables(self, variables):
        """Setup the variable types. """

        ix = 0
        for var in variables:
            if var.double_size > 0:
                lower = var.get_double_lower_bounds()
                upper = var.get_double_upper_bounds()
                for i in range(var.double_size):
                    self.problem.types[ix] = platypus.Real(lower[i], upper[i])
                    ix += 1

            if var.integer_size > 0:
                lower = var.get_integer_lower_bounds()
                upper = var.get_integer_upper_bounds()
                for i in range(var.integer_size):
                    # Integers are cast to real
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
            x = np.array(solution[j])
            assert len(x) == var.double_size + var.integer_size
            if var.double_size > 0:
                var.set_double_variables(np.array(x[:var.double_size]))

            if var.integer_size > 0:
                ints = np.round(np.array(x[-var.integer_size:])).astype(np.int32)
                var.set_integer_variables(ints)

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

import numpy as np
import inspyred
from ..core import Model


class InspyredOptimisationModel(Model):
    """ A pywr.core.Model subclass to enable optimisation using inspyred.

    This classes overloads Model.setup() to create cached variable and objective maps to use with inspyred.

    A generator, bounder and evaluator method are provided to use with the inspyred algorithms.
    """

    def _cache_variable_parameters(self):
        variables = []
        variable_map = [0, ]
        for var in self.variables:
            size = var.double_size + var.integer_size
            variable_map.append(variable_map[-1]+size)
            variables.append(var)

        self._variables = variables
        self._variable_map = variable_map

    def _cache_objectives(self):
        # This is done to make sure the order is fixed during optimisation.
        objectives = []
        for r in self.objectives:
            objectives.append(r)

        self._objectives = objectives

    def setup(self):
        super(InspyredOptimisationModel, self).setup()
        self._cache_variable_parameters()
        self._cache_objectives()

    def generator(self, random, args):

        values = []
        for var in self._variables:
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

            for i in range(var.size):
                values.append(random.uniform(lower[i], upper[i]))

        return values

    def evaluator(self, candidates, args):
        fitness = []
        for i, candidate in enumerate(candidates):
            for ivar, var in enumerate(self._variables):
                l, u = self._variable_map[ivar], self._variable_map[ivar+1]
                if var.double_size > 0:
                    dj = slice(l, l + var.double_size)
                    var.set_double_variables(np.array(candidate[dj]))

                if var.integer_size > 0:
                    ij = slice(l + var.double_size, u)
                    # Convert integer variables to ints
                    ints = np.rint(np.array(candidate[ij])).astype(np.int)
                    var.set_integer_variables(ints)

            self.reset()
            self.run()

            fitness.append(inspyred.ec.emo.Pareto([r.aggregated_value() for r in self._objectives]))
        return fitness

    def bounder(self, candidate, args):
        for ivar, var in enumerate(self._variables):
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

            j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
            candidate[j] = np.minimum(upper, np.maximum(lower, candidate[j]))
        return candidate

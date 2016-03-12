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
        for node in self.nodes:
            for var in node.variables:
                variable_map.append(variable_map[-1]+var.size)
                variables.append(var)

        self._variables = variables
        self._variable_map = variable_map

    def _cache_objectives(self):
        objectives = []
        for r in self.recorders:
            if r.is_objective:
                objectives.append(r)

        self._objectives = objectives

    def setup(self):
        super(InspyredOptimisationModel, self).setup()
        self._cache_variable_parameters()
        self._cache_objectives()

    def generator(self, random, args):
        size = self._variable_map[-1]

        return [random.uniform(0.0, 1.0) for i in range(size)]

    def evaluator(self, candidates, args):

        fitness = []
        for i, candidate in enumerate(candidates):
            print('Running candidiate: {:03d}'.format(i))
            for ivar, var in enumerate(self._variables):
                j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
                var.update(np.array(candidate[j]))

            self.reset()
            self.run()

            fitness.append(inspyred.ec.emo.Pareto([r.value() for r in self._objectives]))
        return fitness

    def bounder(self, candidate, args):
        for ivar, var in enumerate(self._variables):
            lower = var.lower_bounds()
            upper = var.upper_bounds()
            j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
            candidate[j] = np.minimum(upper, np.maximum(lower, candidate[j]))
        return candidate
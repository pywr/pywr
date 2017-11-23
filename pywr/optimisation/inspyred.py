import numpy as np
import inspyred
from . import BaseOptimisationModel


class InspyredOptimisationModel(BaseOptimisationModel):
    """ A pywr.core.Model subclass to enable optimisation using inspyred.

    A generator, bounder and evaluator method are provided to use with the inspyred algorithms.
    """
    def generator(self, random, args):

        values = []
        for var in self._variables:
            l, u = var.lower_bounds(), var.upper_bounds()
            for i in range(var.size):
                values.append(random.uniform(l[i], u[i]))
        return values

    def evaluator(self, candidates, args):
        fitness = []
        for i, candidate in enumerate(candidates):
            for ivar, var in enumerate(self._variables):
                j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
                var.update(np.array(candidate[j]))

            self.reset()
            self.run()

            fitness.append(inspyred.ec.emo.Pareto([r.aggregated_value() for r in self._objectives]))
        return fitness

    def bounder(self, candidate, args):
        for ivar, var in enumerate(self._variables):
            lower = var.lower_bounds()
            upper = var.upper_bounds()
            j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
            candidate[j] = np.minimum(upper, np.maximum(lower, candidate[j]))
        return candidate
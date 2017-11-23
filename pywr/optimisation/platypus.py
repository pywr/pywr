import numpy as np
import platypus
from . import BaseOptimisationModel


class PlatypusOptimisationModel(BaseOptimisationModel):
    """ A pywr.core.Model subclass to enable optimisation using platypus.

    Setup now returns a `platypus.Problem` instance that can be used with
    platypus algorithms.
    """
    def __init__(self, *args, **kwargs):
        super(BaseOptimisationModel, self).__init__(*args, **kwargs)

        self._problem = None

    def setup(self):
        """ Returns a `platypus.Problem` object setup to perform the opimisation of this model """
        super(PlatypusOptimisationModel, self).setup()
        # Initialise the Problem with the correct size
        p = platypus.Problem(self._variable_map[-1], len(self._objectives), len(self._constraints))

        # Setup the variable types
        ix = 0
        for var in self._variables:
            l, u = var.lower_bounds(), var.upper_bounds()
            for i in range(var.size):
                # All assumed to be real valued
                p.types[ix] = platypus.Real(l[i], u[i])
                ix += 1

        # Setup the constraints
        p.constraints[:] = "<=0"

        p.function = self.evaluator
        return p

    def evaluator(self, solution):

        for ivar, var in enumerate(self._variables):
            j = slice(self._variable_map[ivar], self._variable_map[ivar+1])
            var.update(np.array(solution[j]))

        self.reset()
        self.run()

        objectives = [r.aggregated_value() for r in self._objectives]
        constraints = [r.aggregated_value() for r in self._constraints]
        if len(constraints) > 0:
            return objectives, constraints
        else:
            return objectives

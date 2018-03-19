

import numpy as np
import inspyred
from ..core import Model


class BaseOptimisationModel(Model):
    """ A base class pywr.core.Model subclass to enable optimisation.

    This classes overloads Model.setup() to create cached variable, constraints
     and objectives..

    """
    def __init__(self, *args, **kwargs):
        super(BaseOptimisationModel, self).__init__(*args, **kwargs)

        # Setup optimisation cache
        self._variable_map = None
        self._variables = None
        self._constraints = None
        self._objectives = None

    def _cache_variable_parameters(self):
        variables = []
        variable_map = [0, ]
        for var in self.variables:
            variable_map.append(variable_map[-1]+var.size)
            variables.append(var)

        self._variables = variables
        self._variable_map = variable_map

    def _cache_constraints(self):
        constraints = []
        for r in self.constraints:
            constraints.append(r)

        self._constraints = constraints

    def _cache_objectives(self):
        # This is done to make sure the order is fixed during optimisation.
        objectives = []
        for r in self.objectives:
            objectives.append(r)

        self._objectives = objectives

    def setup(self):
        super(BaseOptimisationModel, self).setup()
        self._cache_variable_parameters()
        self._cache_constraints()
        self._cache_objectives()

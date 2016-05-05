# -*- coding: utf-8 -*-
from ._parameters import Parameter
from ..core import Storage
import numpy as np


class KeatingStreamFlowParameter(Parameter):
    """
    A flow Parameter that returns stream flow from an Aquifer based on the groundwater level.

    The approach is based on the lumped parameter model by Keating (1982). This Parameter calculates the
     perennial and winterbourne stream flow components based on the Storage node. It requires the Storage
     node to have a valid level Parameter.

    In contrast to the Keating paper a general coefficient is provided for calibration. The Keating approach
     utilised a coefficient of $B/L$ where "$B$ and $L$ are the dimensions of the aquifer block parallel
     and perpendicular to the stream."

    Keating, T. (1982), A Lumped Parameter Model of a Chalk Aquifer-Stream System in Hampshire,
      United Kingdom. Ground Water, 20: 430–436. doi:10.1111/j.1745-6584.1982.tb02763.x
    """
    def __init__(self, storage_node, levels, transmissivity, coefficient=1.0):
        """

        :param storage_node:
        :param levels:
        :param transmissivity:
        :param coefficient:
        """
        self.storage_node = storage_node

        if len(levels) != len(transmissivity):
            raise ValueError('The number of transmissivity values must equal the number of levels.')

        self.levels = np.array(levels)
        self.transmissivity = np.array(transmissivity)
        self.coefficient = coefficient

    def value(self, ts, scenario_index):
        level = self.storage_node.get_level(ts, scenario_index)

        # Get current values of transmissivity and storage based on aquifer level
        T = self.transmissivity
        # Coefficient
        C = self.coefficient

        # Winterbourne flow component
        Q = 0.0

        # Iterate all but one of the levels
        levels = self.levels
        for i in range(len(levels)-1):
            if level >= levels[i]:
                # interpolate transmissivity to the next level
                # This is used in a simple integration so we take half the value to represent the triangular
                # part between the two levels.
                t = T[i] + (level - levels[i])*(T[i+1] - T[i])/(levels[i+1] - levels[i]) / 2
                Q += 2*t*C*(level - levels[i])

        return Q









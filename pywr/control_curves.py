"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._parameters import Parameter as BaseParameter
import numpy as np


class BaseParameterControlCurve(BaseParameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    pass


class ParameterControlCurvePiecewise(BaseParameterControlCurve):
    """ A control curve Parameter that returns one of two values depending on whether the current
    volume is above or below the control curve.

    """
    def __init__(self, above_curve_value, below_curve_value):
        super(ParameterControlCurvePiecewise, self).__init__()

        self.above_curve_value = above_curve_value
        self.below_curve_value = below_curve_value

    def value(self, ts, scenario_indices=[0]):
        i = self.node.model.scenarios.ravel_indices(scenario_indices)
        control_curve = self.node.control_curve.value(ts, scenario_indices)
        # If level above control curve then return above_curve_cost
        if self.node.current_pc[i] >= control_curve:
            return self.above_curve_value
        return self.below_curve_value

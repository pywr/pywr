"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._control_curves import BaseControlCurveParameter, ControlCurveInterpolatedParameter


class ControlCurvePiecewiseParameter(BaseControlCurveParameter):
    """ A control curve Parameter that returns one of two values depending on whether the current
    volume is above or below the control curve.

    """
    def __init__(self, control_curve, above_curve_value, below_curve_value):
        super(ControlCurvePiecewiseParameter, self).__init__(control_curve)

        self.above_curve_value = above_curve_value
        self.below_curve_value = below_curve_value

    def value(self, ts, scenario_indices=[0]):
        i = self.node.model.scenarios.ravel_indices(scenario_indices)
        control_curve = self.control_curve.value(ts, scenario_indices)
        # If level above control curve then return above_curve_cost
        #print(control_curve, self.node.current_pc[i])
        if self.node.current_pc[i] >= control_curve:
            return self.above_curve_value
        return self.below_curve_value



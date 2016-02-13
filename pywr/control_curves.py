"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._parameters import Parameter as BaseParameter
import numpy as np


class BaseParameterControlCurve(BaseParameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, control_curve):
        """

        Parameters
        ----------
        control_curve : Parameter
            The Parameter object to use as a control_curve. It should not be shared with other
            Nodes and Parameters because this object becomes control_curve.parent
        """
        control_curve.parent = self
        self.control_curve = control_curve

    def setup(self, model):
        self.control_curve.setup(model)
        super(BaseParameterControlCurve, self).setup(model)

    def reset(self):
        self.control_curve.reset()
        super(BaseParameterControlCurve, self).reset()

    def before(self, ts):
        self.control_curve.before(ts)
        super(BaseParameterControlCurve, self).before(ts)

    def after(self, ts):
        self.control_curve.after(ts)
        super(BaseParameterControlCurve, self).after(ts)


class ParameterControlCurvePiecewise(BaseParameterControlCurve):
    """ A control curve Parameter that returns one of two values depending on whether the current
    volume is above or below the control curve.

    """
    def __init__(self, control_curve, above_curve_value, below_curve_value):
        super(ParameterControlCurvePiecewise, self).__init__(control_curve)

        self.above_curve_value = above_curve_value
        self.below_curve_value = below_curve_value

    def value(self, ts, scenario_indices=[0]):
        i = self.node.model.scenarios.ravel_indices(scenario_indices)
        control_curve = self.control_curve.value(ts, scenario_indices)
        # If level above control curve then return above_curve_cost
        if self.node.current_pc[i] >= control_curve:
            return self.above_curve_value
        return self.below_curve_value


class ParameterControlCurveInterpolated(BaseParameterControlCurve):
    """ A control curve Parameter that interpolates between three values.
    """
    def __init__(self, control_curve, values):
        super(ParameterControlCurveInterpolated, self).__init__(control_curve)
        values = np.array(values)
        if len(values) != 3:
            raise ValueError("Three values must be given to define the interpolation knots.")
        self.values = values

    def value(self, ts, scenario_indices=[0]):
        i = self.node.model.scenarios.ravel_indices(scenario_indices)
        control_curve = self.control_curve.value(ts, scenario_indices)
        # return the interpolated value for the current level.
        return np.interp(self.node.current_pc[i], [0.0, control_curve, 1.0], self.values)

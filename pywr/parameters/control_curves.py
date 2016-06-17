"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._control_curves import BaseControlCurveParameter, ControlCurveInterpolatedParameter
from .parameters import parameter_registry
import numpy as np


class ControlCurvePiecewiseParameter(BaseControlCurveParameter):
    """ A control curve Parameter that returns one of two values depending on whether the current
    volume is above or below the control curve.

    """
    def __init__(self, control_curve, above_curve_value, below_curve_value, storage_node=None,
                 above_curve_upper_bounds=None, above_curve_lower_bounds=None,
                 below_curve_upper_bounds=None, below_curve_lower_bounds=None):
        """

        :param control_curve: Parameter to use as the control curve
        :param above_curve_value: Value to return when storage >= control_curve
        :param below_curve_value: Value to return when storage < control_curve
        :param storage_node: Optional Storage node to compare with the control_curve. If None
            self.node is used (i.e. the Node this object is attached to).
        """
        super(ControlCurvePiecewiseParameter, self).__init__(control_curve, storage_node=storage_node)

        self.above_curve_value = above_curve_value
        self.below_curve_value = below_curve_value

        # Bounds for use as a variable (i.e. when self.is_variable = True)
        self.above_curve_upper_bounds = above_curve_upper_bounds
        self.above_curve_lower_bounds = above_curve_lower_bounds
        self.below_curve_upper_bounds = below_curve_upper_bounds
        self.below_curve_lower_bounds = below_curve_lower_bounds
        self._update_variable_properties()

    def _update_variable_properties(self):
        size = 0
        lower_bounds = []
        upper_bounds = []
        if self.above_curve_upper_bounds is not None and self.above_curve_lower_bounds is not None:
            size += 1
            lower_bounds.append(self.above_curve_lower_bounds)
            upper_bounds.append(self.above_curve_upper_bounds)

        if self.below_curve_upper_bounds is not None and self.below_curve_lower_bounds is not None:
            size += 1
            lower_bounds.append(self.below_curve_lower_bounds)
            upper_bounds.append(self.below_curve_upper_bounds)

        self.size = size
        self._lower_bounds = np.array(lower_bounds)
        self._upper_bounds = np.array(upper_bounds)

    def value(self, ts, scenario_index):
        i = scenario_index.global_id
        control_curve = self.control_curve.value(ts, scenario_index)
        node = self.node if self.storage_node is None else self.storage_node
        # If level above control curve then return above_curve_cost
        if node.current_pc[i] >= control_curve:
            return self.above_curve_value
        return self.below_curve_value

    def update(self, values):
        i = 0
        if self.above_curve_upper_bounds is not None and self.above_curve_lower_bounds is not None:
            self.above_curve_value = values[i]
            i += 1

        if self.below_curve_upper_bounds is not None and self.below_curve_lower_bounds is not None:
            self.below_curve_value = values[i]
            i += 1

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds

parameter_registry.add(ControlCurvePiecewiseParameter)

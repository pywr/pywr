"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._control_curves import BaseControlCurveParameter, ControlCurveInterpolatedParameter, ControlCurveIndexParameter, \
    ControlCurveParameter
from .parameters import parameter_registry, load_parameter_values, load_parameter, Parameter, parameter_property
import numpy as np


class PiecewiseLinearControlCurve(Parameter):
    """Piecewise function composed of two linear curves
    
    Parameters
    ----------
    model : Model
    storage_node : Storage
    control_curve : Parameter
    values : [(float, float), (float, float)]
        Iterable of 2-tuples, representing the lower and upper value of the
        linear interpolation below and above the control curve, respectively.
    minimum : float
        The storage considered the bottom of the lower curve, 0-1 (default=0).
    maximum : float
        The storage considered the top of the upper curve, 0-1 (default=1).
    """
    def __init__(self, model, storage_node, control_curve, values, minimum=0.0, maximum=1.0, *args, **kwargs):
        super(PiecewiseLinearControlCurve, self).__init__(model, *args, **kwargs)
        self._control_curve = None
        self.storage_node = storage_node
        self.control_curve = control_curve
        self.below_lower, self.below_upper = values[0]
        self.above_lower, self.above_upper = values[1]
        self.minimum = minimum
        self.maximum = maximum

    control_curve = parameter_property("_control_curve")

    def value(self, timestamp, scenario_index):
        control_curve = self._control_curve.get_value(scenario_index)
        current_pc = self.storage_node.current_pc[scenario_index.global_id]
        if current_pc > control_curve:
            value = self._interpolate(current_pc, control_curve, self.maximum, self.above_lower, self.above_upper)
        else:
            value = self._interpolate(current_pc, self.minimum, control_curve, self.below_lower, self.below_upper)
        return value

    @staticmethod
    def _interpolate(current_position, lower_bound, upper_bound, lower_value, upper_value):
        if current_position < lower_bound:
            value = lower_value
        elif current_position > upper_bound:
            value = upper_value
        else:
            try:
                factor = (current_position - lower_bound) / (upper_bound - lower_bound)
            except ZeroDivisionError:
                factor = 1.0
            value = lower_value + (upper_value - lower_value) * factor
        return value

    def reset(self):
        super(PiecewiseLinearControlCurve, self).setup()
        assert self.maximum > self.minimum
        assert np.isfinite(self.below_lower)
        assert np.isfinite(self.below_upper)
        assert np.isfinite(self.above_lower)
        assert np.isfinite(self.above_upper)

    @classmethod
    def load(cls, model, data):
        storage_node = model._get_node_from_ref(model, data["storage_node"])
        control_curve = load_parameter(model, data["control_curve"])
        values = data["values"]
        kwargs = {}
        if "minimum" in data.keys():
            kwargs["minimum"] = data["minimum"]
        if "maximum" in data.keys():
            kwargs["maximum"] = data["maximum"]
        parameter = cls(model, storage_node, control_curve, values, **kwargs)
        return parameter

PiecewiseLinearControlCurve.register()





class AbstractProfileControlCurveParameter(BaseControlCurveParameter):
    _profile_size = None

    def __init__(self, model, storage_node, control_curves, values, profile=None, scale=1.0, **kwargs):
        super(AbstractProfileControlCurveParameter, self).__init__(model, storage_node, control_curves, **kwargs)

        nvalues = len(self.control_curves) + 1

        values = np.array(values)
        if values.ndim != 2:
            raise ValueError('Values must be two dimensional.')
        if values.shape[0] != nvalues:
            raise ValueError('First dimension of values should be one more than the number of '
                             'control curves ({}).'.format(nvalues))
        if values.shape[1] != self._profile_size:
            raise ValueError("Second dimension values must be size {}.".format(self._profile_size))
        self.values = values

        if isinstance(profile,  Parameter):
            self.profile = profile
            profile.parents.add(self)
        elif profile is not None:
            profile = np.array(profile)
            if profile.shape[0] != self._profile_size:
                raise ValueError("Length of profile must be size {}.".format(self._profile_size))
            self.profile = profile
        else:
            self.profile = np.ones(self._profile_size)
        self.scale = scale

    @classmethod
    def load(cls, model, data):
        control_curves = super(AbstractProfileControlCurveParameter, cls)._load_control_curves(model, data)
        storage_node = super(AbstractProfileControlCurveParameter, cls)._load_storage_node(model, data)
        values = load_parameter_values(model, data)
        # Now try loading a profile
        if 'profile' in data:
            # Profile is present, and this is the data
            pdata = data['profile']
            if 'type' in pdata:
                # If it contains a 'type', assume a `Parameter` object and attempt load
                profile = load_parameter(model, pdata)
            else:
                # Otherwise try to coerce to a numpy array
                profile = np.array(pdata)
        else:
            profile = None

        # Now load a scale if one is present. This should be a simple float
        if 'scale' in data:
            scale = float(data['scale'])
        else:
            scale = 1.0

        return cls(model, storage_node, control_curves, values=values, profile=profile, scale=scale)

    def _profile_index(self, ts, scenario_index):
        raise NotImplementedError()

    def value(self, ts, scenario_index):
        i = scenario_index.global_id
        node = self.node if self.storage_node is None else self.storage_node
        iprofile = self._profile_index(ts, scenario_index)

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.get_value(scenario_index)
            # If level above control curve then return this level's value
            if node.current_pc[i] >= cc:
                val = self.values[j, iprofile]
                break
        else:
            val = self.values[-1, iprofile]

        # Now scale the control curve value by the scale and profile
        scale = self.scale
        if isinstance(self.profile, Parameter):
            scale *= self.profile.get_value(scenario_index)
        else:
            scale *= self.profile[iprofile]
        return val * scale


class MonthlyProfileControlCurveParameter(AbstractProfileControlCurveParameter):
    """ A control curve Parameter that returns values from a set of monthly profiles.

    Parameters
    ----------
    storage_node : `Storage`
        An optional `Storage` node that can be used to query the current percentage volume.
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like
        A two dimensional array_like where the first dimension corresponds to the current level
        of the corresponding `Storage` and the second dimension is of size 12, corresponding to
        the monthly value to return based on current time-step.
    profile : 'Parameter` or array_like of length 12, optional
        An optional profile `Parameter` or monthly array to factor the values by. The default is
        np.ones(12) to have no scaling effect on the returned values.
    scale : float, optional
        An optional constant to factor the values by. The default is 1.0 to have scaling effect on
        the returned values.


    See also
    --------
    `BaseControlCurveParameter`
    `DailyProfileControlCurveParameter`
    """
    _profile_size = 12

    def _profile_index(self, ts, scenario_index):
        return ts.datetime.month - 1

MonthlyProfileControlCurveParameter.register()


class DailyProfileControlCurveParameter(AbstractProfileControlCurveParameter):
    """ A control curve Parameter that returns values from a set of daily profiles.

    Parameters
    ----------
    storage_node : `Storage`
        An optional `Storage` node that can be used to query the current percentage volume.
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like
        A two dimensional array_like where the first dimension corresponds to the current level
        of the corresponding `Storage` and the second dimension is of size 366, corresponding to
        the monthly value to return based on current time-step.
    profile : 'Parameter` or array_like of length 12, optional
        An optional profile `Parameter` or monthly array to factor the values by. The default is
        np.ones(366) to have no scaling effect on the returned values.
    scale : float, optional
        An optional constant to factor the values by. The default is 1.0 to have scaling effect on
        the returned values.

    See also
    --------
    `BaseControlCurveParameter`
    `MonthlyProfileControlCurveParameter`
    """
    _profile_size = 366

    def _profile_index(self, ts, scenario_index):
        return ts.datetime.dayofyear - 1

DailyProfileControlCurveParameter.register()

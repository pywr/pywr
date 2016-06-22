"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._control_curves import BaseControlCurveParameter, ControlCurveInterpolatedParameter
from .parameters import parameter_registry, load_parameter_values, load_parameter, BaseParameter
import numpy as np


class ControlCurveParameter(BaseControlCurveParameter):
    """ A generic multi-levelled control curve Parameter.

     This parameter can be used to return different values when a `Storage` node's current
      volumes is at different percentage of `max_volume` relative to predefined control curves.
      Control curves must be defined in the range [0, 1] corresponding to 0% and 100% volume.

     By default this parameter returns an integer sequence from zero if the first control curve
      is passed, and incrementing by one for each control curve (or "level") the `Storage` node
      is below.

    Parameters
    ----------
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like or `None`, optional
        The values to return if the `Storage` object is above the correspond control curve.
        I.e. the first value is returned if the current volume is above the first control curve,
        and second value if above the second control curve, and so on. The length of `values`
        must be one more than than the length of `control_curves`.
    parameters : iterable `Parameter` objects or `None`, optional
        If `values` is `None` then `parameters` can specify a `Parameter` object to use at level
        of the control curves. In the same way as `values` the first `Parameter` is used if
        `Storage` is above the first control curve, and second `Parameter` if above the
        second control curve, and so on.
    storage_node : `Storage` or `None`, optional
        An optional `Storage` node that can be used to query the current percentage volume. If
        not specified it is assumed that this object is attached to a `Storage` node and therefore
        `self.node` is used.

    Notes
    -----
    If `values` and `parameters` are both `None`, the default, then `values` defaults to
     a range of integers, starting at zero, one more than length of `control_curves`.

    See also
    --------
    `BaseControlCurveParameter`

    """
    def __init__(self, control_curves, values=None, parameters=None, storage_node=None,
                 upper_bounds=None, lower_bounds=None):
        super(ControlCurveParameter, self).__init__(control_curves, storage_node=storage_node)
        # Expected number of values is number of control curves plus one.
        self.size = nvalues = len(self.control_curves) + 1
        self.values = None
        self.parameters = None
        if values is not None:
            if len(values) != nvalues:
                raise ValueError('Length of values should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.values = np.array(values)
        elif parameters is not None:
            if len(parameters) != nvalues:
                raise ValueError('Length of parameters should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.parameters = list(parameters)
        else:
            # No values or parameters given, default to sequence of integers
            self.values = np.arange(nvalues)

        # Default values
        self._upper_bounds = None
        self._lower_bounds = None

        # Bounds for use as a variable (i.e. when self.is_variable = True)
        if upper_bounds is not None:
            if self.values is None:
                raise ValueError('Upper bounds can only be specified if `values` is not `None`.')
            if len(upper_bounds) != nvalues:
                raise ValueError('Length of upper_bounds should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self._upper_bounds = np.array(upper_bounds)

        if lower_bounds is not None:
            if self.values is None:
                raise ValueError('Lower bounds can only be specified if `values` is not `None`.')
            if len(lower_bounds) != nvalues:
                raise ValueError('Length of lower_bounds should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self._lower_bounds = np.array(lower_bounds)

    @classmethod
    def load(cls, model, data):
        control_curves = super(ControlCurveParameter, cls)._load_control_curves(model, data)
        storage_node = super(ControlCurveParameter, cls)._load_storage_node(model, data)

        parameters = None
        values = None
        if 'values' in data:
            values = load_parameter_values(model, data)
        elif 'parameters' in data:
            # Load parameters
            parameters_data = data['parameters']
            parameters = []
            for pdata in parameters_data:
                parameters.append(load_parameter(model, pdata))

        return cls(control_curves, values=values, parameters=parameters, storage_node=storage_node)

    def value(self, ts, scenario_index):
        i = scenario_index.global_id
        node = self.node if self.storage_node is None else self.storage_node

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.value(ts, scenario_index)
            # If level above control curve then return this level's value
            if node.current_pc[i] >= cc:
                if self.parameters is not None:
                    return self.parameters[j].value(ts, scenario_index)
                else:
                    return self.values[j]

        if self.parameters is not None:
            return self.parameters[-1].value(ts, scenario_index)
        else:
            return self.values[-1]

    def update(self, values):
        self.values = np.array(values)

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds

parameter_registry.add(ControlCurveParameter)


class AbstractProfileControlCurveParameter(BaseControlCurveParameter):
    _profile_size = None

    def __init__(self, control_curves, values, storage_node=None, profile=None, scale=1.0):
        super(AbstractProfileControlCurveParameter, self).__init__(control_curves, storage_node=storage_node)

        nvalues = len(self.control_curves) + 1

        values = np.array(values)
        if values.ndim != 2:
            raise ValueError('Values must be two dimensional.')
        if values.shape[0] != nvalues:
            raise ValueError('First dimension of values should be one more than the number of '
                             'control curves ({}).'.format(nvalues))
        if values.shape[1] != self._profile_size:
            raise ValueError("Second dimension values must be size 12.".format(self._profile_size))
        self.values = values

        if isinstance(profile,  BaseParameter):
            self.profile = profile
        elif profile is not None:
            profile = np.array(profile)
            if profile.shape[0] != self._profile_size:
                raise ValueError("Length of profile must be size 12.".format(self._profile_size))
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

        return cls(control_curves, values=values, storage_node=storage_node, profile=profile, scale=scale)

    def _profile_index(self, ts, scenario_index):
        raise NotImplementedError()

    def value(self, ts, scenario_index):
        i = scenario_index.global_id
        node = self.node if self.storage_node is None else self.storage_node
        iprofile = self._profile_index(ts, scenario_index)

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.value(ts, scenario_index)
            # If level above control curve then return this level's value
            if node.current_pc[i] >= cc:
                val = self.values[j, iprofile]
                break
        else:
            val = self.values[-1, iprofile]

        # Now scale the control curve value by the scale and profile
        scale = self.scale
        if isinstance(self.profile, BaseParameter):
            scale *= self.profile.value(ts, scenario_index)
        else:
            scale *= self.profile[iprofile]
        return val * scale


class MonthlyProfileControlCurveParameter(AbstractProfileControlCurveParameter):
    """ A control curve Parameter that returns values from a set of monthly profiles.

    Parameters
    ----------
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like
        A two dimensional array_like where the first dimension corresponds to the current level
        of the corresponding `Storage` and the second dimension is of size 12, corresponding to
        the monthly value to return based on current time-step.
    storage_node : `Storage` or `None`, optional
        An optional `Storage` node that can be used to query the current percentage volume. If
        not specified it is assumed that this object is attached to a `Storage` node and therefore
        `self.node` is used.
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

parameter_registry.add(MonthlyProfileControlCurveParameter)


class DailyProfileControlCurveParameter(AbstractProfileControlCurveParameter):
    """ A control curve Parameter that returns values from a set of daily profiles.

    Parameters
    ----------
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like
        A two dimensional array_like where the first dimension corresponds to the current level
        of the corresponding `Storage` and the second dimension is of size 366, corresponding to
        the monthly value to return based on current time-step.
    storage_node : `Storage` or `None`, optional
        An optional `Storage` node that can be used to query the current percentage volume. If
        not specified it is assumed that this object is attached to a `Storage` node and therefore
        `self.node` is used.
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

parameter_registry.add(DailyProfileControlCurveParameter)

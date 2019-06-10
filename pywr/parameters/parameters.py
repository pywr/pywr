import os
import datetime
from ..parameter_property import parameter_property
from ._parameters import (
    Parameter, parameter_registry, UnutilisedDataWarning, ConstantParameter,
    ConstantScenarioParameter, ConstantScenarioIndexParameter, AnnualHarmonicSeriesParameter,
    ArrayIndexedParameter, ConstantScenarioParameter, IndexedArrayParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter, TablesArrayParameter,
    DailyProfileParameter, MonthlyProfileParameter, WeeklyProfileParameter,
    ArrayIndexedScenarioParameter, ScenarioMonthlyProfileParameter,
    align_and_resample_dataframe, DataFrameParameter, FlowParameter,
    IndexParameter, AggregatedParameter, AggregatedIndexParameter,
    NegativeParameter, MaxParameter, NegativeMaxParameter, MinParameter, NegativeMinParameter,
    DeficitParameter, load_parameter, load_parameter_values, load_dataframe)
from . import licenses
from ._polynomial import Polynomial1DParameter, Polynomial2DStorageParameter
from ._thresholds import StorageThresholdParameter, RecorderThresholdParameter
from ._hydropower import HydropowerTargetParameter
from past.builtins import basestring
import numpy as np
from scipy.interpolate import interp1d
import pandas


class FunctionParameter(Parameter):
    def __init__(self, model, parent, func, *args, **kwargs):
        super(FunctionParameter, self).__init__(model, *args, **kwargs)
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)
FunctionParameter.register()


class ScaledProfileParameter(Parameter):
    def __init__(self, model, scale, profile, *args, **kwargs):
        super(ScaledProfileParameter, self).__init__(model, *args, **kwargs)
        self.scale = scale

        profile.parents.add(self)
        self.profile = profile

    @classmethod
    def load(cls, model, data):
        scale = float(data.pop("scale"))
        profile = load_parameter(model, data.pop("profile"))
        return cls(model, scale, profile, **data)

    def value(self, ts, si):
        p = self.profile.get_value(si)
        return self.scale * p
ScaledProfileParameter.register()


class AbstractInterpolatedParameter(Parameter):
    def __init__(self, model, x, y, interp_kwargs=None, **kwargs):
        super(AbstractInterpolatedParameter, self).__init__(model, **kwargs)
        self.x = x
        self.y = y
        self.interp = None
        default_interp_kwargs = dict(kind='linear', bounds_error=True)
        if interp_kwargs is not None:
            # Overwrite or add to defaults with given values
            default_interp_kwargs.update(interp_kwargs)
        self.interp_kwargs = default_interp_kwargs

    def _value_to_interpolate(self, ts, scenario_index):
        raise NotImplementedError()

    def setup(self):
        super(AbstractInterpolatedParameter, self).setup()
        self.interp = interp1d(self.x, self.y, **self.interp_kwargs)

    def value(self, ts, scenario_index):
        v = self._value_to_interpolate(ts, scenario_index)
        return self.interp(v)


class InterpolatedParameter(AbstractInterpolatedParameter):
    """
    Parameter value is equal to the interpolation of another parameter

    Example
    -------
    >>> x = [0, 5, 10, 20]
    >>> y = [0, 10, 30, -5]
    >>> p1 = ConstantParameter(model, 9.3) # or something more interesting
    >>> p2 = InterpolatedParameter(model, x, y, interp_kwargs={"kind": "linear"})
    """
    def __init__(self, model, parameter, x, y, interp_kwargs=None, **kwargs):
        super(InterpolatedParameter, self).__init__(model, x, y, interp_kwargs, **kwargs)
        self._parameter = None
        self.parameter = parameter

    parameter = parameter_property("_parameter")

    def _value_to_interpolate(self, ts, scenario_index):
        return self._parameter.get_value(scenario_index)


class InterpolatedVolumeParameter(AbstractInterpolatedParameter):
    """
    Generic interpolation parameter calculated from current volume
    """
    def __init__(self, model, node, x, y, interp_kwargs=None, **kwargs):
        super(InterpolatedVolumeParameter, self).__init__(model, x, y, interp_kwargs, **kwargs)
        self._node = node

    def _value_to_interpolate(self, ts, scenario_index):
        return self._node.volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        volumes = np.array(data.pop("volumes"))
        values = np.array(data.pop("values"))
        kind = data.pop("kind", "linear")
        return cls(model, node, volumes, values, interp_kwargs={'kind': kind})
InterpolatedVolumeParameter.register()


class AbstractLinearRoutingParameter(Parameter):
    def __init__(self, model, weighting, time_of_travel, **kwargs):
        super().__init__(model, **kwargs)

        self._weighting = None
        self.weighting = weighting

        self._time_of_travel = None
        self.time_of_travel = time_of_travel

    weighting = parameter_property("_weighting", wrap_constants=True)
    time_of_travel = parameter_property("_time_of_travel", wrap_constants=True)

    def c0(self):
        # This only works for constant parameters. It would require Parameter support
        # in the LP solvers to function with generic parameters
        assert isinstance(self.time_of_travel, ConstantParameter)
        K = self.time_of_travel.get_double_variables()[0]
        assert isinstance(self.weighting, ConstantParameter)
        X = self.weighting.get_double_variables()[0]

        dt = self.model.timestepper.delta.days
        c0 = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
        return c0

    def c1(self, scenario_index):
        dt = self.model.timestepper.delta.days
        X = self.weighting.get_value(scenario_index)
        K = self.time_of_travel.get_value(scenario_index)
        return (dt - 2 * X * K) / (2 * K * (1 - X) + dt)

    def c2(self, scenario_index):
        dt = self.model.timestepper.delta.days
        X = self.weighting.get_value(scenario_index)
        K = self.time_of_travel.get_value(scenario_index)
        return (dt + 2 * X * K) / (2 * K * (1 - X) + dt)


class LinearRoutingParameter(AbstractLinearRoutingParameter):
    def __init__(self, model, inflow_node, outflow_node, weighting, time_of_travel, **kwargs):
        super().__init__(model, weighting, time_of_travel, **kwargs)

        self.inflow_node = inflow_node
        self._inflow_parameter = None
        self.inflow_parameter = FlowParameter(model, inflow_node, initial_value=1000)

        self.outflow_node = outflow_node
        self._outflow_parameter = None
        self.outflow_parameter = FlowParameter(model, outflow_node, initial_value=1000)

    inflow_parameter = parameter_property("_inflow_parameter")
    outflow_parameter = parameter_property("_outflow_parameter")

    def value(self, ts, si):
        c1 = self.c1(si)
        c2 = self.c2(si)
        I = self.inflow_parameter.get_value(si)
        O = self.outflow_parameter.get_value(si)
        return c2*I + (1 - c1 - c2)*O


class RoutedIncrementalFlowParameter(AbstractLinearRoutingParameter):
    def __init__(self, model, upstream_flow_parameters, total_flow_parameter, weighting, time_of_travel, **kwargs):
        super().__init__(model, weighting, time_of_travel, **kwargs)

        self.upstream_flow_parameters = []
        for param in upstream_flow_parameters:
            self.children.add(param)
            self.upstream_flow_parameters.append(param)

        self._total_flow_parameter = None
        self.total_flow_parameter = total_flow_parameter
        self._prev_total_upstream_flow = None
        self._prev_routed_upstream_flow = None

    total_flow_parameter = parameter_property("_total_flow_parameter")

    def reset(self):
        super().reset()
        num_comb = len(self.model.scenarios.combinations)
        self._prev_total_upstream_flow = np.zeros(num_comb)
        self._prev_routed_upstream_flow = np.zeros(num_comb)

    def value(self, ts, si):

        total_upstream_flow = 0
        for param in self.upstream_flow_parameters:
            total_upstream_flow += param.get_value(si)

        total_local_flow = self.total_flow_parameter.get_value(si)

        c0 = self.c0()
        c1 = self.c1(si)
        c2 = self.c2(si)

        routed_upstream_flow = c2*self._prev_total_upstream_flow[si.global_id]
        routed_upstream_flow += (1 - c1 - c2)*self._prev_routed_upstream_flow[si.global_id]
        routed_upstream_flow += c0 * total_upstream_flow

        incremental_flow = total_local_flow - routed_upstream_flow

        # Save upstream & routed flows for next time-step
        self._prev_total_upstream_flow[si.global_id] = total_upstream_flow
        self._prev_routed_upstream_flow[si.global_id] = routed_upstream_flow

        # TODO do we assert > 0 or just use max?
        assert incremental_flow > 0
        incremental_flow = max(incremental_flow, 0)

        return incremental_flow


def pop_kwarg_parameter(kwargs, key, default):
    """Pop a parameter from the keyword arguments dictionary

    Parameters
    ----------
    kwargs : dict
        A keyword arguments dictionary
    key : string
        The argument name, e.g. 'flow'
    default : object
        The default value to use if the dictionary does not have that key

    Returns a Parameter
    """
    value = kwargs.pop(key, default)
    if isinstance(value, Parameter):
        return value
    elif callable(value):
        # TODO this is broken?
        return FunctionParameter(self, value)
    else:
        return value


class PropertiesDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(value, Property):
            value = ConstantParameter(value)
        dict.__setitem__(self, key, value)

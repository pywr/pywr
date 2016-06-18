import os
import datetime
from ._parameters import (
    Parameter as BaseParameter, parameter_registry,
    ConstantScenarioParameter,
    ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter,
    DailyProfileParameter,
    load_parameter, load_parameter_values)
import numpy as np
import pandas

class Parameter(BaseParameter):
    def value(self, ts, scenario_index):
        raise NotImplementedError()


# TODO shared dict with pywr.recorders
agg_funcs = {
    "mean": np.mean,
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "product": np.product,
}
class AggregatedParameter(Parameter):
    """A collection of Parameters

    This object behaves like a set. Licenses can be added to or removed from it.

    """
    def __init__(self, parameters=None, agg_func='mean'):
        super(AggregatedParameter, self).__init__()
        if parameters is None:
            self._parameters = set()
        else:
            self._parameters = set(parameters)
            for param in self._parameters:
                param.parent = self

        self.agg_func = agg_func
        if isinstance(self.agg_func, str):
            self.agg_func = agg_funcs[self.agg_func]

    @classmethod
    def load(cls, model, data):

        try:
            parameters_data = data['parameters']
        except KeyError:
            parameters_data = []

        parameters = []
        for pdata in parameters_data:
            parameters.append(load_parameter(model, pdata))

        agg_func = data.get('agg_func', 'mean')
        return cls(parameters=parameters, agg_func=agg_func)

    def add(self, parameter):
        self._parameters.add(parameter)
        parameter.parent = self

    def remove(self, parameter):
        self._parameters.remove(parameter)
        parameter.parent = None

    def __len__(self):
        return len(self._parameters)

    def value(self, ts, si):
        values = [p.value(ts, si) for p in self._parameters]
        return self.agg_func(values)

    def setup(self, model):
        for parameter in self._parameters:
            parameter.setup(model)

    def after(self, timestep):
        for parameter in self._parameters:
            parameter.after(timestep)

    def reset(self):
        for parameter in self._parameters:
            parameter.reset()
parameter_registry.add(AggregatedParameter)


class ConstantParameter(Parameter):
    def __init__(self, value=None, lower_bounds=0.0, upper_bounds=np.inf):
        super(ConstantParameter, self).__init__()
        self._value = value
        self.size = 1
        self._lower_bounds = np.ones(self.size) * lower_bounds
        self._upper_bounds = np.ones(self.size) * upper_bounds

    def value(self, ts, scenario_index):
        return self._value

    def update(self, values):
        self._value = values[0]

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds
parameter_registry.add(ConstantParameter)


class FunctionParameter(Parameter):
    def __init__(self, parent, func):
        super(FunctionParameter, self).__init__()
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)
parameter_registry.add(FunctionParameter)


class MonthlyProfileParameter(Parameter):
    def __init__(self, values, lower_bounds=0.0, upper_bounds=np.inf):
        super(MonthlyProfileParameter, self).__init__()
        self.size = 12
        if len(values) != self.size:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = np.array(values)
        self._lower_bounds = np.ones(self.size)*lower_bounds
        self._upper_bounds = np.ones(self.size)*upper_bounds

    def value(self, ts, scenario_index):
        return self._values[ts.datetime.month-1]

    def update(self, values):
        self._values[...] = values

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds
parameter_registry.add(MonthlyProfileParameter)


class ScaledProfileParameter(Parameter):
    def __init__(self, scale, profile):
        super(ScaledProfileParameter, self).__init__()
        self.scale = scale

        if profile.parent is not None and profile.parent is not self:
            raise RuntimeError('profile Parameter already has a different parent.')
            profile.parent = self
        self.profile = profile

    def value(self, ts, si):
        p = self.profile.value(ts, si)
        return self.scale * p








class AnnualHarmonicSeriesParameter(Parameter):
    def __init__(self, mean, amplitudes, phases, **kwargs):
        super(AnnualHarmonicSeriesParameter, self).__init__()
        if len(amplitudes) != len(phases):
            raise ValueError("The number  of amplitudes and phases must be the same.")
        n = len(amplitudes)
        self.size = 1 + 2*n
        self.mean = mean
        self.amplitudes = np.array(amplitudes)
        self.phases = np.array(phases)

        self._mean_lower_bounds = kwargs.pop('mean_lower_bounds', 0.0)
        self._mean_upper_bounds = kwargs.pop('mean_upper_bounds', np.inf)
        self._amplitude_lower_bounds = np.ones(n)*kwargs.pop('amplitude_lower_bounds', 0.0)
        self._amplitude_upper_bounds = np.ones(n)*kwargs.pop('amplitude_upper_bounds', np.inf)
        self._phase_lower_bounds = np.ones(n)*kwargs.pop('phase_lower_bounds', 0.0)
        self._phase_upper_bounds = np.ones(n)*kwargs.pop('phase_upper_bounds', np.pi*2)

    def value(self, ts, scenario_index):
        doy = ts.datetime.dayofyear
        n = len(self.amplitudes)
        return self.mean + sum(self.amplitudes[i]*np.cos(doy*(i+1)*np.pi*2/365 + self.phases[i]) for i in range(n))

    def update(self, values):
        n = len(self.amplitudes)
        self.mean = values[0]
        self.amplitudes[...] = values[1:n+1]
        self.phases[...] = values[n+1:]

    def lower_bounds(self):
        return np.r_[self._mean_lower_bounds, self._amplitude_lower_bounds, self._phase_lower_bounds]

    def upper_bounds(self):
        return np.r_[self._mean_upper_bounds, self._amplitude_upper_bounds, self._phase_upper_bounds]
parameter_registry.add(AnnualHarmonicSeriesParameter)


class Timeseries(Parameter):
    def __init__(self, name, df, metadata=None):
        super(Timeseries, self).__init__()
        self.name = name
        self.df = df
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def value(self, ts, scenario_index):
        return self.df[ts.datetime]

    @classmethod
    def read(self, model, **kwargs):
        name = kwargs['name']
        if name in model.data:
            raise ValueError('Timeseries with name "{}" already exists.'.format(name))

        filetype = None
        if 'type' in kwargs:
            filetype = kwargs['type']
        elif 'path' in kwargs:
            ext = kwargs['path'].split('.')[-1].lower()
            if ext == 'csv':
                filetype = 'csv'
            elif ext in ('xls', 'xlsx', 'xlsm'):
                filetype = 'excel'
            else:
                raise ValueError('Unrecognised timeseries type: {}'.format(ext))
        # TODO: other filetypes (SQLite? HDF5?)
        if filetype is None:
            raise ValueError('Unknown timeseries type.')
        path = kwargs['path']
        if not os.path.isabs(path) and model.path is not None:
            path = os.path.join(model.path, path)
        if filetype == 'csv':
            df = pandas.read_csv(
                path,
                index_col=0,
                parse_dates=True,
                dayfirst=True,
            )
        elif filetype == 'excel':
            sheet = kwargs['sheet']
            df = pandas.read_excel(
                path,
                sheet,
                index_col=0,
                dayfirst=True,
            )

        df = df[kwargs['column']]
        # create a new timeseries object
        ts = Timeseries(name, df, metadata=kwargs)
        # register the timeseries in the model
        model.data[name] = ts
        return ts
parameter_registry.add(Timeseries)


class InterpolatedLevelParameter(Parameter):
    """
    Level parameter calculated by interpolation from current volume
    """
    def __init__(self, volumes, levels, kind='linear'):
        from scipy.interpolate import interp1d
        # Create level interpolator
        self.interp = interp1d(volumes, levels, bounds_error=True, kind=kind)

    def value(self, ts, scenario_index):
        # Return interpolated value from current volume
        v = self.node.volume[scenario_index.global_id]
        level = self.interp(v)
        return level


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

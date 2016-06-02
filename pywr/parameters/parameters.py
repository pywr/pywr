import datetime
from xml.etree import ElementTree as ET
from ._parameters import (Parameter as BaseParameter, ConstantScenarioParameter, ArrayIndexedParameter, ArrayIndexedScenarioParameter,
                              ConstantScenarioParameter, ArrayIndexedScenarioMonthlyFactorsParameter, DailyProfileParameter)
import numpy as np
import pandas

class Parameter(BaseParameter):
    def value(self, ts, scenario_index):
        raise NotImplementedError()

    def xml(*args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, model, xml):
        # TODO: this doesn't look nice - need to rethink xml specification?
        parameter_types = {
            'const': ConstantParameter,
            'constant': ConstantParameter,
            'timestamp': ConstantParameter,
            'timedelta': ConstantParameter,
            'datetime': ConstantParameter,
            'timeseries': ConstantParameter,
            'python': FunctionParameter,
        }
        parameter_type = xml.get('type')
        return parameter_types[parameter_type].from_xml(model, xml)

class ParameterCollection(Parameter):
    """A collection of Parameters

    This object behaves like a set. Licenses can be added to or removed from it.

    """
    def __init__(self, parameters=None):
        super(ParameterCollection, self).__init__()
        if parameters is None:
            self._parameters = set()
        else:
            self._parameters = set(parameters)
            for param in self._parameters:
                param.parent = self

    def add(self, parameter):
        self._parameters.add(parameter)
        parameter.parent = self

    def remove(self, parameter):
        self._parameters.remove(parameter)
        parameter.parent = None

    def __len__(self):
        return len(self._parameters)

    def value(self, timestep, scenario_index):
        raise NotImplementedError()

    def setup(self, model):
        for parameter in self._parameters:
            parameter.setup(model)

    def after(self, timestep):
        for parameter in self._parameters:
            parameter.after(timestep)

    def reset(self):
        for parameter in self._parameters:
            parameter.reset()


class MinimumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_index):
        min_available = float('inf')
        for parameter in self._parameters:
            min_available = min(parameter.value(timestep, scenario_index), min_available)
        return min_available


class MaximumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_index):
        max_available = -float('inf')
        for parameter in self._parameters:
            max_available = max(parameter.value(timestep, scenario_index), max_available)
        return max_available


class ConstantParameter(Parameter):
    def __init__(self, value=None):
        super(ConstantParameter, self).__init__()
        self._value = value

    def value(self, ts, scenario_index):
        return self._value

    def xml(self, key):
        parameter_xml = ET.Element('parameter')
        parameter_xml.set('key', key)
        if isinstance(self._value, float):
            parameter_type = 'const'
            parameter_xml.text = str(self._value)
        elif isinstance(self._value, pandas.tslib.Timestamp):
            parameter_type = 'datetime'
            parameter_xml.text = str(self._value)
        elif isinstance(self._value, datetime.timedelta):
            parameter_type = 'timedelta'
            # try to represent the timedelta in sensible units
            total_seconds = self._value.total_seconds()
            if total_seconds % (60*60*24) == 0:
                units = 'days'
                parameter_xml.text = str(int(total_seconds / (60*60*24)))
            elif total_seconds % (60*60) == 0:
                units = 'hours'
                parameter_xml.text = str(int(total_seconds / (60*60)))
            elif total_seconds % (60) == 0:
                units = 'minutes'
                parameter_xml.text = str(int(total_seconds / 60))
            else:
                units = 'seconds'
                parameter_xml.text = str(int(total_seconds))
            parameter_xml.set('units', units)
        else:
            raise TypeError()
        parameter_xml.set('type', parameter_type)
        return parameter_xml

    @classmethod
    def from_xml(cls, model, xml):
        parameter_type = xml.get('type')
        key = xml.get('key')
        if parameter_type == 'const' or parameter_type == 'constant':
            try:
                value = float(xml.text)
            except:
                value = xml.text
            return key, ConstantParameter(value=value)
        elif parameter_type == 'timeseries':
            name = xml.text
            return key, model.data[name]
        elif parameter_type == 'datetime':
            return key, pandas.to_datetime(xml.text)
        elif parameter_type == 'timedelta':
            units = xml.get('units')
            value = float(xml.text)
            if units is None:
                units = 'seconds'
            units = units.lower()
            if units[-1] != 's':
                units = units + 's'
            td = datetime.timedelta(**{units: value})
            return key, td
        else:
            raise NotImplementedError('Unknown parameter type: {}'.format(parameter_type))


class FunctionParameter(Parameter):
    def __init__(self, parent, func):
        super(FunctionParameter, self).__init__()
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')


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

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')


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

    def xml(self, name):
        xml_ts = ET.Element('timeseries')
        xml_ts.set('name', self.name)
        for key, value in self.metadata.items():
            xml_meta = ET.SubElement(xml_ts, key)
            xml_meta.text = value
        return xml_ts

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
        if filetype == 'csv':
            path = model.path_rel_to_xml(kwargs['path'])
            df = pandas.read_csv(
                path,
                index_col=0,
                parse_dates=True,
                dayfirst=True,
            )
        elif filetype == 'excel':
            path = model.path_rel_to_xml(kwargs['path'])
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

    @classmethod
    def from_xml(self, model, xml):
        name = xml.get('name')
        properties = {}
        for child in xml.getchildren():
            properties[child.tag.lower()] = child.text
        properties['name'] = name

        if 'dayfirst' not in properties:
            # default to british dates
            properties['dayfirst'] = True

        ts = self.read(model, **properties)

        return ts


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

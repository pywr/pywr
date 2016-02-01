import datetime
from xml.etree import ElementTree as ET
from pywr._parameters import (Parameter as BaseParameter, ParameterConstantScenario, ParameterArrayIndexed,
                              ParameterConstantScenario, ParameterArrayIndexedScenarioMonthlyFactors)
import pandas


class Parameter(BaseParameter):
    def value(self, ts, scenario_indices=[0]):
        raise NotImplementedError()

    def xml(*args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, model, xml):
        # TODO: this doesn't look nice - need to rethink xml specification?
        parameter_types = {
            'const': ParameterConstant,
            'constant': ParameterConstant,
            'timestamp': ParameterConstant,
            'timedelta': ParameterConstant,
            'datetime': ParameterConstant,
            'timeseries': ParameterConstant,
            'python': ParameterFunction,
        }
        parameter_type = xml.get('type')
        return parameter_types[parameter_type].from_xml(model, xml)

class ParameterCollection(Parameter):
    """A collection of Parameters

    This object behaves like a set. Licenses can be added to or removed from it.

    """
    def __init__(self, parameters=None):
        if parameters is None:
            self._parameters = []
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

    def value(self, timestep, scenario_indices=[0]):
        raise NotImplementedError()


    def after(self, timestep):
        for parameter in self._parameters:
            parameter.after(timestep)

    def reset(self):
        for parameter in self._parameters:
            parameter.reset()


class MinimumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_indices=[0]):
        min_available = float('inf')
        for parameter in self._parameters:
            min_available = min(parameter.value(timestep, scenario_indices), min_available)
        return min_available


class MaximumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_indices=[0]):
        max_available = -float('inf')
        for parameter in self._parameters:
            max_available = max(parameter.value(timestep, scenario_indices), max_available)
        return max_available


class ParameterConstant(Parameter):
    def __init__(self, value=None):
        self._value = value

    def value(self, ts, scenario_indices=[0]):
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
            return key, ParameterConstant(value=value)
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


class ParameterFunction(Parameter):
    def __init__(self, parent, func):
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_indices=[0]):
        return self._func(self._parent, ts, scenario_indices)

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')


class ParameterMonthlyProfile(Parameter):
    def __init__(self, values):
        if len(values) != 12:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = values

    def value(self, ts, scenario_indices=[0]):
        return self._values[ts.datetime.month-1]

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')


class ParameterDailyProfile(Parameter):
    def __init__(self, values):
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = values

    def value(self, ts, scenario_indices=[0]):
        return self._values[ts.datetime.dayofyear-1]

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')


class Timeseries(Parameter):
    def __init__(self, name, df, metadata=None):
        self.name = name
        self.df = df
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def value(self, ts, scenario_indices=[0]):
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
                parse_dates=True,
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
        return ParameterFunction(self, value)
    else:
        return value


class PropertiesDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(value, Property):
            value = ParameterConstant(value)
        dict.__setitem__(self, key, value)
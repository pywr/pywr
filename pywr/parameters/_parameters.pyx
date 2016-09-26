import os
import numpy as np
cimport numpy as np
import pandas
from libc.math cimport cos, M_PI
from libc.limits cimport INT_MIN, INT_MAX
from past.builtins import basestring

cdef enum Predicates:
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4
_predicate_lookup = {"LT": Predicates.LT, "GT": Predicates.GT, "EQ": Predicates.EQ, "LE": Predicates.LE, "GE": Predicates.GE}

parameter_registry = set()

class PairedSet(set):
    def __init__(self, obj, *args, **kwargs):
        set.__init__(self)
        self.obj = obj

    def add(self, item):
        if not isinstance(item, Parameter):
            return
        set.add(self, item)
        if(self is self.obj.parents):
            set.add(item.children, self.obj)
        else:
            set.add(item.parents, self.obj)

    def remove(self, item):
        set.remove(self, item)
        if(self is self.obj.parents):
            set.remove(item.children, self.obj)
        else:
            set.remove(item.parents, self.obj)

    def clear(self):
        if(self is self.obj.parents):
            for parent in list(self):
                set.remove(parent.children, self.obj)
        else:
            for child in list(self):
                set.remove(child.parents, self.obj)
        set.clear(self)

cdef class Parameter:
    def __init__(self, name=None, comment=None):
        self.name = name
        self.comment = comment
        self.parents = PairedSet(self)
        self.children = PairedSet(self)
        self._recorders = []

    cpdef setup(self, model):
        cdef Parameter child
        for child in self.children:
            child.setup(model)

    cpdef reset(self):
        cdef Parameter child
        for child in self.children:
            child.reset()

    cpdef before(self, Timestep ts):
        cdef Parameter child
        for child in self.children:
            child.before(ts)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return 0

    cpdef after(self, Timestep ts):
        cdef Parameter child
        for child in self.children:
            child.after(ts)

    cpdef update(self, double[:] values):
        raise NotImplementedError()

    cpdef double[:] lower_bounds(self):
        raise NotImplementedError()

    cpdef double[:] upper_bounds(self):
        raise NotImplementedError()

    property size:
        def __get__(self):
            return self._size

        def __set__(self, value):
            self._size = value

    property is_variable:
        def __get__(self):
            return self._is_variable

        def __set__(self, value):
            self._is_variable = value

    property variables:
        def __get__(self):
            cdef Parameter var
            vars = []
            if self._is_variable:
                vars.append(self)
            for var in self.children:
                vars.extend(var.variables)
            return vars

    property recorders:
        """ Returns a list of `Recorder` objects attached to this node.

         See also
         --------
         `Recorder`
         """
        def __get__(self):
            return self._recorders

    @classmethod
    def load(cls, model, data):
        values = load_parameter_values(model, data)
        data.pop("values", None)
        data.pop("url", None)
        name = data.pop("name", None)
        comment = data.pop("comment", None)
        if data:
            key = list(data.keys())[0]
            raise TypeError("'{}' is an invalid keyword argument for this function".format(key))
        return cls(values, name=name, comment=None)

parameter_registry.add(Parameter)

cdef class ConstantParameter(Parameter):
    def __init__(self, value, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(ConstantParameter, self).__init__(**kwargs)
        self._value = value
        self.size = 1
        self._lower_bounds = np.ones(self.size) * lower_bounds
        self._upper_bounds = np.ones(self.size) * upper_bounds

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._value

    cpdef update(self, double[:] values):
        self._value = values[0]

    cpdef double[:] lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] upper_bounds(self):
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        # accept a dataframe
        if 'url' in data:
            value = load_dataframe(model, data)
        else:
            # accept both "value" and "values"
            value = data.pop("values", None)
            if value is None:
                value = data.pop("value")
        parameter = cls(value, **data)
        return parameter

parameter_registry.add(ConstantParameter)


cdef class CachedParameter(IndexParameter):
    """Wrapper for Parameters which caches the result"""
    def __init__(self, parameter, *args, **kwargs):
        super(IndexParameter, self).__init__(*args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.timestep = None
        self.scenario_index = None

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        if not timestep is self.timestep or not scenario_index is scenario_index:
            # refresh the cache
            self.cached_value = self.parameter.value(timestep, scenario_index)
            self.timestep = timestep
            self.scenario_index = scenario_index
        return self.cached_value

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        if not timestep is self.timestep or not scenario_index is scenario_index:
            # refresh the cache
            self.cached_index = self.parameter.index(timestep, scenario_index)
            self.timestep = timestep
            self.scenario_index = scenario_index
        return self.cached_index

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(parameter, **data)

parameter_registry.add(CachedParameter)


cdef class ArrayIndexedParameter(Parameter):
    """Time varying parameter using an array and Timestep._index

    The values in this parameter are constant across all scenarios.
    """
    def __init__(self, values, *args, **kwargs):
        super(ArrayIndexedParameter, self).__init__(*args, **kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts._index]
parameter_registry.add(ArrayIndexedParameter)


cdef class ArrayIndexedScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are vary in time based on index and vary within a single Scenario.
    """
    def __init__(self, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ArrayIndexedScenarioParameter, self).__init__(*args, **kwargs)
        cdef int i
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Values must be two dimensional.")
        if scenario._size != values.shape[1]:
            raise ValueError("The size of the second dimension of values must equal the size of the scenario.")
        self.values = values
        self._scenario = scenario

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self.values[ts._index, scenario_index._indices[self._scenario_index]]


cdef class ConstantScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioParameter, self).__init__(*args, **kwargs)
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
parameter_registry.add(ConstantScenarioParameter)


cdef class ArrayIndexedScenarioMonthlyFactorsParameter(Parameter):
    """Time varying parameter using an array and Timestep._index with
    multiplicative factors per Scenario
    """
    def __init__(self, Scenario scenario, values, factors, *args, **kwargs):
        """
        values is the baseline timeseries data that is perturbed by a factor. The
        factor is taken from factors which is shape (scenario.size, 12). Therefore
        factors vary with the individual scenarios in scenario and month.
        """
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).__init__(*args, **kwargs)

        values = np.asarray(values, dtype=np.float64)
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError("Factors must be two dimensional.")

        if scenario._size != factors.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if factors.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = values
        self._factors = factors

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.datetime.month-1
        return self._values[ts._index]*self._factors[scenario_index._indices[self._scenario_index], imth]
parameter_registry.add(ArrayIndexedScenarioMonthlyFactorsParameter)


cdef inline bint is_leap_year(int year):
    # http://stackoverflow.com/a/11595914/1300519
    return ((year & 3) == 0 and ((year % 25) != 0 or (year & 15) == 0))

cdef class DailyProfileParameter(Parameter):
    def __init__(self, values, *args, **kwargs):
        super(DailyProfileParameter, self).__init__(*args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.dayofyear - 1
        if not is_leap_year(<int>(ts._datetime.year)):
            if i > 58: # 28th Feb
                i += 1
        return self._values[i]
parameter_registry.add(DailyProfileParameter)

cdef class IndexParameter(Parameter):
    """Base parameter providing an `index` method

    See also
    --------
    IndexedArrayParameter
    ControlCurveIndexParameter
    """
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index as a float"""
        # return index as a float
        return float(self.index(timestep, scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index"""
        # return index as an integer
        return 0
parameter_registry.add(IndexParameter)

cdef class IndexedArrayParameter(Parameter):
    """Parameter which uses an IndexParameter to index an array of Parameters

    An example use of this parameter is to return a demand saving factor (as
    a float) based on the current demand saving level (calculated by an
    `IndexParameter`).

    Parameters
    ----------
    index_parameter : `IndexParameter`
    params : iterable of `Parameters` or floats


    Notes
    -----
    Float arguments `params` are converted to `ConstantParameter`
    """
    def __init__(self, index_parameter=None, params=None, **kwargs):
        super(IndexedArrayParameter, self).__init__(**kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.children.add(index_parameter)

        self.params = []
        for p in params:
            if not isinstance(p, Parameter):
                from pywr.parameters import ConstantParameter
                p = ConstantParameter(p)
            self.params.append(p)

        for param in params:
            self.children.add(param)
        self.children.add(index_parameter)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the Parameter at the current index"""
        cdef int index
        index = self.index_parameter.index(timestep, scenario_index)
        cdef Parameter parameter = self.params[index]
        return parameter.value(timestep, scenario_index)

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop("index_parameter"))
        params = [load_parameter(model, parameter_data) for parameter_data in data.pop("params")]
        return cls(index_parameter, params, **data)
parameter_registry.add(IndexedArrayParameter)


cdef class AnnualHarmonicSeriesParameter(Parameter):
    """ A `Parameter` which returns the value from an annual harmonic series

    This `Parameter` comprises a series N cosine function with a period of 365
     days. The calculation is performed using the Julien day of the year minus 1
     This causes a small discontinuity in non-leap years.

    .. math:: f(t) = A + \sum_{n=1}^N A_n\cdot \cos(\tfrac{2\pi nt}{365}+\phi_n)

    Parameters
    ----------

    mean : float
        Mean value for the series (i.e. the position of zeroth harmonic)
    amplitudes : array_like
        The amplitudes for the N harmonic cosine functions. Must be the same
        length as phases.
    phases : array_like
        The phase shift of the N harmonic cosine functions. Must be the same
        length as amplitudes.

    """
    def __init__(self, mean, amplitudes, phases, *args, **kwargs):
        if len(amplitudes) != len(phases):
            raise ValueError("The number  of amplitudes and phases must be the same.")
        n = len(amplitudes)
        self.size = 1 + 2*n
        self.mean = mean
        self._amplitudes = np.array(amplitudes)
        self._phases = np.array(phases)

        self._mean_lower_bounds = kwargs.pop('mean_lower_bounds', 0.0)
        self._mean_upper_bounds = kwargs.pop('mean_upper_bounds', np.inf)
        self._amplitude_lower_bounds = np.ones(n)*kwargs.pop('amplitude_lower_bounds', 0.0)
        self._amplitude_upper_bounds = np.ones(n)*kwargs.pop('amplitude_upper_bounds', np.inf)
        self._phase_lower_bounds = np.ones(n)*kwargs.pop('phase_lower_bounds', 0.0)
        self._phase_upper_bounds = np.ones(n)*kwargs.pop('phase_upper_bounds', np.pi*2)
        super(AnnualHarmonicSeriesParameter, self).__init__(*args, **kwargs)
        self._value_cache = 0.0
        self._ts_index_cache = -1

    @classmethod
    def load(cls, model, data):
        mean = data.pop('mean')
        amplitudes = data.pop('amplitudes')
        phases = data.pop('phases')

        return cls(mean, amplitudes, phases, **data)

    property amplitudes:
        def __get__(self):
            return np.array(self._amplitudes)

    property phases:
        def __get__(self):
            return np.array(self._phases)

    cpdef reset(self):
        Parameter.reset(self)
        self._value_cache = 0.0
        self._ts_index_cache = -1

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef int ts_index = timestep._index
        cdef int doy = timestep._datetime.dayofyear - 1
        cdef int n = self._amplitudes.shape[0]
        cdef int i
        cdef double val
        if ts_index == self._ts_index_cache:
            val = self._value_cache
        else:
            val = self.mean
            for i in range(n):
                val += self._amplitudes[i]*cos(doy*(i+1)*M_PI*2/365 + self._phases[i])
            self._value_cache = val
            self._ts_index_cache = ts_index
        return val

    cpdef update(self, double[:] values):
        n = len(self.amplitudes)
        self.mean = values[0]
        self.amplitudes[...] = values[1:n+1]
        self.phases[...] = values[n+1:]

    cpdef double[:] lower_bounds(self):
        return np.r_[self._mean_lower_bounds, self._amplitude_lower_bounds, self._phase_lower_bounds]

    cpdef double[:] upper_bounds(self):
        return np.r_[self._mean_upper_bounds, self._amplitude_upper_bounds, self._phase_upper_bounds]
parameter_registry.add(AnnualHarmonicSeriesParameter)

cdef enum AggFuncs:
    SUM = 0
    MIN = 1
    MAX = 2
    MEAN = 3
    PRODUCT = 4
    CUSTOM = 5
    ANY = 6
    ALL = 7
_agg_func_lookup = {
    "sum": AggFuncs.SUM,
    "min": AggFuncs.MIN,
    "max": AggFuncs.MAX,
    "mean": AggFuncs.MEAN,
    "product": AggFuncs.PRODUCT,
    "custom": AggFuncs.CUSTOM,
    "any": AggFuncs.ANY,
    "all": AggFuncs.ALL,
}

cdef class AggregatedParameterBase(IndexParameter):
    """Base class for aggregated parameters

    Do not instance this class directly. Use one of the subclasses.
    """
    @classmethod
    def load(cls, model, data):
        parameters_data = data.pop("parameters")
        parameters = set()
        for pdata in parameters_data:
            parameters.add(load_parameter(model, pdata))

        agg_func = data.pop("agg_func", None)
        return cls(parameters=parameters, agg_func=agg_func, **data)

    cpdef add(self, Parameter parameter):
        self.parameters.add(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        self.parameters.remove(parameter)
        parameter.parent.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self, model):
        for parameter in self.parameters:
            parameter.setup(model)

    cpdef after(self, Timestep timestep):
        for parameter in self.parameters:
            parameter.after(timestep)

    cpdef reset(self):
        for parameter in self.parameters:
            parameter.reset()

cdef class AggregatedParameter(AggregatedParameterBase):
    """A collection of IndexParameters

    This class behaves like a set. Parameters can be added or removed from it.
    It's value is the value of it's child parameters aggregated using a
    aggregating function (e.g. sum).

    Parameters
    ----------
    parameters : iterable of `IndexParameter`
        The parameters to aggregate
    agg_func : callable or str
        The aggregation function. Must be one of {"sum", "min", "max", "mean",
        "product"}, or a callable function which accepts a list of values.
    """
    def __init__(self, parameters, agg_func=None, **kwargs):
        super(AggregatedParameter, self).__init__(**kwargs)
        if isinstance(agg_func, basestring):
            agg_func = _agg_func_lookup[agg_func.lower()]
        elif callable(agg_func):
            self.agg_func = agg_func
            agg_func = AggFuncs.CUSTOM
        else:
            raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
        self._agg_func = agg_func
        self.parameters = set(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef Parameter parameter
        cdef double value, value2
        assert(len(self.parameters))
        if self._agg_func == AggFuncs.PRODUCT:
            value = 1.0
            for parameter in self.parameters:
                value *= parameter.value(timestep, scenario_index)
        elif self._agg_func == AggFuncs.SUM:
            value = 0
            for parameter in self.parameters:
                value += parameter.value(timestep, scenario_index)
        elif self._agg_func == AggFuncs.MAX:
            value = float("-inf")
            for parameter in self.parameters:
                value2 = parameter.value(timestep, scenario_index)
                if value2 > value:
                    value = value2
        elif self._agg_func == AggFuncs.MIN:
            value = float("inf")
            for parameter in self.parameters:
                value2 = parameter.value(timestep, scenario_index)
                if value2 < value:
                    value = value2
        elif self._agg_func == AggFuncs.MEAN:
            value = 0
            for parameter in self.parameters:
                value += parameter.value(timestep, scenario_index)
            value /= len(self.parameters)
        elif self._agg_func == AggFuncs.CUSTOM:
            value = self.agg_func([parameter.value(timestep, scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")
        return value

parameter_registry.add(AggregatedParameter)

cdef class AggregatedIndexParameter(AggregatedParameterBase):
    """A collection of IndexParameters

    This class behaves like a set. Parameters can be added or removed from it.
    It's index is the index of it's child parameters aggregated using a
    aggregating function (e.g. sum).

    Parameters
    ----------
    parameters : iterable of `IndexParameter`
        The parameters to aggregate
    agg_func : callable or str
        The aggregation function. Must be one of {"sum", "min", "max"}, or a
        callable function which accepts a list of values.
    """
    def __init__(self, parameters, agg_func=None, **kwargs):
        super(AggregatedIndexParameter, self).__init__(**kwargs)
        if isinstance(agg_func, basestring):
            agg_func = _agg_func_lookup[agg_func.lower()]
        elif callable(agg_func):
            self.agg_func = agg_func
            agg_func = AggFuncs.CUSTOM
        else:
            raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
        self._agg_func = agg_func
        self.parameters = set(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef IndexParameter parameter
        cdef int value, value2
        assert(len(self.parameters))
        if self._agg_func == AggFuncs.SUM:
            value = 0
            for parameter in self.parameters:
                value += parameter.index(timestep, scenario_index)
        elif self._agg_func == AggFuncs.MAX:
            value = INT_MIN
            for parameter in self.parameters:
                value2 = parameter.index(timestep, scenario_index)
                if value2 > value:
                    value = value2
        elif self._agg_func == AggFuncs.MIN:
            value = INT_MAX
            for parameter in self.parameters:
                value2 = parameter.index(timestep, scenario_index)
                if value2 < value:
                    value = value2
        elif self._agg_func == AggFuncs.ANY:
            value = 0
            for parameter in self.parameters:
                value2 = parameter.index(timestep, scenario_index)
                if value2:
                    value = 1
                    break
        elif self._agg_func == AggFuncs.ALL:
            value = 1
            for parameter in self.parameters:
                value2 = parameter.index(timestep, scenario_index)
                if not value2:
                    value = 0
                    break
        elif self._agg_func == AggFuncs.CUSTOM:
            value = self.agg_func([parameter.value(timestep, scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")
        return value

parameter_registry.add(AggregatedIndexParameter)

cdef class RecorderThresholdParameter(IndexParameter):
    """Returns one of two values depending on a Recorder value and a threshold

    Parameters
    ----------
    recorder : `pywr.recorder.Recorder`
    threshold : double
        Threshold to compare the value of the recorder to
    values : iterable of doubles
        If the predicate evaluates False the zeroth value is returned,
        otherwise the first value is returned.
    predicate : string
        One of {"LT", "GT", "EQ", "LE", "GE"}.

    Methods
    -------
    value(timestep, scenario_index)
        Returns a value from the `values` attribute, using the index.
    index(timestep, scenario_index)
        Returns 1 if the predicate evaluates True, else 0.

    Notes
    -----
    On the first day of the model run the recorder will not have a value for
    the previous day. In this case the predicate evaluates to True.
    """

    def __init__(self, Recorder recorder, threshold, values=None, predicate=None):
        super(RecorderThresholdParameter, self).__init__()
        self.recorder = recorder
        self.threshold = threshold
        if values is None:
            self.values = None
        else:
            self.values = np.array(values, np.float64)
        if predicate is None:
            predicate = Predicates.LT
        elif isinstance(predicate, basestring):
            predicate = _predicate_lookup[predicate.upper()]
        self.predicate = predicate

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns a value from the values attribute, using the index"""
        cdef int ind = self.index(timestep, scenario_index)
        cdef double v
        if self.values is not None:
            v = self.values[ind]
        else:
            raise ValueError("values method called, but values not set")
        return v

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0"""
        cdef int index = timestep.index
        cdef double x
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = 1
        else:
            x = self.recorder.data[index-1, scenario_index.global_id]
            if self.predicate == Predicates.LT:
                ind = x < self.threshold
            elif self.predicate == Predicates.GT:
                ind = x > self.threshold
            elif self.predicate == Predicates.LE:
                ind = x <= self.threshold
            elif self.predicate == Predicates.GE:
                ind = x >= self.threshold
            else:
                ind = x == self.threshold
        return ind

    @classmethod
    def load(cls, model, data):
        from pywr._recorders import load_recorder  # delayed to prevent circular reference
        recorder = load_recorder(model, data.pop("recorder"))
        threshold = data.pop("threshold")
        values = data.pop("values")
        predicate = data.pop("predicate", None)
        return cls(recorder, threshold, values, predicate, **data)

parameter_registry.add(RecorderThresholdParameter)

def load_parameter(model, data, parameter_name=None):
    """Load a parameter from a dict"""
    if isinstance(data, basestring):
        # parameter is a reference
        try:
            parameter = model.parameters[data]
        except KeyError:
            parameter = None
        if parameter is None:
            if hasattr(model, "_parameters_to_load"):
                # we're still in the process of loading data from JSON and
                # the parameter requested hasn't been loaded yet - do it now
                name = data
                try:
                    data = model._parameters_to_load.pop(name)
                except KeyError:
                    raise KeyError("Unknown parameter: '{}'".format(data))
                parameter = load_parameter(model, data)
            else:
                raise KeyError("Unknown parameter: '{}'".format(data))
    elif isinstance(data, (float, int)) or data is None:
        # parameter is a constant
        parameter = data
    elif "cached" in data.keys() and data["cached"]:
        # cached parameter wrapper
        del(data["cached"])
        if "name" in data:
            parameter_name = data["name"]
            del(data["name"])
        param = load_parameter(model, data)
        parameter = CachedParameter(param)
    else:
        # parameter is dynamic
        parameter_type = data['type']
        try:
            parameter_name = data["name"]
        except:
            pass

        cls = None
        name2 = parameter_type.lower().replace('parameter', '')
        for parameter_class in parameter_registry:
            name1 = parameter_class.__name__.lower().replace('parameter', '')
            if name1 == name2:
                cls = parameter_class

        if cls is None:
            raise TypeError('Unknown parameter type: "{}"'.format(parameter_type))

        kwargs = dict([(k,v) for k,v in data.items()])
        del(kwargs["type"])
        if "name" in kwargs:
            del(kwargs["name"])
        parameter = cls.load(model, kwargs)

    if parameter_name is not None:
        # TODO FIXME: memory leak if parameter is subsequently removed from the model
        parameter.name = parameter_name
        model.parameters[parameter_name] = parameter

    return parameter


def load_parameter_values(model, data, values_key='values'):
    """ Function to load values from a data dictionary.

    This function tries to load values in to a `np.ndarray` if 'values_key' is
    in 'data'. Otherwise it tries to `load_dataframe` from a 'url' key.

    Parameters
    ----------
    model - `Model` instance
    data - dict
    values_key - str
        Key in data to load values directly to a `np.ndarray`

    """
    if values_key in data:
        # values are given as an array
        values = np.array(data.pop(values_key), np.float64)
    else:
        df = load_dataframe(model, data)
        values = np.squeeze(df.values.astype(np.float64))
    return values


def load_dataframe(model, data):

    # values reference data in an external file
    url = data.pop('url')
    if not os.path.isabs(url) and model.path is not None:
        url = os.path.join(model.path, url)
    try:
        filetype = data.pop('filetype')
    except KeyError:
        # guess file type based on extension
        if url.endswith(('.xls', '.xlsx')):
            filetype = "excel"
        elif url.endswith(('.csv', '.gz')):
            filetype = "csv"
        elif url.endswith(('.hdf', '.hdf5', '.h5')):
            filetype = "hdf"
        else:
            raise NotImplementedError('Unknown file extension: "{}"'.format(url))

    column = data.pop("column", None)
    index = data.pop("index", None)

    if filetype == "csv":
        if hasattr(data, "index_col"):
            data["parse_dates"] = True
            if "dayfirst" not in data.keys():
                data["dayfirst"] = True # we're bias towards non-American dates here
        df = pandas.read_csv(url, **data) # automatically decompressed gzipped data!
    elif filetype == "excel":
        df = pandas.read_excel(url, **data)
    elif filetype == "hdf":
        df = pandas.read_hdf(url, columns=[column], **data)

    if df.index.dtype.name == "object" and data.get("parse_dates", False):
        # catch dates that haven't been parsed yet
        raise TypeError("Invalid DataFrame index type \"{}\" in \"{}\".".format(df.index.dtype.name, url))

    # clean up
    data.pop("parse_dates", None)
    data.pop("dayfirst", None)
    data.pop("index_col", None)

    # if column is not specified, use the whole dataframe
    if column is not None:
        try:
            df = df[column]
        except KeyError:
            raise KeyError('Column "{}" not found in dataset "{}"'.format(column, url))

    if index is not None:
        try:
            df = df.loc[index]
        except KeyError:
            raise KeyError('Index "{}" not found in dataset "{}"'.format(column, url))

    try:
        freq = df.index.inferred_freq
    except AttributeError:
        pass
    else:
        # Convert to regular frequency
        freq = df.index.inferred_freq
        if freq is None:
            raise IndexError("Failed to identify frequency of dataset \"{}\"".format(url))
        df = df.asfreq(freq)
    return df

import os
import numpy as np
cimport numpy as np
from scipy.interpolate import Rbf
import pandas
import json
import calendar
from libc.math cimport cos, M_PI
from libc.limits cimport INT_MIN, INT_MAX
from pywr.h5tools import H5Store
from pywr.hashes import check_hash
from .._core cimport is_leap_year
from ..dataframe_tools import align_and_resample_dataframe, load_dataframe, read_dataframe
import warnings


parameter_registry = {}


class UnutilisedDataWarning(Warning):
    """ Simple warning to indicate that not all data has been used. """
    pass

class TypeNotFoundError(KeyError):
    """
      Key Error, specifically designed for when the 'type' key is not found
      in a dataset. This takes the data value and outputs a summary of it, to
      aid in debugging.
    """
    def __init__(self, data):
        #Try to print out some sensible amount of data without overloading
        #the terminal with data. 1000 chars should be enough to get an idea
        #of what the data looks like. If more than 1000 chars, do a pandas-style
        #summary using ...
        data_str = json.dumps(data)
        if len(data_str) > 1000:
            data_summary = f"{data_str[:500]} ... {data_str[-500:]}"
        else:
            data_summary = data_str

        return f"Unable to find key 'type' in {data_summary}"

cdef class Parameter(Component):
    def __init__(self, *args, is_variable=False, **kwargs):
        super(Parameter, self).__init__(*args, **kwargs)
        self.is_variable = is_variable
        self.double_size = 0
        self.integer_size = 0
        self.is_constant = False

    @classmethod
    def register(cls):
        parameter_registry[cls.__name__.lower()] = cls

    @classmethod
    def unregister(cls):
        del(parameter_registry[cls.__name__.lower()])

    cpdef setup(self):
        super(Parameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__values = np.empty([num_comb], np.float64)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        raise NotImplementedError("Parameter must be subclassed")

    cdef calc_values(self, Timestep timestep):
        # default implementation calls Parameter.value in loop
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self.__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef double get_value(self, ScenarioIndex scenario_index):
        return self.__values[<int>(scenario_index.global_id)]

    cpdef double[:] get_all_values(self):
        return self.__values

    cpdef double get_constant_value(self):
        """Return a constant value.
        
        This method should only be implemented and called if `is_constant` is True. 
        """
        raise NotImplementedError()

    cpdef set_double_variables(self, double[:] values):
        raise NotImplementedError()

    cpdef double[:] get_double_variables(self):
        raise NotImplementedError()

    cpdef double[:] get_double_lower_bounds(self):
        raise NotImplementedError()

    cpdef double[:] get_double_upper_bounds(self):
        raise NotImplementedError()

    cpdef set_integer_variables(self, int[:] values):
        raise NotImplementedError()

    cpdef int[:] get_integer_variables(self):
        raise NotImplementedError()

    cpdef int[:] get_integer_lower_bounds(self):
        raise NotImplementedError()

    cpdef int[:] get_integer_upper_bounds(self):
        raise NotImplementedError()

    property size:
        def __get__(self):
            warnings.warn("Use of the `size` property on Parameters has been deprecated."
                          "Please use either `double_size` or `integer_size` instead.", DeprecationWarning)
            return self.double_size

        def __set__(self, value):
            warnings.warn("Use of the `size` property on Parameters has been deprecated."
                          "Please use either `double_size` or `integer_size` instead.", DeprecationWarning)
            self.double_size = value

    @classmethod
    def load(cls, model, data):
        # If a scenario is given don't pass this to the load values methods
        scenario = data.pop('scenario', None)

        values = load_parameter_values(model, data)
        data.pop("values", None)
        data.pop("url", None)
        name = data.pop("name", None)
        comment = data.pop("comment", None)

        if scenario is not None:
            scenario = model.scenarios[scenario]
            # Only pass scenario object if one provided; most Parameter subclasses
            # do not accept a scenario argument.
            return cls(model, scenario=scenario, values=values, name=name, comment=None, **data)
        else:
            return cls(model, values=values, name=name, comment=None, **data)
Parameter.register()


cdef class ConstantParameter(Parameter):
    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, scale=1.0, offset=0.0, **kwargs):
        super(ConstantParameter, self).__init__(model, **kwargs)
        self._value = value
        self.scale = scale
        self.offset = offset
        self.double_size = 1
        self.integer_size = 0
        self.is_constant = True
        self._lower_bounds = np.ones(self.double_size) * lower_bounds
        self._upper_bounds = np.ones(self.double_size) * upper_bounds

    cdef calc_values(self, Timestep timestep):
        # constant parameter can just set the entire array to one value
        self.__values[...] = self.get_constant_value()

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self.get_constant_value()

    cpdef double get_constant_value(self):
        return self.offset + self._value * self.scale

    cpdef set_double_variables(self, double[:] values):
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter

ConstantParameter.register()


cdef class DataFrameParameter(Parameter):
    """Timeseries parameter with automatic alignment and resampling

    Parameters
    ----------
    model : pywr.model.Model
    dataframe : pandas.DataFrame or pandas.Series
    scenario: pywr._core.Scenario (optional)
    """
    def __init__(self, model, dataframe, scenario=None, **kwargs):
        super(DataFrameParameter, self).__init__(model, *kwargs)
        self.dataframe = dataframe
        self.scenario = scenario

    cpdef setup(self):
        cdef Py_ssize_t i
        super(DataFrameParameter, self).setup()
        # align and resample the dataframe
        dataframe_resampled = align_and_resample_dataframe(self.dataframe, self.model.timestepper.datetime_index)
        if dataframe_resampled.ndim == 1:
            dataframe_resampled = pandas.DataFrame(dataframe_resampled)
        # dataframe should now have the correct number of timesteps for the model
        if len(dataframe_resampled) != len(self.model.timestepper):
            raise ValueError("Aligning DataFrame failed with a different length compared with model timesteps.")
        # check that if a 2D DataFrame is given that we also have a scenario assigned with it
        if dataframe_resampled.ndim == 2 and dataframe_resampled.shape[1] > 1:
            if self.scenario is None:
                raise ValueError("Scenario must be given for a DataFrame input with multiple columns.")
            if self.scenario.size != dataframe_resampled.shape[1]:
                raise ValueError("Scenario size ({}) is different to the number of columns ({}) "
                                 "in the DataFrame input.".format(self.scenario.size, dataframe_resampled.shape[1]))

        if self.scenario:
            # if possible, only load the data required
            scenario_indices = None
            # Default to index that is just out of bounds to cause IndexError if something goes wrong
            self._scenario_ids = np.ones(self.scenario.size, dtype=np.int32) * self.scenario.size

            # Calculate the scenario indices to load dependning on how scenario combinations are defined.
            if self.model.scenarios.user_combinations:
                scenario_indices = set()
                for user_combination in self.model.scenarios.user_combinations:
                    scenario_indices.add(user_combination[self._scenario_index])
                scenario_indices = sorted(list(scenario_indices))
            elif self.scenario.slice:
                scenario_indices = range(*self.scenario.slice.indices(self.scenario.slice.stop))
            else:
                # scenario is defined, but all data required
                self._scenario_ids = None

            if scenario_indices is not None:
                # Now load only the required data
                for n, i in enumerate(scenario_indices):
                    self._scenario_ids[i] = n
                dataframe_resampled = dataframe_resampled.iloc[:, scenario_indices]

        self._values = dataframe_resampled.values.astype(np.float64)
        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef double value
        cdef Py_ssize_t i = timestep.index
        cdef Py_ssize_t j

        if self.scenario is not None:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            value = self._values[i, j]
        else:
            value = self._values[i, 0]
        return value

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        df = load_dataframe(model, data)
        return cls(model, df, scenario=scenario, **data)

DataFrameParameter.register()

cdef class ArrayIndexedParameter(Parameter):
    """Time varying parameter using an array and Timestep.index

    The values in this parameter are constant across all scenarios.
    """
    def __init__(self, model, values, *args, **kwargs):
        super(ArrayIndexedParameter, self).__init__(model, *args, **kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    cdef calc_values(self, Timestep ts):
        # constant parameter can just set the entire array to one value
        self.__values[...] = self.values[ts.index]

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts.index]
ArrayIndexedParameter.register()


cdef class ArrayIndexedScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are vary in time based on index and vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ArrayIndexedScenarioParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Values must be two dimensional.")
        if scenario._size != values.shape[1]:
            raise ValueError("The size of the second dimension of values must equal the size of the scenario.")
        self.values = values
        self._scenario = scenario

    cpdef setup(self):
        super(ArrayIndexedScenarioParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self.values[ts.index, scenario_index._indices[self._scenario_index]]

ArrayIndexedScenarioParameter.register()


cdef class TablesArrayParameter(IndexParameter):
    def __init__(self, model, h5file, node, where='/', scenario=None, **kwargs):
        """
        This Parameter reads array data from a PyTables HDF database.

        The parameter reads data using the PyTables array interface and therefore
        does not require loading the entire dataset in to memory. This is useful
        for large model runs.

        Parameters
        ----------
        h5file : tables.File or filename
            The tables file handle or filename to attach the CArray objects to. If a
            filename is given the object will open and close the file handles.
        node : string
            Name of the node in the tables database to read data from
        where : string
            Path to read the node from.
        scenario : Scenario
            Scenario to use as the second index in the array.
        """
        super(TablesArrayParameter, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        self.node = node
        self.where = where
        self.scenario = scenario

        # Private attributes, initialised during setup()
        self._values_dbl = None  # Stores the loaded data if float
        self._values_int = None  # Stores the loaded data if integer
        # If a scenario is present this is the index in the model list of scenarios
        self._scenario_index = -1
        self._scenario_ids = None  # Lookup of scenario index to the loaded data index

    cpdef setup(self):
        cdef Py_ssize_t n, i

        super(TablesArrayParameter, self).setup()
        self._scenario_index = -1
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

        self.h5store = H5Store(self.h5file, None, "r")
        node = self.h5store.file.get_node(self.where, self.node)
        if not node.dtype in (np.float32, np.float64, np.int8, np.int16, np.int32):
            raise TypeError("Unexpected dtype in array: {}".format(node.dtype))

        # check the shape of the data is valid
        if self.scenario is not None:
            if node.shape[1] < self.scenario.size:
                raise IndexError('The length of the second dimension ({:d}) of the tables node ({}:{}) '
                                 'should be the same as the size of the specified Scenario ({:d}).'
                                 .format(node.shape[1], node._v_file.filename, node._v_pathname, self.scenario.size))
            elif node.shape[1] > self.scenario.size:
                warnings.warn('The length of the second dimension ({:d}) of the tables node ({}:{}) '
                              'is greater than the size of the specified Scenario ({:d}). '
                              'Not all data is being used!'.format(node.shape[1], node._v_file.filename, node._v_pathname, self.scenario.size),
                              UnutilisedDataWarning)
        if node.shape[0] < len(self.model.timestepper):
            raise IndexError('The length of the first dimension ({:d}) of the tables node ({}:{}) '
                             'should be equal to or greater than the number of timesteps.'
                             .format(node.shape[0], node._v_file.filename, node._v_pathname, len(self.model.timestepper)))
        elif node.shape[0] > len(self.model.timestepper):
            warnings.warn('The length of the first dimension ({:d}) of the tables node ({}:{}) '
                          'is greater than the number of timesteps. Not all data is being used!'
                          .format(node.shape[0], node._v_file.filename, node._v_pathname, len(self.model.timestepper)),
                          UnutilisedDataWarning)

        # detect data type and read into memoryview
        self._values_dbl = None
        self._values_int = None
        if self.scenario:
            # if possible, only load the data required
            scenario_indices = None
            # Default to index that is just out of bounds to cause IndexError if something goes wrong
            self._scenario_ids = np.ones(self.scenario.size, dtype=np.int32) * self.scenario.size

            # Calculate the scenario indices to load dependning on how scenario combinations are defined.
            if self.model.scenarios.user_combinations:
                scenario_indices = set()
                for user_combination in self.model.scenarios.user_combinations:
                    scenario_indices.add(user_combination[self._scenario_index])
                scenario_indices = sorted(list(scenario_indices))
            elif self.scenario.slice:
                scenario_indices = range(*self.scenario.slice.indices(self.scenario.slice.stop))
            else:
                # scenario is defined, but all data required
                self._scenario_ids = None

            if scenario_indices is not None:
                # Now load only the required data
                for n, i in enumerate(scenario_indices):
                    self._scenario_ids[i] = n

                if node.dtype in (np.float32, np.float64):
                    self._values_dbl = node[:len(self.model.timestepper), scenario_indices].astype(np.float64)
                else:
                    self._values_int = node[:len(self.model.timestepper), scenario_indices].astype(np.int32)

        if node.dtype in (np.float32, np.float64):
            if self._values_dbl is None:
                self._values_dbl = node.read().astype(np.float64)
            # negative values are often erroneous
            if np.min(self._values_dbl) < 0.0:
                warnings.warn('Negative values in input file "{}" from node: {}'.format(self.h5file, self.node))
            if not np.all(np.isfinite(self._values_dbl)):
                raise ValueError('Non-finite values in input file "{}" from node: {}'.format(self.h5file, self.node))
        else:
            if self._values_int is None:
                self._values_int = node.read().astype(np.int32)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef Py_ssize_t i = ts.index
        cdef Py_ssize_t j
        if self._values_dbl is None:
            return float(self.index(ts, scenario_index))
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_dbl[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_dbl[i, j]

    cpdef int index(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef Py_ssize_t i = ts.index
        cdef Py_ssize_t j
        if self._values_int is None:
            return int(self.value(ts, scenario_index))
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_int[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_int[i, j]

    cpdef finish(self):
        self.h5store = None

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]

        url = data.pop('url')
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        node = data.pop('node')
        where = data.pop('where', '/')

        # Check hashes if given before reading the data
        checksums = data.pop('checksum', {})
        for algo, hash in checksums.items():
            check_hash(url, hash, algorithm=algo)

        return cls(model, url, node, where=where, scenario=scenario)
TablesArrayParameter.register()


cdef class ConstantScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self):
        super(ConstantScenarioParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
ConstantScenarioParameter.register()


cdef class ArrayIndexedScenarioMonthlyFactorsParameter(Parameter):
    """Time varying parameter using an array and Timestep.index with
    multiplicative factors per Scenario
    """
    def __init__(self, model, Scenario scenario, values, factors, *args, **kwargs):
        """
        values is the baseline timeseries data that is perturbed by a factor. The
        factor is taken from factors which is shape (scenario.size, 12). Therefore
        factors vary with the individual scenarios in scenario and month.
        """
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).__init__(model, *args, **kwargs)

        values = np.asarray(values, dtype=np.float64)
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError("Factors must be two dimensional.")

        if factors.shape[0] != scenario._size:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if factors.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = values
        self._factors = factors

    cpdef setup(self):
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.month-1
        cdef int i = scenario_index._indices[self._scenario_index]
        return self._values[ts.index]*self._factors[i, imth]

    @classmethod
    def load(cls, model, data):
        scenario = data.pop("scenario", None)
        if scenario is not None:
            scenario = model.scenarios[scenario]

        if isinstance(data["values"], list):
            values = np.asarray(data["values"], np.float64)
        elif isinstance(data["values"], dict):
            values = load_parameter_values(model, data["values"])
        else:
            raise TypeError("Unexpected type for \"values\" in {}".format(cls.__name__))

        if isinstance(data["factors"], list):
            factors = np.asarray(data["factors"], np.float64)
        elif isinstance(data["factors"], dict):
            factors = load_parameter_values(model, data["factors"])
        else:
            raise TypeError("Unexpected type for \"factors\" in {}".format(cls.__name__))

        return cls(model, scenario, values, factors)

ArrayIndexedScenarioMonthlyFactorsParameter.register()


cdef class DailyProfileParameter(Parameter):
    """ An annual profile consisting of daily values.

    This parameter provides a repeating annual profile with a daily resolution. A total of 366 values
    must be provided. These values are coerced to a `numpy.array` internally.

    Parameters
    ----------
    values : iterable, array
        The 366 values that represent the daily profile.

    """
    def __init__(self, model, values, *args, **kwargs):
        super(DailyProfileParameter, self).__init__(model, *args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[ts.dayofyear_index]
DailyProfileParameter.register()

cdef class WeeklyProfileParameter(Parameter):
    """Weekly profile (52-week year)

    The last week of the year will have more than 7 days, as 365 / 7 is not whole.
    """
    def __init__(self, model, values, *args, **kwargs):
        super(WeeklyProfileParameter, self).__init__(model, *args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) == 53:
            values = values[:52]
            warnings.warn("Truncating 53 week profile to 52 weeks.")
        if len(values) != 52:
            raise ValueError("52 values must be given for a weekly profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[ts.week_index]
WeeklyProfileParameter.register()


cdef class MonthlyProfileParameter(Parameter):
    """Parameter which provides a monthly profile.

    The monthly profile returns a different value based on the month of the current
    time-step. By default this creates a piecewise profile with a step change at the
    beginning of each month. An optional `interp_day` keyword can instead create a
    linearly interpolated daily profile assuming the given values correspond to either
    the first or last day of the month.

    Parameters
    ----------
    values : iterable, array
        The 12 values that represent the monthly profile.
    lower_bounds : float or array_like (default=0.0)
        Defines the lower bounds when using optimisation. If a float given, same bound applied for every
        month. Otherwise an array like object of length 12 should be given for as separate value each month.
    upper_bounds : float or array_like (default=np.inf)
        Defines the upper bounds when using optimisation. If a float given, same bound applied for every
        month. Otherwise an array like object of length 12 should be given for as separate value each month.
    interp_day : str or None (default=None)
        If `interp_day` is None then no interpolation is undertaken, and the parameter
         returns values representing a piecewise monthly profile. Otherwise `interp_day`
         must be a string of either "first" or "last" representing which day of the month
         each of the 12 values represents. The parameter then returns linearly
         interpolated values between the given day of the month.


    See also
    --------
    ScenarioMonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, values, lower_bounds=0.0, upper_bounds=np.inf, interp_day=None, **kwargs):
        super(MonthlyProfileParameter, self).__init__(model, **kwargs)
        self.double_size = 12
        self.integer_size = 0
        if len(values) != self.double_size:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = np.array(values)
        self.interp_day = interp_day

        if np.isscalar(lower_bounds):
            lb = np.ones(self.double_size) * lower_bounds
        else:
            lb = np.array(lower_bounds)
            if len(lb) != self.double_size:
                raise ValueError("Lower bounds must be a scalar or array like of 12 values.")
        self._lower_bounds = lb

        if np.isscalar(upper_bounds):
            ub = np.ones(self.double_size) * upper_bounds
        else:
            ub = np.array(upper_bounds)
            if len(ub) != self.double_size:
                raise ValueError("Upper bounds must be a scalar or array like of 12 values.")
        self._upper_bounds = ub


    cpdef reset(self):
        Parameter.reset(self)
        # The interpolated profile is recalculated during reset so that
        # it will update when the _values array is updated via `set_double_variables`
        # and the model is rerun. I.e. during optimisation (where setup is not redone).
        if self.interp_day is not None:
            self._interpolate()

    cpdef _interpolate(self):

        # Create an array to save the daily profile in.
        self._interp_values = np.zeros(366)
        cdef int i = 0
        cdef int mth

        # Create interpolation knots depending on values
        if self.interp_day == 'first':
            x = [1]  # First month
            y = []
            for mth in range(1, 13):
                x.append(x[-1] + calendar.monthrange(2015, mth)[1])
                y.append(self._values[mth-1])
            y.append(self._values[0])
        elif self.interp_day == 'last':
            x = [0]  # End of previous year
            y = [self._values[11]]  # Use value from December
            for mth in range(1, 13):
                x.append(x[-1] + calendar.monthrange(2015, mth)[1])
                y.append(self._values[mth-1])
        else:
            raise ValueError(f'Interpolation day "{self.interp_day}" not supported.')

        # Do the interpolation
        values = np.interp(np.arange(365) + 1, x, y)
        # Make the daily profile of 366 values repeating the same value for 28th & 29th Feb.
        for i in range(365):
            if i < 58:
                self._interp_values[i] = values[i]
            elif i == 58:
                self._interp_values[i] = values[i]
                self._interp_values[i+1] = values[i]
            elif i > 58:
                self._interp_values[i+1] = values[i]

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        if self.interp_day is None:
            return self._values[ts.month-1]
        else:
            return self._interp_values[ts.dayofyear_index]

    cpdef set_double_variables(self, double[:] values):
        self._values[...] = values

    cpdef double[:] get_double_variables(self):
        # Make sure we return a copy of the data instead of a view.
        return np.array(self._values).copy()

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds
MonthlyProfileParameter.register()


cdef class ScenarioMonthlyProfileParameter(Parameter):
    """ Parameter that provides a monthly profile per scenario

    Behaviour is the same as `MonthlyProfileParameter` except a different
    profile is returned for each ensemble in a given scenario.

    See also
    --------
    MonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, Scenario scenario, values, **kwargs):
        super(ScenarioMonthlyProfileParameter, self).__init__(model, **kwargs)

        if values.ndim != 2:
            raise ValueError("Factors must be two dimensional.")

        if scenario._size != values.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if values.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = np.array(values)

    cpdef setup(self):
        super(ScenarioMonthlyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[scenario_index._indices[self._scenario_index], ts.month-1]

ScenarioMonthlyProfileParameter.register()

cdef class ScenarioWeeklyProfileParameter(Parameter):
    """Parameter that provides a weekly profile per scenario

    This parameter provides a repeating annual profile with a weekly resolution. A
    different profile is returned for each member of a given scenario

    Parameters
    ----------
    scenario: Scenario
        Scenario object over which different profiles should be provided.
    values : iterable, array
        Length of 1st dimension should equal the number of members in the scenario object
        and the length of the second dimension should be 52

    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        values = np.array(values)
        if values.ndim != 2:
            raise ValueError("Factors must be two dimensional.")
        if scenario._size != values.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if values.shape[1] != 52:
            raise ValueError("52 values must be given for a weekly profile.")
        self._values = values
        self._scenario = scenario

    cpdef setup(self):
        super(ScenarioWeeklyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[scenario_index._indices[self._scenario_index], ts.week_index]

ScenarioWeeklyProfileParameter.register()

cdef class ScenarioDailyProfileParameter(Parameter):
    """Parameter which provides a daily profile per scenario.

    This parameter provides a repeating annual profile with a daily resolution. A
    different profile is returned for each member of a given scenario

    Parameters
    ----------
    scenario: Scenario
        Scenario object over which different profiles should be provided
    values : iterable, array
        Length of 1st dimension should equal the number of members in the scenario object
        and the length of the second dimension should be 366

    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        values = np.array(values)
        if values.ndim != 2:
            raise ValueError("Factors must be two dimensional.")
        if scenario._size != values.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if values.shape[1] != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = values
        self._scenario = scenario

    cpdef setup(self):
        super(ScenarioDailyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.dayofyear_index
        return self._values[scenario_index._indices[self._scenario_index], i]

ScenarioDailyProfileParameter.register()


cdef class UniformDrawdownProfileParameter(Parameter):
    """Parameter which provides a uniformly reducing value from one to zero.

     This parameter is intended to be used with an `AnnualVirtualStorage` node to provide a profile
     that represents perfect average utilisation of the annual volume. It returns a value of 1 on the
     reset day, and subsequently reduces by 1/366 every day afterward.

    Parameters
    ----------
    reset_day: int
        The day of the month (1-31) to reset the volume to the initial value.
    reset_month: int
        The month of the year (1-12) to reset the volume to the initial value.
    residual_days: int
        The number of days of residual licence to target for the end of the year.

    See also
    --------
    AnnualVirtualStorage
    """
    def __init__(self, model, reset_day=1, reset_month=1, residual_days=0, **kwargs):
        super().__init__(model, **kwargs)
        self.reset_day = reset_day
        self.reset_month = reset_month
        self.residual_days = residual_days

    cpdef reset(self):
        super(UniformDrawdownProfileParameter, self).reset()
        # Reset day of the year based on a leap year.
        # Note that this is zero-based
        self._reset_idoy = pandas.Period(year=2016, month=self.reset_month, day=self.reset_day, freq='D').dayofyear - 1

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int current_idoy = ts.dayofyear_index
        cdef int total_days_in_period
        cdef int days_into_period
        cdef int year = ts.year
        cdef double residual_proportion, slope

        days_into_period = current_idoy - self._reset_idoy
        if days_into_period < 0:
            # We're not past the reset day yet; use the previous year
            year -= 1

        if self._reset_idoy > 59:
            # Reset occurs after February therefore next year's February might be a leap year?
            year += 1

        # Determine the number of days in the period based on whether there is a leap year or not in the current period
        if is_leap_year(year):
            total_days_in_period = 366
        else:
            total_days_in_period = 365

        # Now determine number of days we're into the period if it has wrapped around to a new year
        if days_into_period < 0:
            days_into_period += 366
            # Need to adjust for post 29th Feb in non-leap years.
            # Recall `current_idoy` was incremented by 1 if it is a non-leap already (hence comparison to 59)
            if not is_leap_year(ts.year) and current_idoy > 59:
                days_into_period -= 1

        residual_proportion = self.residual_days / total_days_in_period
        slope = (residual_proportion - 1.0) / total_days_in_period

        return 1 + (slope * days_into_period)

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
UniformDrawdownProfileParameter.register()


cdef class RbfProfileParameter(Parameter):
    """Parameter which interpolates a daily profile using a radial basis function (RBF).

    The daily profile is computed during model `reset` using a radial basis function with
    day-of-year as the independent variables. The days of the year are defined by the user
    alongside the values to use on each of those days for the interpolation. The first
    day of the years should always be one, and its value is repeated as the 366th value.
    In addition the second and penultimate values are mirrored to encourage a consistent
    gradient to appear across the boundary. The RBF calculations are undertaken using
    the `scipy.interpolate.Rbf` object, please refer to Scipy's documentation for more
    information.

    Parameters
    ----------
    days_of_year : iterable, integer
        The days of the year at which the interpolation values are defined. The first
        value should be one.
    values : iterable, float
        Values to use for interpolation corresponding to the `days_of_year`.
    lower_bounds : float or array_like (default=0.0)
        Defines the lower bounds when using optimisation. If a float given, same bound applied for every day of the
        year. Otherwise an array like object of length equal to the number of days of the year should be given.
    upper_bounds : float or array_like (default=np.inf)
        Defines the upper bounds when using optimisation. If a float given, same bound applied for every day of the
        year. Otherwise an array like object of length equal to the number of days of the year should be given.
    variable_days_of_year_range : int (default=0)
        The maximum bounds (positive or negative) for the days of year during optimisation. A non-zero value
        will cause the days of the year values to be exposed as integer variables (except the first value which
        remains at day 1). This value is bounds on those variables as maximum shift from the given `days_of_year`.
    min_value, max_value : float
        Optionally cap the interpolated daily profile to a minimum and/or maximum value. The default values
        are negative and positive infinity for minimum and maximum respectively.
    rbf_kwargs: Optional, dict
        Optional dictionary of keyword arguments to base to the Rbf object.

    """
    def __init__(self, model, days_of_year, values, lower_bounds=0.0, upper_bounds=np.inf, rbf_kwargs=None,
                 variable_days_of_year_range=0, min_value=-np.inf, max_value=np.inf, **kwargs):
        super(RbfProfileParameter, self).__init__(model, **kwargs)

        if len(days_of_year) != len(values):
            raise ValueError(f"The length of values ({len(values)}) must equal the length of "
                             f"`days_of_year` ({len(days_of_year)}).")

        self.variable_days_of_year_range = variable_days_of_year_range
        self.double_size = len(values)
        self._values = np.array(values, dtype=np.float64)
        self.days_of_year = days_of_year
        self.min_value = min_value
        self.max_value = max_value

        if np.isscalar(lower_bounds):
            lb = np.ones(self.double_size) * lower_bounds
        else:
            lb = np.array(lower_bounds)
            if len(lb) != self.double_size:
                raise ValueError("Lower bounds must be a scalar or array like with length equivalent to rbf values")
        self._lower_bounds = lb

        if np.isscalar(upper_bounds):
            ub = np.ones(self.double_size) * upper_bounds
        else:
            ub = np.array(upper_bounds)
            if len(ub) != self.double_size:
               raise ValueError("Upper bounds must be a scalar or array like with length equivalent to rbf values")
        self._upper_bounds = ub

        if self.variable_days_of_year_range > 0:
            if np.any(np.diff(self.days_of_year) <= 2*self.variable_days_of_year_range):
                raise ValueError(f"The days of the year are too close together for the given "
                                 f"`variable_days_of_year_range`. This could cause the optimised days"
                                 f"of the year to overlap and become out of order.  Either increase the"
                                 f"spacing of the days of the year or reduce `variable_days_of_year_range` to"
                                 f"less than half the closest distance between the days of the year.")
            self.integer_size = len(values) - 1
            self._doy_lower_bounds = np.array([d - self.variable_days_of_year_range
                                               for d in self.days_of_year[1:]], dtype=np.int32)
            self._doy_upper_bounds = np.array([d + self.variable_days_of_year_range
                                               for d in self.days_of_year[1:]], dtype=np.int32)
        else:
            self.integer_size = 0

        self.rbf_kwargs = rbf_kwargs if rbf_kwargs is not None else {}

    property days_of_year:
        def __get__(self):
            return np.array(self._days_of_year)
        def __set__(self, values):
            values = np.array(values, dtype=np.int32)
            if values[0] != 1:
                raise ValueError('The first day of the years must be 1.')
            if len(values) < 3:
                raise ValueError('At least 3 days of the year are required.')
            if np.any(np.diff(values) <= 0):
                raise ValueError('The days of the year should be strictly monotonically increasing.')
            if np.any(0 > values > 365):
                raise ValueError('Days of the years should be between 1 and 365 inclusive.')
            self._days_of_year = values

    cpdef reset(self):
        Parameter.reset(self)
        # The interpolated profile is recalculated during reset so that
        # it will update when the _values array is updated via `set_double_variables`
        # and the model is rerun. I.e. during optimisation (where setup is not redone).
        self._interpolate()

    cpdef _interpolate(self):
        cdef int i
        cdef double[:] values
        cdef double v

        days_of_year = list(self._days_of_year)
        # Append day 365 to the list and mirror the penultimate and second DOY at the start and end
        # of the list respectively. This helps ensure the gradient is roughly the same across the boundary
        # between days 365 and 0.
        days_of_year = [days_of_year[-1]-365] + list(days_of_year) + [366, 366+days_of_year[1]-1]
        # Create the corresponding y values including the mirrored entries
        y = list(self._values)
        y = [y[-1]] + y + [y[0], y[1]]
        rbfi = Rbf(days_of_year, y, **self.rbf_kwargs)

        # Do the interpolation
        values = rbfi(np.arange(365) + 1)

        # Create an array to save the daily profile in.
        self._interp_values = np.zeros(366)
        # Make the daily profile of 366 values repeating the same value for 28th & 29th Feb.
        for i in range(365):
            v = max(min(values[i], self.max_value), self.min_value)
            if i < 58:
                self._interp_values[i] = v
            elif i == 58:
                self._interp_values[i] = v
                self._interp_values[i+1] = v
            elif i > 58:
                self._interp_values[i+1] = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.dayofyear_index
        return self._interp_values[i]

    # Double variables are for the known interpolation values (y-axis)
    cpdef set_double_variables(self, double[:] values):
        self._values[...] = values

    cpdef double[:] get_double_variables(self):
        # Make sure we return a copy of the data instead of a view.
        return np.array(self._values).copy()

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds

    # Integer variables are for the days of the year positions (if optimised)
    cpdef set_integer_variables(self, int[:] values):
        self.days_of_year = [1] + np.array(values).tolist()

    cpdef int[:] get_integer_variables(self):
        return np.array(self.days_of_year[1:], dtype=np.int32)

    cpdef int[:] get_integer_lower_bounds(self):
        return self._doy_lower_bounds

    cpdef int[:] get_integer_upper_bounds(self):
        return self._doy_upper_bounds

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
RbfProfileParameter.register()


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
        return float(self.get_index(scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index"""
        # return index as an integer
        return 0

    cpdef setup(self):
        super(IndexParameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__indices = np.empty([num_comb], np.int32)

    cdef calc_values(self, Timestep timestep):
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self.__indices[<int>(scenario_index.global_id)] = self.index(timestep, scenario_index)
            self.__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef int get_index(self, ScenarioIndex scenario_index):
        return self.__indices[<int>(scenario_index.global_id)]

    cpdef int[:] get_all_indices(self):
        return self.__indices
IndexParameter.register()


cdef class ConstantScenarioIndexParameter(IndexParameter):
    """A Scenario varying IndexParameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioIndexParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size, dtype=np.int32)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self):
        super(ConstantScenarioIndexParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
ConstantScenarioIndexParameter.register()


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
    def __init__(self, model, index_parameter, params, **kwargs):
        super(IndexedArrayParameter, self).__init__(model, **kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.children.add(index_parameter)

        self.params = []
        for p in params:
            if not isinstance(p, Parameter):
                from pywr.parameters import ConstantParameter
                p = ConstantParameter(model, p)
            self.params.append(p)

        for param in self.params:
            self.children.add(param)
        self.children.add(index_parameter)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the Parameter at the current index"""
        cdef int index
        index = self.index_parameter.get_index(scenario_index)
        cdef Parameter parameter = self.params[index]
        return parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop("index_parameter"))
        try:
            parameters = data.pop("params")
        except KeyError:
            parameters = data.pop("parameters")
        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
        return cls(model, index_parameter, parameters, **data)
IndexedArrayParameter.register()


cdef class AnnualHarmonicSeriesParameter(Parameter):
    """ A `Parameter` which returns the value from an annual harmonic series

    This `Parameter` comprises a series N cosine function with a period of 365
     days. The calculation is performed using the Julien day of the year minus 1
     This causes a small discontinuity in non-leap years.

    .. math:: f(t) = A + \sum_{n=1}^N A_n\cdot \cos((2\pi nt)/365+\phi_n)

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
    def __init__(self, model, mean, amplitudes, phases, *args, **kwargs):
        if len(amplitudes) != len(phases):
            raise ValueError("The number  of amplitudes and phases must be the same.")
        n = len(amplitudes)
        self.mean = mean
        self._amplitudes = np.array(amplitudes)
        self._phases = np.array(phases)

        self._mean_lower_bounds = kwargs.pop('mean_lower_bounds', 0.0)
        self._mean_upper_bounds = kwargs.pop('mean_upper_bounds', np.inf)
        self._amplitude_lower_bounds = np.ones(n)*kwargs.pop('amplitude_lower_bounds', 0.0)
        self._amplitude_upper_bounds = np.ones(n)*kwargs.pop('amplitude_upper_bounds', np.inf)
        self._phase_lower_bounds = np.ones(n)*kwargs.pop('phase_lower_bounds', 0.0)
        self._phase_upper_bounds = np.ones(n)*kwargs.pop('phase_upper_bounds', np.pi*2)
        super(AnnualHarmonicSeriesParameter, self).__init__(model, *args, **kwargs)
        # Size must be set after call to super.
        self.double_size = 1 + 2*n
        self._value_cache = 0.0
        self._ts_index_cache = -1

    @classmethod
    def load(cls, model, data):
        mean = data.pop('mean')
        amplitudes = data.pop('amplitudes')
        phases = data.pop('phases')

        return cls(model, mean, amplitudes, phases, **data)

    property amplitudes:
        def __get__(self):
            return np.asarray(self._amplitudes)

    property phases:
        def __get__(self):
            return np.asarray(self._phases)

    cpdef reset(self):
        Parameter.reset(self)
        self._value_cache = 0.0
        self._ts_index_cache = -1

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef int ts_index = timestep.index
        cdef int doy = timestep.dayofyear - 1
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

    cpdef set_double_variables(self, double[:] values):
        n = len(self.amplitudes)
        self.mean = values[0]
        self._amplitudes[...] = values[1:n+1]
        self._phases[...] = values[n+1:]

    cpdef double[:] get_double_variables(self):
        return np.r_[np.array([self.mean, ]), np.array(self.amplitudes), np.array(self.phases)]

    cpdef double[:] get_double_lower_bounds(self):
        return np.r_[self._mean_lower_bounds, self._amplitude_lower_bounds, self._phase_lower_bounds]

    cpdef double[:] get_double_upper_bounds(self):
        return np.r_[self._mean_upper_bounds, self._amplitude_upper_bounds, self._phase_upper_bounds]
AnnualHarmonicSeriesParameter.register()

cdef enum AggFuncs:
    SUM = 0
    MIN = 1
    MAX = 2
    MEAN = 3
    PRODUCT = 4
    CUSTOM = 5
    ANY = 6
    ALL = 7
    MEDIAN = 8
_agg_func_lookup = {
    "sum": AggFuncs.SUM,
    "min": AggFuncs.MIN,
    "max": AggFuncs.MAX,
    "mean": AggFuncs.MEAN,
    "product": AggFuncs.PRODUCT,
    "custom": AggFuncs.CUSTOM,
    "any": AggFuncs.ANY,
    "all": AggFuncs.ALL,
    "median": AggFuncs.MEDIAN,
}
_agg_func_lookup_reverse = {v: k for k, v in _agg_func_lookup.items()}

def wrap_const(model, value):
    if isinstance(value, (int, float)):
        value = ConstantParameter(model, value)
    return value


cdef class AggregatedParameter(Parameter):
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
    def __init__(self, model, parameters, agg_func=None, **kwargs):
        super(AggregatedParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        parameters_data = data.pop("parameters")
        parameters = []
        for pdata in parameters_data:
            parameter = load_parameter(model, pdata)
            parameters.append(wrap_const(model, parameter))

        agg_func = data.pop("agg_func", None)
        return cls(model, parameters=parameters, agg_func=agg_func, **data)

    property agg_func:
        def __get__(self):
            if self._agg_func == AggFuncs.CUSTOM:
                return self._agg_user_func
            return _agg_func_lookup_reverse[self._agg_func]
        def __set__(self, agg_func):
            self._agg_user_func = None
            if isinstance(agg_func, str):
                agg_func = _agg_func_lookup[agg_func.lower()]
            elif callable(agg_func):
                self._agg_user_func = agg_func
                agg_func = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
            self._agg_func = agg_func

    cpdef add(self, Parameter parameter):
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        self.parameters.remove(parameter)
        parameter.parents.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        super(AggregatedParameter, self).setup()
        assert(len(self.parameters))

    cdef calc_values(self, Timestep timestep):
        cdef Parameter parameter
        cdef double[:] accum = self.__values  # View of the underlying location for the data
        cdef double[:] values
        cdef int i
        cdef int nparam
        cdef int n = accum.shape[0]
        cdef ScenarioIndex scenario_index

        if self._agg_func == AggFuncs.PRODUCT:
            accum[...] = 1.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] *= values[i]
        elif self._agg_func == AggFuncs.SUM:
            accum[...] = 0.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] += values[i]
        elif self._agg_func == AggFuncs.MAX:
            accum[...] = np.NINF
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    if values[i] > accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MIN:
            accum[...] = np.PINF
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    if values[i] < accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MEAN:
            accum[...] = 0.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] += values[i]

            nparam = len(self.parameters)
            for i in range(n):
                accum[i] /= nparam

        elif self._agg_func == AggFuncs.MEDIAN:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = np.median([parameter.get_value(scenario_index) for parameter in self.parameters])
        elif self._agg_func == AggFuncs.CUSTOM:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = self._agg_user_func([parameter.get_value(scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")
AggregatedParameter.register()

cdef class AggregatedIndexParameter(IndexParameter):
    """A collection of IndexParameters

    This class behaves like a set. Parameters can be added or removed from it.
    Its index is the index of it's child parameters aggregated using a
    aggregating function (e.g. sum).

    Parameters
    ----------
    parameters : iterable of `IndexParameter`
        The parameters to aggregate
    agg_func : callable or str
        The aggregation function. Must be one of {"sum", "min", "max", "any",
        "all", "product"}, or a callable function which accepts a list of values.
    """
    def __init__(self, model, parameters, agg_func=None, **kwargs):
        super(AggregatedIndexParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        parameters_data = data.pop("parameters")
        parameters = list()
        for pdata in parameters_data:
            parameter = load_parameter(model, pdata)
            parameters.append(wrap_const(model, parameter))

        agg_func = data.pop("agg_func", None)
        return cls(model, parameters=parameters, agg_func=agg_func, **data)

    property agg_func:
        def __get__(self):
            if self._agg_func == AggFuncs.CUSTOM:
               return self._agg_user_func
            return _agg_func_lookup_reverse[self._agg_func]
        def __set__(self, agg_func):
            self._agg_user_func = None
            if isinstance(agg_func, str):
                agg_func = _agg_func_lookup[agg_func.lower()]
            elif callable(agg_func):
                self._agg_user_func = agg_func
                agg_func = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
            self._agg_func = agg_func

    cpdef add(self, Parameter parameter):
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        self.parameters.remove(parameter)
        parameter.parents.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        super(AggregatedIndexParameter, self).setup()
        assert len(self.parameters)
        assert all([isinstance(parameter, IndexParameter) for parameter in self.parameters])

    cdef calc_values(self, Timestep timestep):
        cdef IndexParameter parameter
        cdef int[:] accum = self.__indices  # View of the underlying location for the data
        cdef int[:] values
        cdef int i
        cdef int nparam
        cdef int n = accum.shape[0]
        cdef ScenarioIndex scenario_index

        if self._agg_func == AggFuncs.PRODUCT:
            accum[...] = 1
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    accum[i] *= values[i]
        elif self._agg_func == AggFuncs.SUM:
            accum[...] = 0
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    accum[i] += values[i]
        elif self._agg_func == AggFuncs.MAX:
            accum[...] = INT_MIN
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i] > accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MIN:
            accum[...] = INT_MAX
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i] < accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.ANY:
            accum[...] = 0
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i]:
                        accum[i] = 1
        elif self._agg_func == AggFuncs.ALL:
            accum[...] = 1
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if not values[i]:
                        accum[i] = 0

        elif self._agg_func == AggFuncs.CUSTOM:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = self._agg_user_func([parameter.get_index(scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")

        # Finally set the float values
        for i in range(n):
            self.__values[i] = accum[i]


AggregatedIndexParameter.register()


cdef class DivisionParameter(Parameter):
    """ Parameter that divides one `Parameter` by another.

    Parameters
    ----------
    denominator : `Parameter`
        The parameter to use as the denominator (or divisor).
    numerator : `Parameter`
        The parameter to use as the numerator (or dividend).
    """
    def __init__(self, model, numerator, denominator, **kwargs):
        super().__init__(model, **kwargs)
        self._numerator = None
        self._denominator = None
        self.numerator = numerator
        self.denominator = denominator

    property numerator:
        def __get__(self):
            return self._numerator
        def __set__(self, parameter):
            # remove any existing parameter
            if self._numerator is not None:
                self._numerator.parents.remove(self)

            self._numerator = parameter
            self.children.add(parameter)

    property denominator:
        def __get__(self):
            return self._denominator
        def __set__(self, parameter):
            # remove any existing parameter
            if self._denominator is not None:
                self._denominator.parents.remove(self)

            self._denominator = parameter
            self.children.add(parameter)

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = self._numerator.__values[i] / self._denominator.__values[i]

    @classmethod
    def load(cls, model, data):
        numerator = load_parameter(model, data.pop("numerator"))
        denominator = load_parameter(model, data.pop("denominator"))
        return cls(model, numerator, denominator, **data)
DivisionParameter.register()


cdef class NegativeParameter(Parameter):
    """ Parameter that takes negative of another `Parameter`

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    """
    def __init__(self, model, parameter, *args, **kwargs):
        super(NegativeParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = -self.parameter.__values[i]

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
NegativeParameter.register()


cdef class MaxParameter(Parameter):
    """ Parameter that takes maximum of another `Parameter` and constant value (threshold)

    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is compared to constant value.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    threshold : float (default=0.0)
        The threshold value to compare with the given parameter.
    """
    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        super(MaxParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = max(self.parameter.__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MaxParameter.register()


cdef class NegativeMaxParameter(MaxParameter):
    """ Parameter that takes maximum of the negative of a `Parameter` and constant value (threshold) """
    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = max(-self.parameter.__values[i], self.threshold)

NegativeMaxParameter.register()


cdef class MinParameter(Parameter):
    """ Parameter that takes minimum of another `Parameter` and constant value (threshold)

    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is compared to constant value.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    threshold : float (default=0.0)
        The threshold value to compare with the given parameter.
    """
    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        super(MinParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = min(self.parameter.__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MinParameter.register()


cdef class NegativeMinParameter(MinParameter):
    """ Parameter that takes minimum of the negative of a `Parameter` and constant value (threshold) """
    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = min(-self.parameter.__values[i], self.threshold)
NegativeMinParameter.register()


cdef class OffsetParameter(Parameter):
    """Parameter that offsets another `Parameter` by a constant value.

    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is offset by a constant value.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to compare with the float.
    offset : float (default=0.0)
        The offset to apply to the value returned by `parameter`.
    lower_bounds : float (default=0.0)
        The lower bounds of the offset when used during optimisation.
    upper_bounds : float (default=np.inf)
        The upper bounds of the offset when used during optimisation.
    """
    def __init__(self, model, parameter, offset=0.0, lower_bounds=0.0, upper_bounds=np.inf, *args, **kwargs):
        super(OffsetParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.offset = offset
        self.double_size = 1
        self._lower_bounds = np.ones(self.double_size) * lower_bounds
        self._upper_bounds = np.ones(self.double_size) * upper_bounds

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = self.parameter.__values[i] + self.offset

    cpdef set_double_variables(self, double[:] values):
        self.offset = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self.offset, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
OffsetParameter.register()


cdef class DeficitParameter(Parameter):
    """Parameter track the deficit (max_flow - actual flow) of a Node

    Parameters
    ----------
    model : pywr.model.Model
    node : Node
      The node that will have it's deficit tracked

    Notes
    -----
    This parameter is a little unusual in that it's value is calculated during
    the after method, not calc_values. It is intended to be used in combination
    with a recorder (e.g. NumpyArrayNodeRecorder) to record the deficit (
    defined as requested - actual flow) at a node. Note that this means
    recording this parameter does *not* give you the value that was used by
    the solver in this timestep. Alternatively, this parameter can be used
    in the model by other parameters and will evaluate to *yesterdays* deficit,
    where the deficit in the zeroth timestep is zero.
    """
    def __init__(self, model, node, *args, **kwargs):
        super(DeficitParameter, self).__init__(model, *args, **kwargs)
        self.node = node

    cpdef reset(self):
        self.__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        pass # calculation done in after

    cpdef after(self):
        cdef double[:] max_flow
        cdef int i
        if self.node._max_flow_param is None:
            for i in range(0, self.node._flow.shape[0]):
                self.__values[i] = self.node._max_flow - self.node._flow[i]
        else:
            max_flow = self.node._max_flow_param.get_all_values()
            for i in range(0, self.node._flow.shape[0]):
                self.__values[i] = max_flow[i] - self.node._flow[i]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        return cls(model, node=node, **data)

DeficitParameter.register()


cdef class FlowParameter(Parameter):
    """Parameter that provides the flow from a node from the previous time-step.

    Parameters
    ----------
    model : pywr.model.Model
    node : Node
      The node that will have its flow tracked
    initial_value : float (default=0.0)
      The value to return on the first  time-step before the node has any past flow.

    Notes
    -----
    This parameter keeps track of the previous time step's flow on the given node. These
    values can be used in calculations for the current timestep as though this was any
    other parameter.
    """
    def __init__(self, model, node, *args, **kwargs):
        self.initial_value = kwargs.pop('initial_value', 0)
        super().__init__(model, *args, **kwargs)
        self.node = node

    cpdef setup(self):
        super(FlowParameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__next_values = np.empty([num_comb], np.float64)

    cpdef reset(self):
        self.__next_values[...] = self.initial_value
        self.__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        cdef int i
        for i in range(self.__values.shape[0]):
            self.__values[i] = self.__next_values[i]

    cpdef after(self):
        cdef int i
        for i in range(self.node._flow.shape[0]):
            self.__next_values[i] = self.node._flow[i]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        return cls(model, node=node, **data)
FlowParameter.register()


cdef class PiecewiseIntegralParameter(Parameter):
    """Parameter that integrates a piecewise function.

    This parameter calculates the integral of a piecewise function. The
    piecewise function is given as two arrays (`x` and `y`) and is assumed to
    start from (0, 0). The values of `x` should be monotonically increasing
    and greater than zero.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter the defines the right hand bounds of the integration.
    x : iterable of doubles
    y : iterable of doubles

    """
    def __init__(self, model, parameter, x, y, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

    cpdef setup(self):
        super(PiecewiseIntegralParameter, self).setup()

        if len(self.x) != len(self.y):
            raise ValueError('The length of `x` and `y` should be the same.')

        if np.any(np.diff(self.x) < 0):
            raise ValueError('The array `x` should be monotonically increasing.')

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef double integral = 0.0
        cdef double x = self.parameter.get_value(scenario_index)
        cdef int i
        cdef double dx, prev_x

        prev_x = 0
        for i in range(self.x.shape[0]):
            if x < self.x[i]:
                dx = x - prev_x
            else:
                dx = self.x[i] - prev_x

            if dx < 0.0:
                break
            else:
                integral += dx * self.y[i]
            prev_x = self.x[i]
        return integral

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop('parameter'))
        return cls(model, parameter, **data)
PiecewiseIntegralParameter.register()


cdef class FlowDelayParameter(Parameter):
    """Parameter that returns the delayed flow for a node after a given number of timesteps or days

    Parameters
    ----------
    model : `pywr.model.Model`
    node: Node
        The node to delay for.
    timesteps: int
        Number of timesteps to delay the flow.
    days: int
        Number of days to delay the flow. Specifying a number of days (instead of a number
        of timesteps) is only valid if the number of days is exactly divisible by the model timestep length.
    initial_flow: float
        Flow value to return for initial model timesteps prior to any delayed flow being available. This
        value is constant across all delayed timesteps and any model scenarios. Default is 0.0.
    """

    def __init__(self, model, node, *args, **kwargs):
        self.node = node
        self.timesteps = kwargs.pop('timesteps', 0)
        self.days = kwargs.pop('days', 0)
        self.initial_flow = kwargs.pop('initial_flow', 0.0)
        super().__init__(model, *args, **kwargs)

    cpdef setup(self):
        super(FlowDelayParameter, self).setup()
        cdef int r
        if self.days > 0:
            r = self.days % self.model.timestepper.delta
            if r == 0:
                self.timesteps = self.days / self.model.timestepper.delta
            else:
                raise ValueError('The delay defined as number of days is not exactly divisible by the timestep delta.')
        if self.timesteps < 1:
            raise ValueError('The number of time-steps for a FlowDelayParameter node must be greater than one.')
        self._memory = np.zeros((self.timesteps,  len(self.model.scenarios.combinations)))
        self._memory_pointer = 0

    cpdef reset(self):
        self._memory[...] = self.initial_flow
        self._memory_pointer = 0

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._memory[self._memory_pointer, scenario_index.global_id]

    cpdef after(self):
        for i in range(self._memory.shape[1]):
            self._memory[self._memory_pointer, i] = self.node._flow[i]
        if self.timesteps > 1:
            self._memory_pointer = (self._memory_pointer + 1) % self.timesteps

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        return cls(model, node, **data)

FlowDelayParameter.register()


cdef class DiscountFactorParameter(Parameter):
    """Parameter that returns the current discount factor based on discount rate and a base year.

    Parameters
    ----------
    discount_rate : float
        Discount rate (expressed as 0 - 1) used calculate discount factor for each year.
    base_year : int
        Discounting base year (i.e. the year with a discount factor equal to 1.0).
    """

    def __init__(self, model, rate, base_year, **kwargs):
        super(DiscountFactorParameter, self).__init__(model, **kwargs)
        self.rate = rate
        self.base_year = base_year

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return 1 / pow(1.0 + self.rate, ts.year - self.base_year)

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)

DiscountFactorParameter.register()


def get_parameter_from_registry(parameter_type):
    key = parameter_type.lower()
    try:
        return parameter_registry[key]
    except KeyError:
        pass
    if key.endswith("parameter"):
        key.replace("parameter", "")
    else:
        key = key + "parameter"
    try:
        return parameter_registry[key]
    except KeyError:
        raise TypeError('Unknown parameter type: "{}"'.format(parameter_type))


def load_parameter(model, data, parameter_name=None):
    """Load a parameter from a dict"""
    if isinstance(data, str):
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
    else:
        # parameter is dynamic

        try:
             parameter_type = data['type']
        except KeyError:
            #raise custom exception that makes the error a bit easier to interpret
            raise TypeNotFoundError(data)

        try:
            parameter_name = data["name"]
        except:
            pass

        cls = get_parameter_from_registry(parameter_type)

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


def load_parameter_values(model, data, values_key='values', url_key='url',
                          table_key='table'):
    """ Function to load values from a data dictionary.

    This function tries to load values in to a `np.ndarray` if 'values_key' is
    in 'data'. Otherwise it tries to `load_dataframe` from a 'url' key.

    Parameters
    ----------
    model - `Model` instance
    data - dict
    values_key - str
        Key in data to load values directly to a `np.ndarray`
    url_key - str
        Key in data to load values directly from an external file reference (using pandas)
    table_key - str
        Key in data to load values directly from an external file reference (using pandas)
    """
    if values_key in data:
        # values are given as an array
        values = np.array(data.pop(values_key), np.float64)
    elif url_key in data or table_key in data:
        df = load_dataframe(model, data)
        try:
            # If it's a DataFrame we coerce to a numpy array
            values = df.values
        except AttributeError:
            values = df
        values = np.squeeze(values.astype(np.float64))
    else:
        # Try to get some useful information about the parameter for the error message
        name = data.get('name', None)
        ptype = data.get('type', None)
        raise ValueError("Parameter ('{name}' of type '{ptype}' is missing a valid key to load its values. "
                         "Please provide either a '{}', '{}' or '{}' entry.".format(values_key, url_key, table_key, name=name, ptype=ptype))
    return values

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
    """Abstract class used to create a model parameter. This is a component that returns
    a value for each model time step and scenario.

    Attributes
    ----------
    model : Model
        The model instance.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
    double_size : int
        The number of double variables in the parameter.
    integer_size : int
        The number of integer variables in the parameter.
    is_constant : bool
        Whether the parameter is constant (i.e. the value does not change with the timestep).
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, *args, is_variable=False, **kwargs):
        """Initialise the class.
        
        Parameters
        ----------
        args : tuple
            Any positional argument.
        is_variable : bool, default=False
            Whether the parameter is set as variable to solve an optimisation problem.
        kwargs : dict
            Any keyword argument.
        """
        super(Parameter, self).__init__(*args, **kwargs)
        self.is_variable = is_variable
        self.double_size = 0
        self.integer_size = 0
        self.is_constant = False

    @classmethod
    def register(cls):
        """Register the parameter in the global registry."""
        parameter_registry[cls.__name__.lower()] = cls

    @classmethod
    def unregister(cls):
        """Remove the parameter from the global registry."""
        del(parameter_registry[cls.__name__.lower()])

    cpdef setup(self):
        """Setup the parameter. This initialises the internal values as empty array."""
        super(Parameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self._Parameter__values = np.empty([num_comb], np.float64)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        raise NotImplementedError("Parameter must be subclassed")

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        # default implementation calls Parameter.value in loop
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self._Parameter__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef double get_value(self, ScenarioIndex scenario_index):
        """Get the parameter value for a scenario when its value was last updated.
        
        Parameters
        ----------
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self._Parameter__values[<int>(scenario_index.global_id)]

    cpdef double[:] get_all_values(self):
        """Get all parameter values for all scenarios when its value was last updated.

        Returns
        -------
        numpy.typing.NDArray[np.number]
            An array with the values. The array size equals the number of scenarios.
        """
        return self._Parameter__values

    cpdef double get_constant_value(self):
        """Return a constant value.
        
        Returns
        -------
        float
            The value.
        Notes
        ------
        This method should only be implemented and called if `is_constant` is True. 

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.number]
            The variable to set. The size must equal the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef set_integer_variables(self, int[:] values):
        """Set the parameter integer variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.int_]
            The variable to set. The size must equal the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef int[:] get_integer_variables(self):
        """Get the parameter integer variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.int_]
            The array with the variables. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef int[:] get_integer_lower_bounds(self):
        """Get the lower bounds of the integer variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.int_]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
        raise NotImplementedError()

    cpdef int[:] get_integer_upper_bounds(self):
        """Get the upper bounds of the integer variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.int_]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.

        Raises
        -------
        NotImplementedError
            If the parameter does not support variable values.
        """
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
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        Parameter
            The loaded class.
        """
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
    """A parameter that returns a non-time dependant number.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import ConstantParameter
    
    model = Model()
    ConstantParameter(model=model, value=1.0, scale=0.5, name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "ConstantParameter",
            "value": 1.0,
            "scale": 0.5
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    scale : Optional[float], default=1.0
        Scale the value by the given amount.
    offset : Optional[float], default=1.0
        Offset the value by the given amount.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
    lower_bounds : Optional[float], default=0
        The lower bound to use for the value during an optimisation problem.
    upper_bounds : Optional[float], default=np.inf
        The upper bound to use for the value during an optimisation problem.
    double_size : int
        The number of double variables in the parameter.
    integer_size : int
        The number of integer variables in the parameter.
    is_constant : bool
        Whether the parameter is constant (i.e. the value does not change with the timestep).
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    
    Optimisation
    -----------
    This parameter can be optimised.
    """

    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, scale=1.0, offset=0.0, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        value : float | int
            The constant value to use at each timestep.
        scale : Optional[float], default=1.0
            Scale the value by the given amount.
        offset : Optional[float], default=1.0
            Offset the value by the given amount.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        lower_bounds : Optional[float], default=0
            The lower bound to use for the value during an optimisation problem.
        upper_bounds : Optional[float], default=np.inf
            The upper bound to use for the value during an optimisation problem.
        kwargs : dict
            Any other keyword argument.
        """
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
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        # constant parameter can just set the entire array to one value
        self._Parameter__values[...] = self.get_constant_value()

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self.get_constant_value()

    cpdef double get_constant_value(self):
        """Get the parameter value with `scale` and `offset`. 
        
        Returns
        -------
        float
            The parameter value.
        """
        return self.offset + self._value * self.scale

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.number]
            The variable to set. The size must equal the number of variables the parameter handles.
        """
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        ConstantParameter
            The loaded class.
        """
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter

ConstantParameter.register()


cdef class DataFrameParameter(Parameter):
    """Timeseries parameter with automatic alignment and resampling from a pandas Series or DataFrame object.
    When a DataFrame is provided, each column represents a different [pywr.core.Scenario][] to run.


    Examples
    -------
    Python
    ======
    ```python
    import pandas as pd
    from pywr.parameters import DataFrameParameter
    df = pd.read_csv("file.csv", index_col=[0], parse_dates=True, dayfirst=True)
    DataFrameParameter(
        model=model, 
        dataframe=df["Flow"], 
    )
    ```

    JSON
    ======
    File in `"url"` is parsed by the `load_dataframe` function in Pywe which accepts
    Pandas's `read_csv` parameters. The series is extracted using the `"column"` option.
    ```json
    {
        "My parameter": {
            "type": "DataFrameParameter",
            "url": "file.csv" 
            "index_col": [0], 
            "parse_dates": True,
            "dayfirst": True,
            "column": "Flow"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    dataframe : pandas.DataFrame | pandas.Series
        The pandas DataFrame or Series object conntaining the data. The index must contain
        the dates.
    scenario: Optional[Scenario]
        When a DataFrame instead oa a Series is provided, you must specify the scenario the DataFrame
        refers to. Each column in the DataFrame represent a scenario ensemble; the number of column must
        equal the scenario size.
    timestep_offset : int
        Optional offset to apply to the timestep look-up. This can be used to look forward (positive value) or
        backward (negative value) in the dataset. The offset is applied to dataset after alignment and resampling.
        If the offset takes the indexing out of the data bounds then the parameter will return the first or last
        value available.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, dataframe, scenario=None, timestep_offset=0, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The pywr model instance.
        dataframe : pandas.DataFrame | pandas.Series
            The pandas DataFrame or Series object conntaining the data. The index must contain
            the dates.
        scenario: Optional[Scenario], default=None
            When a DataFrame instead oa a Series is provided, you must specify the scenario the DataFrame
            refers to. Each column in the DataFrame represent a scenario ensemble; the number of column must
            equal the scenario size.
        timestep_offset : Optional[int], default=0
            Optional offset to apply to the timestep look-up. This can be used to look forward (positive value) or
            backward (negative value) in the dataset. The offset is applied to dataset after alignment and resampling.
            If the offset takes the indexing out of the data bounds then the parameter will return the first or last
            value available.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(DataFrameParameter, self).__init__(model, *kwargs)
        self.dataframe = dataframe
        self.scenario = scenario
        self.timestep_offset = timestep_offset

    cpdef setup(self):
        """Setup the parameter by aligning and resampling the data.
        
        Raises
        ------
        ValueError
            If Pywr fails to align the DataFrame to the [pywr.core.Timestepper][] data or the 
            DataFrame size does not match the scenario size.
        """
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

        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)
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

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef double value
        cdef Py_ssize_t i = min(max(timestep.index + self.timestep_offset, 0), self._values.shape[0] - 1)
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
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        DataFrameParameter
            The loaded class.
        """
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        timestep_offset = data.pop('timestep_offset', 0)
        # This will consume all keyword arguments silently in pandas. I.e. don't rely on **data passing keywords
        df = load_dataframe(model, data)
        return cls(model, df, scenario=scenario, timestep_offset=timestep_offset, **data)

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
        self._Parameter__values[...] = self.values[ts.index]

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
    """
    This Parameter reads array data from a [PyTables HDF database](https://www.pytables.org).

    The parameter reads data using the PyTables array interface and therefore
    does not require loading the entire dataset into memory. This is useful
    for large model runs.

    Attributes
    ----------
    model : Model
        The model instance.
    h5file : tables.File | str | Path
        The tables file handle or filename to attach the CArray objects to. If a
        filename is given the object will open and close the file handles.
    h5store : H5Store
        The H5Store object with the data. 
    node : string
        Name of the node in the tables database to read data from
    where : string
        Path to read the node from.
    scenario : Scenario
        Scenario to use as the second index in the array.
    timestep_offset : int
        Optional offset to apply to the timestep look-up. This can be used to look forward (positive value) or
        backward (negative value) in the dataset. The offset is applied to dataset after alignment and resampling.
        If the offset takes the indexing out of the data bounds then the parameter will return the first or last
        value available.
    """
    def __init__(self, model, h5file, node, where='/', scenario=None, timestep_offset=0, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        h5file : tables.File | str | Path
            The tables file handle or filename to attach the CArray objects to. If a
            filename is given the object will open and close the file handles.
        node : string
            Name of the node in the tables database to read data from
        where : Optional[string], default="/"
            Path to read the node from.
        scenario : Optional[Scenario], default=None
            Scenario to use as the second index in the array.
        timestep_offset : Optional[int], default=0
            Optional offset to apply to the timestep look-up. This can be used to look forward (positive value) or
            backward (negative value) in the dataset. The offset is applied to dataset after alignment and resampling.
            If the offset takes the indexing out of the data bounds then the parameter will return the first or last
            value available.
        """
        super(TablesArrayParameter, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        self.node = node
        self.where = where
        self.scenario = scenario
        self.timestep_offset = timestep_offset

        # Private attributes, initialised during setup()
        self._values_dbl = None  # Stores the loaded data if float
        self._values_int = None  # Stores the loaded data if integer
        # If a scenario is present this is the index in the model list of scenarios
        self._scenario_index = -1
        self._scenario_ids = None  # Lookup of scenario index to the loaded data index

    cpdef setup(self):
        """Read the data.
        
        Raises
        -------
        TypeError
            If the node's data is not a valid int or float.
        IndexError
            If the table dimensions do not match the scenario size or number of model timesteps.
        """
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
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        if self._values_dbl is None:
            return float(self.index(ts, scenario_index))

        i = min(max(ts.index + self.timestep_offset, 0), self._values_dbl.shape[0] - 1)
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_dbl[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_dbl[i, j]

    cpdef int index(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Returns the integer value.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The value.
        """
        cdef Py_ssize_t i
        cdef Py_ssize_t j
        if self._values_int is None:
            return int(self.value(ts, scenario_index))

        i = min(max(ts.index + self.timestep_offset, 0), self._values_int.shape[0] - 1)
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_int[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_int[i, j]

    cpdef finish(self):
        """Reset the store."""
        self.h5store = None

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).
        This also checks the file checksum if given.

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        TablesArrayParameter
            The loaded class.
        """
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
            model.check_hash(url, hash, algorithm=algo)

        return cls(model, url, node, where=where, scenario=scenario)
TablesArrayParameter.register()


cdef class ConstantScenarioParameter(Parameter):
    """A [pywr.core.Scenario][] varying [pywr.parameters.Parameter][].

    The values in this parameter are constant in time, but vary within a single Scenario. Use this
    parameter if you are using model scenarios and you want to change a constant value based on the
    scenario Pywr is running.

    Attributes
    ----------
    model : Model
        The model instance.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            The scenario the constant parameters are applied to.
        values : Iterable[float | int]
            The constant values to use at each timestep and scenario. This iterable must have the same length as scenario.size.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        kwargs : dict
            Any other keyword argument.
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
        """Setup the parameter."""
        super(ConstantScenarioParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
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
            values = np.asarray(data.pop("values"), np.float64)
        elif isinstance(data["values"], dict):
            values = load_parameter_values(model, data.pop("values"))
        else:
            raise TypeError("Unexpected type for \"values\" in {}".format(cls.__name__))

        if isinstance(data["factors"], list):
            factors = np.asarray(data.pop("factors"), np.float64)
        elif isinstance(data["factors"], dict):
            factors = load_parameter_values(model, data.pop("factors"))
        else:
            raise TypeError("Unexpected type for \"factors\" in {}".format(cls.__name__))

        return cls(model, scenario, values, factors, **data)

ArrayIndexedScenarioMonthlyFactorsParameter.register()


cdef class DailyProfileParameter(Parameter):
    """An annual profile consisting of 366 values.

    This parameter provides a repeating annual profile with a daily resolution. A total of 366 values
    must be provided. These values are coerced to a `numpy.array` internally.

    Attributes
    ----------
    model : Model
        The model instance.
    values : Iterable[float] | numpy.typing.NDArray[np.number]
        The 366 values that represent the daily profile.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
     lower_bounds : Optional[float]
        The lower bound to use for the value during an optimisation problem.
    upper_bounds : Optional[float]
        The upper bound to use for the value during an optimisation problem.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    Optimisation
    -----------
    This parameter can be optimised.

    """
    def __init__(self, model, values, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        values : Iterable[float] | numpy.typing.NDArray[np.number]
            The 366 values that represent the daily profile.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        lower_bounds : Optional[float], default=0
            The lower bound to use for the value during an optimisation problem.
        upper_bounds : Optional[float], default=np.inf
            The upper bound to use for the value during an optimisation problem.
        kwargs : dict
            Any other keyword argument.
        """

        super(DailyProfileParameter, self).__init__(model, *args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self._values[ts.dayofyear_index]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array(self._values).copy()
DailyProfileParameter.register()

cdef class WeeklyProfileParameter(Parameter):
    """Weekly profile of 52 weeks.

    The last week of the year will have more than 7 days, as 365 / 7 is not whole.

    Attributes
    ----------
    model : Model
        The model instance.
    values : Iterable[float] | numpy.typing.NDArray[np.number]
        The 52 values that represent the daily profile. If the iterable has
        53 values, it it truncated to 52.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, values, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        values : Iterable[float] | numpy.typing.NDArray[np.number]
            The 52 values that represent the daily profile. If the iterable has
            53 values, it it truncated to 52.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        kwargs : dict
            Any other keyword argument.
        """
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
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self._values[ts.week_index]
WeeklyProfileParameter.register()


cdef class MonthlyProfileParameter(Parameter):
    """Parameter which provides a monthly profile with 12 values.

    The monthly profile returns a different value based on the month of the current
    time-step. By default this creates a piecewise profile with a step change at the
    beginning of each month. An optional `interp_day` keyword can instead create a
    linearly interpolated daily profile assuming the given values correspond to either
    the first or last day of the month.

    Attributes
    ----------
    model : Model
        The model instance.
    values : Iterable[float] | numpy.typing.NDArray[np.number]
        The 12 values that represent the monthly profile.
    interp_day : Optional[Literal["first", "last"]]
        If `interp_day` is None then no interpolation is undertaken, and the parameter
        returns values representing a piecewise monthly profile. Otherwise `interp_day`
        must be a string of either "first" or "last" representing which day of the month
        each of the 12 values represents. The parameter then returns linearly
        interpolated values between the given day of the month.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
    lower_bounds : Optional[float | numpy.typing.NDArray[np.number]]
        The lower bounds when using optimisation. If a float given, the same bound applied for every
        month. 
    upper_bounds : Optional[float | numpy.typing.NDArray[np.number]]
        The upper bounds when using optimisation. If a float given, the same bound applied for every
        month. 
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    Optimisation
    -----------
    This parameter can be optimised.

    See also
    --------
    ScenarioMonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, values, lower_bounds=0.0, upper_bounds=np.inf, interp_day=None, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
        The model instance.
        values : Iterable[float] | numpy.typing.NDArray[np.number]
            The 12 values that represent the monthly profile.
        interp_day : Optional[Literal["first", "last"]], default=None
            If `interp_day` is `None` then no interpolation is undertaken, and the parameter
            returns values representing a piecewise monthly profile. Otherwise `interp_day`
            must be a string of either "first" or "last" representing which day of the month
            each of the 12 values represents. The parameter then returns linearly
            interpolated values between the given day of the month.
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        lower_bounds : Optional[float | numpy.typing.NDArray[np.number]]
            Define the lower bounds when using optimisation. If a float given, the same bound applied for every
            month. Otherwise an array like object of length 12 should be given for as separate value each month.
        upper_bounds : Optional[float | numpy.typing.NDArray[np.number]]
            Define the upper bounds when using optimisation. If a float given, the same bound applied for every
            month. Otherwise an array like object of length 12 should be given for as separate value each month.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
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
        """Reset the interal values."""
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
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        if self.interp_day is None:
            return self._values[ts.month-1]
        else:
            return self._interp_values[ts.dayofyear_index]

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.number]
            The variable to set. The size must equal the number of variables the parameter handles.
        """
        self._values[...] = values

    cpdef double[:] get_daily_values(self):
        """Get a profile of daily values.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the monthly profile converted to a daily profile of 366 values.
        """
        cdef int i, mth
        if self.interp_day is not None:
            return np.array(self._interp_values).copy()
        else:
            daily_values = []
            for mth in range(0, 12):
                for i in range(0, calendar.monthrange(2016, mth+1)[1]):
                    daily_values.append(self._values[mth])
            return np.asarray(daily_values)

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        # Make sure we return a copy of the data instead of a view.
        return np.array(self._values).copy()

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._upper_bounds
MonthlyProfileParameter.register()


cdef class ScenarioMonthlyProfileParameter(Parameter):
    """Parameter that provides a monthly profile per scenario

    This parameter provides a repeating annual profile with a monthly resolution. A
    different profile is returned for each member of a given scenario

    Attributes
    ----------
    model : Model
        The model instance.
    scenario : Scenario
        Scenario object over which different profiles should be provided
    values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
        Length of 1st dimension should equal the number of members in the scenario object
        and the length of the second dimension should be 12.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs.

    See also
    --------
    MonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, Scenario scenario, values, **kwargs):
        """Initialise the class.
            
        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            Scenario object over which different profiles should be provided.
        values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
            Length of 1st dimension should equal the number of members in the scenario object
            and the length of the second dimension should be 12.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.

        Raises
        -------
        ValueError
            If the first dimension of `values` is less than 2 or the second dimension is not 12.
        """
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
        """Setup the parameter."""
        super(ScenarioMonthlyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self._values[scenario_index._indices[self._scenario_index], ts.month-1]

ScenarioMonthlyProfileParameter.register()

cdef class ScenarioWeeklyProfileParameter(Parameter):
    """Parameter that provides a weekly profile per scenario

    This parameter provides a repeating annual profile with a weekly resolution. A
    different profile is returned for each member of a given scenario.

    Attributes
    ----------
    model : Model
        The model instance.
    scenario : Scenario
        Scenario object over which different profiles should be provided
    values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
        Length of 1st dimension should equal the number of members in the scenario object
        and the length of the second dimension should be 52
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs.

    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """Initialise the class.
            
        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            Scenario object over which different profiles should be provided.
        values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
            Length of 1st dimension should equal the number of members in the scenario object
            and the length of the second dimension should be 52.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.

        Raises
        -------
        ValueError
            If the first dimension of `values` is less than 2 or the second dimension is not 52.
        """
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
        """Setup the parameter."""
        super(ScenarioWeeklyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        return self._values[scenario_index._indices[self._scenario_index], ts.week_index]

ScenarioWeeklyProfileParameter.register()

cdef class ScenarioDailyProfileParameter(Parameter):
    """Parameter which provides a daily profile per scenario.

    This parameter provides a repeating annual profile with a daily resolution. A
    different profile is returned for each member of a given scenario.

    Attributes
    ----------
    model : Model
        The model instance.
    scenario : Scenario
        Scenario object over which different profiles should be provided
    values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
        Length of 1st dimension should equal the number of members in the scenario object
        and the length of the second dimension should be 366
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """Initialise the class.
            
        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            Scenario object over which different profiles should be provided.
        values : Iterable[Iterable[float]] | numpy.typing.NDArray[numpy.typing.NDArray[np.number]]
            Length of 1st dimension should equal the number of members in the scenario object
            and the length of the second dimension should be 366.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.

        Raises
        -------
        ValueError
            If the first dimension of `values` is less than 2 or the second dimension is not 366.
        """
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
        """Setup the parameter."""
        super(ScenarioDailyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef int i = ts.dayofyear_index
        return self._values[scenario_index._indices[self._scenario_index], i]

ScenarioDailyProfileParameter.register()


cdef class UniformDrawdownProfileParameter(Parameter):
    """Parameter which provides a uniformly reducing value from one to zero.

     This parameter is intended to be used with an [pywr.nodes.AnnualVirtualStorage][] node to provide a profile
     that represents perfect average utilisation of the annual volume. It returns a value of 1 on the
     reset day, and subsequently reduces by 1/366 every day afterward.

    Attributes
    ----------
    model : Model
        The model instance.
    reset_day : int
        The day of the month (1-31) to reset the volume to the initial value.
    reset_month : int
        The month of the year (1-12) to reset the volume to the initial value.
    residual_days : int
        The number of days of residual licence to target for the end of the year.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    See also
    --------
    AnnualVirtualStorage
    """
    def __init__(self, model, reset_day=1, reset_month=1, residual_days=0, **kwargs):
        """Initiliase the parameter.

        Parameters
        ----------
        model : Model
            The model instance.
        reset_day : Optional[int], default=1
            The day of the month (1-31) to reset the volume to the initial value.
        reset_month : Optional[int], default=1
            The month of the year (1-12) to reset the volume to the initial value.
        residual_days : Optional[int], default=0
            The number of days of residual licence to target for the end of the year.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super().__init__(model, **kwargs)
        self.reset_day = reset_day
        self.reset_month = reset_month
        self.residual_days = residual_days

    cpdef reset(self):
        """Reset the day of the year."""
        super(UniformDrawdownProfileParameter, self).reset()
        # Reset day of the year based on a leap year.
        # Note that this is zero-based
        self._reset_idoy = pandas.Period(year=2016, month=self.reset_month, day=self.reset_day, freq='D').dayofyear - 1

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
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
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        UniformDrawdownProfileParameter
            The loaded class.
        """
        return cls(model, **data)
UniformDrawdownProfileParameter.register()


cdef class RbfProfileParameter(Parameter):
    """Parameter which interpolates a daily profile using a radial basis function (RBF).

    The daily profile is computed during model `reset` using a radial basis function with
    day-of-year as the independent variables. The days of the year are defined by the user
    alongside the values to use on each of those days for the interpolation. The first
    day of the years should always be one, and its value is repeated as the 366<sup>th</sup> value.
    In addition the second and penultimate values are mirrored to encourage a consistent
    gradient to appear across the boundary. The RBF calculations are undertaken using
    the `scipy.interpolate.Rbf` object, please refer to 
    [Scipy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html#scipy.interpolate.Rbf)
    for more information.

    Examples
    -------
    Python
    ======
    ```python
    model = Model()
    # optimise parameter and varies x2 between 35 and 55, x3 between 220 and 240,
    # varies y1 between 0 and 0.2, y2 between 0.4 and 0.8 and y3 between 0.6 and 0.8.
    RbfProfileParameter(
        model=model, 
        name="My parameter", 
        days_of_year=[1, 45, 230], 
        values=[0.2, 0.5, 0.7],
        is_variable=True,
        lower_bounds=0.1,
        upper_bounds=[0.1, 0.3, 0.1],
        variable_days_of_year_range=10
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "RbfProfileParameter",
            "days_of_year" [1, 45, 230], 
            "values": [0.2, 0.5, 0.7],
            "is_variable": true,
            "lower_bounds" 0.1,
            "upper_bounds": [0.1, 0.3, 0.1],
            "variable_days_of_year_range": 10
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    days_of_year : Iterable[integer]
        The days of the year at which the interpolation values are defined. The first
        value should be one.
    values : Iterable[float]
        Values (or y coordinates) to use for interpolation corresponding to the `days_of_year`.
    min_value : float
        Optionally cap the interpolated daily profile to a minimum value. 
    max_value : float
        Optionally cap the interpolated daily profile to a maximum value. 
    rbf_kwargs : dict
        Optional dictionary of keyword arguments to base to the Rbf object.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
    variable_days_of_year_range : int
        The maximum bounds (positive or negative) for the days of year during optimisation. A non-zero value
        will cause the days of the year values to be exposed as integer variables (except the first value which
        remains at day 1). This value is bounds on those variables as maximum shift from the given `days_of_year`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs.
        
    Optimisation
    -----------
    This parameter can be optimised.
    """
    def __init__(self, model, days_of_year, values, lower_bounds=0.0, upper_bounds=np.inf, rbf_kwargs=None,
                 variable_days_of_year_range=0, min_value=-np.inf, max_value=np.inf, **kwargs):
        """Initialise the class.
        
        Parameters
        ----------
        model : Model
            The model instance.
        days_of_year : Iterable[integer]
            The days of the year at which the interpolation values are defined. The first
            value should be one.
        values : Iterable[float]
            Values (or y coordinates) to use for interpolation corresponding to the `days_of_year`.
        min_value : Optional[float], default=-np.inf
            Optionally cap the interpolated daily profile to a minimum value. 
        max_value : Optional[float], default=np.inf
            Optionally cap the interpolated daily profile to a maximum value. 
        rbf_kwargs : Optional[dict], default=None
            Optional dictionary of keyword arguments to base to the Rbf object.
        is_variable : Optional[bool], default=False
            Whether the parameter is set as variable to solve an optimisation problem.
        variable_days_of_year_range : Optional[int], default=0
            The maximum bounds (positive or negative) for the days of year during optimisation. A non-zero value
            will cause the days of the year values to be exposed as integer variables (except the first value which
            remains at day 1). This value is bounds on those variables as maximum shift from the given `days_of_year`.
        lower_bounds : float | numpy.typing.NDArray[np.number], default=0.0
            Defines the lower bounds when using optimisation. If a float is iven, the same bound applied for every day of the
            year. Otherwise an array like object of length equal to the number of days of the year should be given.
        upper_bounds : float | numpy.typing.NDArray[np.number], default=0.0
            Defines the upper bounds when using optimisation. If a float is given, the same bound applied for every day of the
            year. Otherwise an array like object of length equal to the number of days of the year should be given.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.

        Raises
        ------
        ValueError
            If the optimisation bounds are not valid.
        """

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
            if np.any((0 > values) | (values > 365)):
                raise ValueError('Days of the years should be between 1 and 365 inclusive.')
            self._days_of_year = values

    cpdef reset(self):
        """Reset the interal values."""
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
        """Get the parameter value. This is scaled and offset and the given timestep and scenario are ignored.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef int i = ts.dayofyear_index
        return self._interp_values[i]

    # Double variables are for the known interpolation values (y-axis)
    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.number]
            The variable to set. The size must equal the number of y coordinates.
        """
        self._values[...] = values

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of y coordinates.
        """
        # Make sure we return a copy of the data instead of a view.
        return np.array(self._values).copy()

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the lower bounds for each variable. The array size equals the number of y coordinates.
        """
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._upper_bounds

    # Integer variables are for the days of the year positions (if optimised)
    cpdef set_integer_variables(self, int[:] values):
        """Set the parameter integer variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.int_]
            The variable to set. The size must equal the number of x coordinates minus 1. Day
            1 is already added by this method in the array of x variables.
        """
        self.days_of_year = [1] + np.array(values).tolist()

    cpdef int[:] get_integer_variables(self):
        """Get the parameter integer variable values during an optimisation problem.

        Returns
        ----------
        numpy.typing.NDArray[np.int_]
            The array with the values to optimised. The size equals the number of x coordinates minus 1.
        """
        return np.array(self.days_of_year[1:], dtype=np.int32)

    cpdef int[:] get_integer_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.int_]
            The array with the lower bounds for each variable. The size equals the number of x coordinates minus 1.
        """
        return self._doy_lower_bounds

    cpdef int[:] get_integer_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.int_]
            The array with the upper bounds for each variable. The size equals the number of x coordinates minus 1.
        """
        return self._doy_upper_bounds

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        RbfProfileParameter
            The loaded class.
        """
        return cls(model, **data)
RbfProfileParameter.register()


cdef class IndexParameter(Parameter):
    """Base parameter providing an `index` method. This is similar to a Parameter but instead
    of returning a float via the `value` method, it returns an index via the `index` method.

    Attributes
    ----------
    model : Model
        The model instance.
    is_variable : bool
        Whether the parameter is set as variable to solve an optimisation problem.
    double_size : int
        The number of double variables in the parameter.
    integer_size : int
        The number of integer variables in the parameter.
    is_constant : bool
        Whether the parameter is constant (i.e. the value does not change with the timestep).
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    See also
    --------
    IndexedArrayParameter
    ControlCurveIndexParameter
    """
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index as a float.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter index as float.
        """
        # return index as a float
        return float(self.get_index(scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The parameter index.
        """
        # return index as an integer
        return 0

    cpdef setup(self):
        """Setup the parameter. This initialises the internal indexes as empty array."""
        super(IndexParameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self._IndexParameter__indices = np.empty([num_comb], np.int32)

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter indexes and values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self._IndexParameter__indices[<int>(scenario_index.global_id)] = self.index(timestep, scenario_index)
            self._Parameter__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef int get_index(self, ScenarioIndex scenario_index):
        """Get the parameter index for a scenario when its index was last updated.
        
        Parameters
        ----------
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter index.
        """
        return self._IndexParameter__indices[<int>(scenario_index.global_id)]

    cpdef int[:] get_all_indices(self):
        """Get all parameter indexes for all scenarios when its indexes were last updated.

        Returns
        -------
        numpy.typing.NDArray[np.int_]
            An array with the indexes. The array size equals the number of scenarios.
        """
        return self._IndexParameter__indices
IndexParameter.register()


cdef class ConstantScenarioIndexParameter(IndexParameter): 
    """A [pywr.core.Scenario][] varying [pywr.parameters.IndexParameter][].

    The indexes in this parameter are constant in time, but vary within a single Scenario. Use this
    parameter if you are using model scenarios and you want to change a constant index based on the
    scenario Pywr is running.

    Attributes
    ----------
    model : Model
        The model instance.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            The scenario the constant parameters are applied to.
        values : Iterable[float | int]
            The constant values to use at each timestep and scenario. This iterable must have the same length as scenario.size.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        kwargs : dict
            Any other keyword argument.
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
        """Setup the parameter."""
        super(ConstantScenarioIndexParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Get the parameter index for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The parameter index.
        """
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
    """A collection of [pywr.parameters.Parameter][]s whose values can be aggregated
    using an aggregating function (e.g. sum).

    This class behaves like a set. Parameters can be added or removed from it.

    Examples
    -------
    Python
    ======
    ```python
    model = Model()
    p1 = ComstantParameter(model=model, value=1.0)
    p2 = MonthlyProfileParameter(model=model, values=range(0, 12))
    AggregatedParameter(model=model, parameters=[p1, p2], agg_func="sum", name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "AggregatedParameter",
            "parameters": [
                {
                    "type": "Constant",
                    "value": 1.0
                },
                {
                    "type": "MonthlyProfile",
                    "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                }
            ],
            "agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    children : Iterable[Parameter]
        The list of parameters to aggregate.
    agg_func : Optional[Callable[[Iterable[float], None]] | Literal["sum", "min", "max", "mean", "product", "any", "all", "median"]], default=None
        The aggregation function. 
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, parameters, agg_func=None, **kwargs):
        """Initialise the class.
        
        Parameters
        ----------
        model : Model
            The model instance.
        parameters : Iterable[Parameter]
            The parameters to aggregate.
        agg_func : Optional[Callable[[Iterable[float], None]] | Literal["sum", "min", "max", "mean", "product", "any", "all", "median"]], default=None
            The aggregation function. Must be one of {"sum", "min", "max", "mean",
            "product", "any", "all", "median"}, or a callable function which accepts a list of
            [pywr.parameters.Parameter][]s.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(AggregatedParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        AggregatedParameter
            The loaded class.
        """
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
        """Add a new parameter to the aggregation.
        
        Parameters
        ----------
        parameter : Parameter
            The parameter instance to add.
        """
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        """Remove a new parameter to the aggregation.
        
        Parameters
        ----------
        parameter : Parameter
            The parameter instance to remove.
        """
        self.parameters.remove(parameter)
        parameter.parents.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        """Setup the parameter."""
        super(AggregatedParameter, self).setup()
        assert(len(self.parameters))

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new aggregated values.
        """
        cdef Parameter parameter
        cdef double[:] accum = self._Parameter__values  # View of the underlying location for the data
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
            accum[...] = -np.inf
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    if values[i] > accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MIN:
            accum[...] = np.inf
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
    """A collection of [pywr.parameters.IndexParameter][]s whose indexes can be aggregated
    using an aggregating function (e.g. sum).

    This class behaves like a set. Parameters can be added or removed from it.

    Attributes
    ----------
    model : Model
        The model instance.
    children : Iterable[IndexParameter]
        The list of parameters to aggregate.
    agg_func : Optional[Callable[[Iterable[IndexParameter], None]] | Literal["sum", "min", "max", "mean", "product", "any", "all", "median"]], default=None
        The aggregation function. 
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(self, model, parameters, agg_func=None, **kwargs):
        """Initialise the class.
        
        Parameters
        ----------
        model : Model
            The model instance.
        parameters : Iterable[IndexParameter]
            The index parameters to aggregate.
        agg_func : Optional[Callable[[Iterable[IndexParameter], None]] | Literal["sum", "min", "max", "mean", "product", "any", "all", "median"]], default=None
            The aggregation function. Must be one of {"sum", "min", "max", "mean",
            "product", "any", "all", "median"}, or a callable function which accepts a list of
            [pywr.parameters.Parameter][]s.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(AggregatedIndexParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        AggregatedIndexParameter
            The loaded class.
        """
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
        """Add a new parameter to the aggregation.
        
        Parameters
        ----------
        parameter : IndexParameter
            The parameter instance to add.
        """
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        """Remove a new parameter to the aggregation.
        
        Parameters
        ----------
        parameter : IndexParameter
            The parameter instance to remove.
        """
        self.parameters.remove(parameter)
        parameter.parents.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        """Setup the parameter."""
        super(AggregatedIndexParameter, self).setup()
        assert len(self.parameters)
        assert all([isinstance(parameter, IndexParameter) for parameter in self.parameters])

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new aggregated values.
        """
        cdef IndexParameter parameter
        cdef int[:] accum = self._IndexParameter__indices  # View of the underlying location for the data
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
            self._Parameter__values[i] = accum[i]


AggregatedIndexParameter.register()


cdef class DivisionParameter(Parameter):
    """ Parameter that divides one [pywr.parameters.Parameter][] by another.

    Examples
    -------
    Python
    ======
    ```python
    model = Model()
    p1 = ComstantParameter(model=model, value=1.0)
    p2 = MonthlyProfileParameter(model=model, values=range(0, 12))
    DivisionParameter(model=model, numerator=p1, denominator=p2, name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "DivisionParameter",
            "numerator":{
                "type": "Constant",
                "value": 1.0
            },
            "denominator":{
                "type": "MonthlyProfile",
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    denominator : Parameter
        The parameter to use as the denominator (or divisor).
    numerator : Parameter
        The parameter to use as the numerator (or dividend).
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, numerator, denominator, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        denominator : Parameter
            The parameter to use as the denominator (or divisor).
        numerator : Parameter
            The parameter to use as the numerator (or dividend).
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
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
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = self._numerator._Parameter__values[i] / self._denominator._Parameter__values[i]

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        DivisionParameter
            The loaded class.
        """
        numerator = load_parameter(model, data.pop("numerator"))
        denominator = load_parameter(model, data.pop("denominator"))
        return cls(model, numerator, denominator, **data)
DivisionParameter.register()


cdef class NegativeParameter(Parameter):
    """ Parameter that takes negative of another [pywr.parameters.Parameter][].

    Examples
    -------
    This returns the nagtive of the current month index.

    Python
    ======
    ```python
    model = Model()
    p = MonthlyProfileParameter(model=model, values=range(0, 12))
    NegativeParameter(model=model, parameter=p, name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "NegativeParameter",
            "parameter":{
                "type": "MonthlyProfile",
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }
        }
    }
    ```
    
    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to negate.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, parameter, *args, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter to negate.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super(NegativeParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = -self.parameter._Parameter__values[i]

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        NegativeParameter
            The loaded class.
        """
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
NegativeParameter.register()


cdef class MaxParameter(Parameter):
    """Parameter that takes maximum between a [pywr.parameters.Parameter][]'s value and constant value (threshold).

    Examples
    -------
    Python
    ======
    ```python
    model = Model()
    p = MonthlyProfileParameter(model=model, values=range(0, 12))
    MaxParameter(model=model, parameter=p1, threshold=5, name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "MaxParameter",
            "threshold": 5,
            "parameter":{
                "type": "MonthlyProfile",
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }
        }
    }
    ```

    Notes
    -----
    This class is a more efficient version of [pywr.parameters.AggregatedParameter][] where
    a single `Parameter` is compared to constant value.

    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to compare with the float.
    threshold : float
        The threshold value to compare with the given parameter.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter to compare with the float.
        threshold : float, default=0.0
            The threshold value to compare with the given parameter.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super(MaxParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = max(self.parameter._Parameter__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        MaxParameter
            The loaded class.
        """
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MaxParameter.register()


cdef class NegativeMaxParameter(MaxParameter):
    """Parameter that takes maximum of the negative of a [pywr.parameters.Parameter][] and constant value (threshold).
    
    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to compare with the float.
    threshold : float
        The threshold value to compare with the given parameter.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    
    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = max(-self.parameter._Parameter__values[i], self.threshold)

NegativeMaxParameter.register()


cdef class MinParameter(Parameter):
    """Parameter that takes minimum of another [pywr.parameters.Parameter][]. and constant value (threshold)

    Examples
    -------
    Python
    ======
    ```python
    model = Model()
    p = MonthlyProfileParameter(model=model, values=range(0, 12))
    MinParameter(model=model, parameter=p1, threshold=5, name="My parameter")
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "MinParameter",
            "threshold": 5,
            "parameter":{
                "type": "MonthlyProfile",
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            }
        }
    }
    ```

    Notes
    -----
    This class is a more efficient version of [pywr.parameters.AggregatedParameter][] where
    a single `Parameter` is compared to constant value.

    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to compare with the float.
    threshold : float
        The threshold value to compare with the given parameter.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter to compare with the float.
        threshold : float, default=0.0
            The threshold value to compare with the given parameter.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super(MinParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = min(self.parameter._Parameter__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        MaxParameter
            The loaded class.
        """
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MinParameter.register()


cdef class NegativeMinParameter(MinParameter):
    """Parameter that takes minimum of the negative of a [pywr.parameters.Parameter][] and constant value (threshold).
 
    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to compare with the float.
    threshold : float
        The threshold value to compare with the given parameter.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = min(-self.parameter._Parameter__values[i], self.threshold)
NegativeMinParameter.register()


cdef class OffsetParameter(Parameter):
    """Parameter that offsets another [pywr.parameters.Parameter][] by a constant value (offset).
    
    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter to compare with the float.
    offset : float
        The offset to apply to the value returned by `parameter`.
    lower_bounds : float
        The lower bounds of the offset when used during optimisation.
    upper_bounds : float
        The upper bounds of the offset when used during optimisation.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    Notes
    -----
    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is offset by a constant value.

    
    """
    def __init__(self, model, parameter, offset=0.0, lower_bounds=0.0, upper_bounds=np.inf, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter to compare with the float.
        offset : Optional[float], default=0.0
            The offset to apply to the value returned by `parameter`.
        lower_bounds : Optional[float], default=0.0
            The lower bounds of the offset when used during optimisation.
        upper_bounds : Optional[float], default=np.inf
            The upper bounds of the offset when used during optimisation.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super(OffsetParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.offset = offset
        self.double_size = 1
        self._lower_bounds = np.ones(self.double_size) * lower_bounds
        self._upper_bounds = np.ones(self.double_size) * upper_bounds

    cdef calc_values(self, Timestep timestep):
        """Calculate the parameter values for all scenarios for the given timestep.
        
        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        
        Returns
        -------
        None
            This only internally stores the new values.
        """
        cdef int i
        cdef int n = self._Parameter__values.shape[0]

        for i in range(n):
            self._Parameter__values[i] = self.parameter._Parameter__values[i] + self.offset

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[np.number]
            The variable to set. The size must equal the number of variables the parameter handles.
        """
        self.offset = values[0]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array([self.offset, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[np.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        OffsetParameter
            The loaded class.
        """
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
        self._Parameter__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        pass # calculation done in after

    cpdef after(self):
        cdef double[:] max_flow
        cdef int i
        if self.node._max_flow_param is None:
            for i in range(0, self.node._flow.shape[0]):
                self._Parameter__values[i] = self.node._max_flow - self.node._flow[i]
        else:
            max_flow = self.node._max_flow_param.get_all_values()
            for i in range(0, self.node._flow.shape[0]):
                self._Parameter__values[i] = max_flow[i] - self.node._flow[i]

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
        self._Parameter__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        cdef int i
        for i in range(self._Parameter__values.shape[0]):
            self._Parameter__values[i] = self.__next_values[i]

    cpdef after(self):
        cdef int i
        for i in range(self.node._flow.shape[0]):
            self.__next_values[i] = self.node._flow[i]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        return cls(model, node=node, **data)
FlowParameter.register()


cdef class StorageParameter(Parameter):
    """Parameter that provides the current volume from a storage node.

    Parameters
    ----------
    model : pywr.model.Model
    storage_node : AbstractStorage
      The node that will have its volume tracked
    use_proportional_volume : bool
        An optional boolean only to switch between returning absolute or proportional volume.

    Notes
    -----
    This parameter returns the current volume of the given storage node. These
    values can be used in calculations for the current timestep as though this was any
    other parameter.
    """
    def __init__(self, model, storage_node, *args, use_proportional_volume=False, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.storage_node = storage_node
        self.use_proportional_volume = use_proportional_volume

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        if self.use_proportional_volume:
            return self.storage_node._current_pc[scenario_index.global_id]
        else:
            return self.storage_node._volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        storage_node = model.nodes[data.pop("storage_node")]
        return cls(model, storage_node=storage_node, **data)
StorageParameter.register()


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


cdef class RollingMeanFlowNodeParameter(Parameter):
    """Returns the mean flow of a Node for the previous N timesteps or days.

    Parameters
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    timesteps : int (optional)
        The number of timesteps to calculate the mean flow for. If `days` is provided then timesteps is ignored.
    days : int (optional)
        The number of days to calculate the mean flow for. This is converted into a number of timesteps
        internally provided the timestep is a number of days.
    name : str (optional)
        The name of the parameter
    initial_flow : float
        The initial value to use in the first timestep before any flows have been recorded.
    """
    def __init__(self, model, node, timesteps=None, days=None, initial_flow=0.0, **kwargs):
        super(RollingMeanFlowNodeParameter, self).__init__(model, **kwargs)
        self.node = node
        self.initial_flow = initial_flow

        if not timesteps and not days:
            raise ValueError("Either `timesteps` or `days` must be specified.")
        if timesteps:
            self.timesteps = int(timesteps)
        else:
            self.timesteps = 0
        if days:
            self.days = int(days)
        else:
            self.days = 0
        self._memory = None
        self.position = 0

    cpdef setup(self):
        super(RollingMeanFlowNodeParameter, self).setup()
        if self.days > 0:
            try:
                self.timesteps = self.days // self.model.timestepper.delta
            except TypeError:
                raise TypeError('A rolling window defined as a number of days is only valid with daily time-steps.')
        if self.timesteps == 0:
            raise ValueError("Timesteps is less than 1.")
        self._memory = np.zeros([len(self.model.scenarios.combinations), self.timesteps])

    cpdef reset(self):
        super(RollingMeanFlowNodeParameter, self).reset()
        self.position = 0
        self._memory[:] = 0.0

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:

        cdef int n
        # No data in memory yet
        if ts.index == 0:
            return self.initial_flow

        # Calculate the mean flow from the memory
        if ts.index <= self.timesteps:
            n = ts.index
        else:
            n = self.timesteps
        return np.mean(self._memory[scenario_index.global_id, :n])

    cpdef after(self):
        cdef int i
        # save today's flow (NB - this won't change the parameter until tomorrow)
        for i in range(0, self._memory.shape[0]):
            self._memory[i, self.position] = self.node._flow[i]

        # prepare for the next timestep
        self.position += 1
        if self.position >= self.timesteps:
            self.position = 0

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        return cls(model, node, **data)

RollingMeanFlowNodeParameter.register()


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
            # Not a parameter, try to load values
            return float(load_parameter_values(model, data))

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

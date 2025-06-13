"""Various activation functions that are useful for modelling binary-like variables in optimisation problems. """
from ._parameters import load_parameter
import numpy as np
cimport numpy as np


cdef class BinaryStepParameter(Parameter):
    """This parameter implements an activation function that outputs a constant value when
    the parameter's internal value is positive (strictly above 0) using:

    ```python
    if internal_value <= 0.0:
        return 0.0
    else:
        return output
    ```

    This allows for a model component's value to change abruptly (for example, to represent
    on/off states or decision points).


    Attributes
    ----------
    model : Model
        The model instance.
    value : float
        The current value of the internal variable.
    output : float
        The value to return when the internal variable is positive.
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
    def __init__(self, model, value=0.0, lower_bounds=-1.0, upper_bounds=1.0, output=1.0, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        value : Optional[float], default=0
            The current value of the internal variable.
        output : Optional[float], default=1
            The value to return when the internal variable is positive.
        lower_bounds : Optional[float], default=1
            The lower bound to use for the value during an optimisation problem.
        upper_bounds : Optional[float], default=1
            The upper bound to use for the value during an optimisation problem.

        Other Parameters
        ----------------
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(BinaryStepParameter, self).__init__(model, **kwargs)
        self._value = value
        self.output = output
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self.is_constant = True

    cpdef double get_constant_value(self):
        """Return the parameter's output based on the internal value.
        
        Returns
        -------
        float
            The value.
        """
        if self._value <= 0.0:
            return 0.0
        else:
            return self.output

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
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The variables to set. The array must have size of 1.
        """
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._upper_bounds

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
        BinaryStepParameter
            The loaded class.
        """
        return cls(model, **data)
BinaryStepParameter.register()


cdef class RectifierParameter(Parameter):
    """This parameter implements an activation function that follows a
    [ramp function](https://en.wikipedia.org/wiki/Ramp_function) if its internal variable is positive using:

    ```python
    if internal_value <= 0.0:
        return 0.0
    else:
        return min_output + (max_output - min_output) * internal_value / upper_bound
    ```

    ![Rump function](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ramp_function.svg/650px-Ramp_function.svg.png)
    
    Attributes
    ----------
    model : Model
        The model instance.
    value : float
        The current value of the internal variable.
    max_output : float
        The maximum value to return when the internal variable is at its upper bounds
    min_output : float
        The value to return when the internal variable is at 0.0.
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
    def __init__(self, model, value=0.0, lower_bounds=-1.0, upper_bounds=1.0, min_output=0.0, max_output=1.0, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        value : Optional[float], default=0
            The current value of the internal variable.
        lower_bounds : Optional[float], default=-1
            The lower bound to use for the value during an optimisation problem.
        upper_bounds : Optional[float], default=1
            The upper bound to use for the value during an optimisation problem.
        max_output : Optional[float], default=1
            The maximum value to return when the internal variable is at its upper bounds
        min_output : Optional[float], default=0
            The value to return when the internal variable is at 0.0.

        Other Parameters
        ----------------
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(RectifierParameter, self).__init__(model, **kwargs)
        self._value = value
        self.min_output = min_output
        self.max_output = max_output
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self.is_constant = True

    cpdef double get_constant_value(self):
        """Return the ramp function value.
        
        Returns
        -------
        float
            The value.
        """
        if self._value <= 0.0:
            return 0.0
        else:
            return self.min_output + (self.max_output - self.min_output) * self._value / self._upper_bounds

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
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The variables to set. The array must have size of 1.
        """
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._upper_bounds

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
        RectifierParameter
            The loaded class.
        """
        return cls(model, **data)
RectifierParameter.register()

cdef class LogisticParameter(Parameter):
    """This parameter implements an activation function using the 
    [logistic function](https://en.wikipedia.org/wiki/Logistic_function) using:

    ```python
    max_output / (1 + np.exp(-growth_rate*internal_value))
    ```

    ![Logistic function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

    Attributes
    ----------
    model : Model
        The model instance.
    value : float
        The current value of the internal variable.
    max_output : float
        The maximum value to return when the logistic function is at its upper bounds.
    growth_rate : float
        The growth rate (or steepness) of the logistic function.
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
    def __init__(self, model, value=0.0, lower_bounds=-6.0, upper_bounds=6.0, max_output=1.0, growth_rate=1.0,
                 **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        value : Optional[float], default=0
            The current value of the internal variable.
        lower_bounds : Optional[float], default=-6
            The lower bound to use for the value during an optimisation problem.
        upper_bounds : Optional[float], default=6
            The upper bound to use for the value during an optimisation problem.
        max_output : Optional[float], default=1
            The maximum value to return when the internal variable is at its upper bounds
        growth_rate : Optional[float], default=1
            The growth rate (or steepness) of the logistic function.

        Other Parameters
        ----------------
        is_variable : bool
            Whether the parameter is set as variable to solve an optimisation problem.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(LogisticParameter, self).__init__(model, **kwargs)
        self._value = value
        self.max_output = max_output
        self.growth_rate = growth_rate
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self.is_constant = True

    cpdef double get_constant_value(self):
        """Return the logistic function value.
        
        Returns
        -------
        float
            The value.
        """
        return self.max_output / (1 + np.exp(-self.growth_rate*self._value))

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
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        """Set the parameter double variable values during an optimisation problem.

        Parameters
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The variables to set. The array must have size of 1.
        """
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        """Get the parameter double variable values for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the variables. The array size equals the number of variables the parameter handles.
        """
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        """Get the lower bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the lower bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        """Get the upper bounds of the double variables for an optimisation problem.

        Returns
        ----------
        values : numpy.typing.NDArray[numpy.number]
            The array with the upper bounds for each variable. The array size equals the number of variables the parameter handles.
        """
        return np.ones(self.double_size) * self._upper_bounds

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
        LogisticParameter
            The loaded class.
        """
        return cls(model, **data)
LogisticParameter.register()

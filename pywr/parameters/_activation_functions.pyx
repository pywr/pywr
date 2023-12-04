"""Various activation functions that are useful for modelling binary-like variables in optimisation problems. """
from ._parameters import load_parameter
import numpy as np
cimport numpy as np


cdef class BinaryStepParameter(Parameter):
    """An activation function that returns a constant value if its internal variable is positive.

    Parameters
    ----------
    value : float
        The current value of the internal variable (default = 0.0)
    lower_bounds, upper_bounds : float
        The valid ranges of the internal variable for optimisation (default = [-1.0, 1.0]).
    output : float
        The value to return when the internal variable is positive (default = 1.0).
    """
    def __init__(self, model, value=0.0, lower_bounds=-1.0, upper_bounds=1.0, output=1.0, **kwargs):
        super(BinaryStepParameter, self).__init__(model, **kwargs)
        self._value = value
        self.output = output
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self.is_constant = True

    cpdef double get_constant_value(self):
        if self._value <= 0.0:
            return 0.0
        else:
            return self.output

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return np.ones(self.double_size) * self._upper_bounds

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
BinaryStepParameter.register()


cdef class RectifierParameter(Parameter):
    """An activation function that follows a ramp function if its internal variable is positive.

    Parameters
    ----------
    value : float
        The current value of the internal variable (default = 0.0)
    lower_bounds, upper_bounds : float
        The valid ranges of the internal variable for optimisation (default = [-1.0, 1.0]).
    max_output : float
        The maximum value to return when the internal variable is at its upper bounds (default = 1.0).
    min_output : float
        The value to return when the internal variable is at 0.0.
    """
    def __init__(self, model, value=0.0, lower_bounds=-1.0, upper_bounds=1.0, min_output=0.0, max_output=1.0, **kwargs):
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
        if self._value <= 0.0:
            return 0.0
        else:
            return self.min_output + (self.max_output - self.min_output) * self._value / self._upper_bounds

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return np.ones(self.double_size) * self._upper_bounds

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
RectifierParameter.register()

cdef class LogisticParameter(Parameter):
    """An activation function that follows a logistic function using its interval value.

    Parameters
    ----------
    value : float
        The current value of the internal variable (default = 0.0)
    lower_bounds, upper_bounds : float
        The valid ranges of the internal variable for optimisation (default = [-1.0, 1.0]).
    max_output : float
        The maximum value to return when the logistic function is at its upper bounds (default = 1.0).
    growth_rate : float
        The growth rate (or steepness) of the logistic function (default = 1.0).
    """
    def __init__(self, model, value=0.0, lower_bounds=-6.0, upper_bounds=6.0, max_output=1.0, growth_rate=1.0,
                 **kwargs):
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
        return self.max_output / (1 + np.exp(-self.growth_rate*self._value))

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is fine because value doesn't depend on timestep or scenario
        return self.get_constant_value()

    cpdef set_double_variables(self, double[:] values):
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return np.ones(self.double_size) * self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return np.ones(self.double_size) * self._upper_bounds

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
LogisticParameter.register()

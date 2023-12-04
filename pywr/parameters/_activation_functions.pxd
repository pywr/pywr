from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, Storage, AbstractStorage
from ._parameters cimport Parameter, IndexParameter


cdef class BinaryStepParameter(Parameter):
    cdef double _value
    cdef public double output
    cdef double _lower_bounds
    cdef double _upper_bounds

cdef class RectifierParameter(Parameter):
    cdef double _value
    cdef public double max_output
    cdef public double min_output
    cdef double _lower_bounds
    cdef double _upper_bounds

cdef class LogisticParameter(Parameter):
    cdef double _value
    cdef public double max_output
    cdef public double growth_rate
    cdef double _lower_bounds
    cdef double _upper_bounds

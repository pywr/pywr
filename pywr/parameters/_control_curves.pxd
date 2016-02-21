from .._core cimport Timestep, Scenario, AbstractNode, Storage
from ._parameters cimport Parameter

cdef class BaseControlCurveParameter(Parameter):
    cdef Storage _storage_node
    cdef Parameter _control_curve

cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    cdef double lower_value
    cdef double curve_value
    cdef double upper_value
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, Storage
from ._parameters cimport Parameter

cdef class BaseControlCurveParameter(Parameter):
    cdef Storage _storage_node
    cdef list _control_curves

cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    cdef double[:] _values

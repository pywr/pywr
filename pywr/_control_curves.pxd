from _core cimport Timestep, Scenario, AbstractNode
from _parameters cimport Parameter

cdef class BaseParameterControlCurve(Parameter):
    cdef Parameter _control_curve

cdef class ParameterControlCurveInterpolated(BaseParameterControlCurve):
    cdef double[:] _interp_values
    cdef double[:] values
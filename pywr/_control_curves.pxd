from _core cimport Timestep, Scenario, AbstractNode
from _parameters cimport Parameter

cdef class BaseParameterControlCurve(Parameter):
    cdef Parameter _control_curve

cdef class ParameterControlCurveInterpolated(BaseParameterControlCurve):
    cdef double lower_value
    cdef double curve_value
    cdef double upper_value
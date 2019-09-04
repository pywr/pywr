from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, Storage, AbstractStorage
from ._parameters cimport Parameter, IndexParameter

cdef class PiecewiseLinearControlCurve(Parameter):
    cdef public AbstractStorage storage_node
    cdef Parameter _control_curve
    cdef public double below_lower
    cdef public double below_upper
    cdef public double above_lower
    cdef public double above_upper
    cdef public double minimum
    cdef public double maximum

cpdef double _interpolate(double current_position, double lower_bound, double upper_bound, double lower_value, double upper_value)

cdef class BaseControlCurveParameter(Parameter):
    cdef AbstractStorage _storage_node
    cdef list _control_curves

cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    cdef double[:] _values
    cdef public list parameters

cdef class ControlCurveIndexParameter(IndexParameter):
    cdef public AbstractStorage storage_node
    cdef list _control_curves


cdef class ControlCurveParameter(BaseControlCurveParameter):
    cdef double[:] _values
    cdef public list parameters
    cdef int[:] _variable_indices
    cdef double[:] _upper_bounds
    cdef double[:] _lower_bounds

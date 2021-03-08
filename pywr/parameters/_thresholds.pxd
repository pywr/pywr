from ._parameters cimport IndexParameter, Parameter
from pywr.recorders._recorders cimport Recorder
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, AbstractStorage
cimport numpy as np
ctypedef np.uint8_t uint8


cdef class AbstractThresholdParameter(IndexParameter):
    cdef public double _threshold
    cdef public Parameter _threshold_parameter
    cdef double[:] values
    cdef int predicate
    cdef public bint ratchet
    cdef uint8[:] _triggered
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class StorageThresholdParameter(AbstractThresholdParameter):
    cdef public AbstractStorage storage

cdef class NodeThresholdParameter(AbstractThresholdParameter):
    cdef public AbstractNode node

cdef class MultipleThresholdIndexParameter(IndexParameter):
    cdef public AbstractNode node
    cdef list thresholds

cdef class MultipleThresholdParameterIndexParameter(IndexParameter):
    cdef public Parameter parameter
    cdef list thresholds


cdef class ParameterThresholdParameter(AbstractThresholdParameter):
    cdef public Parameter param

cdef class RecorderThresholdParameter(AbstractThresholdParameter):
    cdef public Recorder recorder
    cdef public initial_value

cdef class CurrentYearThresholdParameter(AbstractThresholdParameter):
    pass

cdef class CurrentOrdinalDayThresholdParameter(AbstractThresholdParameter):
    pass

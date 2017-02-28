from ._parameters cimport IndexParameter
from pywr.recorders._recorders cimport Recorder
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, AbstractStorage

cdef class AbstractThresholdParameter(IndexParameter):
    cdef public double threshold
    cdef double[:] values
    cdef int predicate
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class StorageThresholdParameter(AbstractThresholdParameter):
    cdef AbstractStorage storage

cdef class NodeThresholdParameter(AbstractThresholdParameter):
    cdef AbstractNode node


cdef class RecorderThresholdParameter(AbstractThresholdParameter):
    cdef Recorder recorder

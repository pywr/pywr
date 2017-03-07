from pywr.parameters._parameters cimport IndexParameter, Parameter
from pywr.recorders._recorders cimport Recorder, StorageRecorder, NodeRecorder
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, AbstractStorage, Storage

cdef class StorageThresholdRecorder(StorageRecorder):
    cdef public double threshold
    cdef int predicate
    cdef bint[:] _state

cdef class NodeThresholdRecorder(NodeRecorder):
    cdef public double threshold
    cdef int predicate
    cdef bint[:] _state

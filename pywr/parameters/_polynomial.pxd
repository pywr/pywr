from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, Storage, AbstractStorage
from ._parameters cimport Parameter, IndexParameter


cdef class Polynomial1DParameter(Parameter):
    cdef public double[:] coefficients
    cdef AbstractNode _other_node
    cdef AbstractStorage _storage_node
    cdef public bint use_proportional_volume
    cdef Parameter _parameter



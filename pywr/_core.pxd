from .parameters._parameters cimport Parameter
from _recorders cimport Recorder

cdef class Scenario:
    cdef basestring _name
    cdef int _size
    cdef public slice slice

cdef class ScenarioCollection:
    cdef public object model
    cdef list _scenarios
    cdef public list combinations
    cpdef int get_scenario_index(self, Scenario sc) except? -1
    cpdef add_scenario(self, Scenario sc)
    cpdef int ravel_indices(self, int[:] scenario_indices) except? -1

cdef class ScenarioCombinations:
    cdef ScenarioCollection _collection

cdef class ScenarioIndex:
    cdef int _global_id
    cdef int[:] _indices

cdef class Timestep:
    cdef object _datetime
    cdef int _index
    cdef double _days
    cdef readonly int dayofyear
    cdef readonly int month

cdef class Domain:
    cdef object name

cdef class AbstractNode:
    cdef double[:] _prev_flow
    cdef double[:] _flow
    cdef list _recorders
    cdef Domain _domain
    cdef AbstractNode _parent
    cdef object _model
    cdef object _name
    cdef bint _allow_isolated
    cdef public bint virtual
    cdef public object __data
    cdef public basestring comment

    cdef Parameter _cost_param

    cpdef setup(self, model)
    cpdef reset(self)
    cpdef before(self, Timestep ts)
    cpdef commit(self, int scenario_index, double value)
    cpdef commit_all(self, double[:] value)
    cpdef after(self, Timestep ts)
    cpdef finish(self)
    cpdef check(self,)

cdef class Node(AbstractNode):
    cdef double _cost
    cdef double _min_flow
    cdef double _max_flow
    cdef double _conversion_factor
    cdef object _min_flow_param
    cdef Parameter _max_flow_param

    cdef Parameter _conversion_factor_param
    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_min_flow(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_max_flow(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_conversion_factor(self) except? -1
    cdef set_parameters(self, Timestep ts, ScenarioIndex scenario_index)

cdef class AggregatedNode(AbstractNode):
    cdef list _nodes
    cdef double[:] _factors
    cdef double _max_flow
    cdef double _min_flow

cdef class BaseInput(Node):
    cdef object _licenses

cdef class AbstractStorage(AbstractNode):
    cdef public double[:] _volume
    cdef public double[:] _current_pc

cdef class Storage(AbstractStorage):
    cdef double _cost
    cdef double _initial_volume
    cdef double _min_volume
    cdef double _max_volume
    cdef double _level
    cdef Parameter _min_volume_param
    cdef Parameter _max_volume_param
    cdef Parameter _level_param
    cpdef _reset_storage_only(self)
    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_min_volume(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_max_volume(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef double get_level(self, Timestep ts, ScenarioIndex scenario_index) except? -1

cdef class AggregatedStorage(AbstractStorage):
    cdef list _storage_nodes

cdef class VirtualStorage(Storage):
    cdef list _nodes
    cdef double[:] _factors

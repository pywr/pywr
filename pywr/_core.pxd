from .parameters._parameters cimport Parameter
from _recorders cimport Recorder

cdef class Scenario:
    cdef str _name
    cdef int _size

cdef class ScenarioCollection:
    cdef list _scenarios
    cdef public ScenarioCombinations combinations
    cpdef get_scenario_index(self, Scenario sc)
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

    cdef Parameter _cost_param

    cpdef setup(self, model)
    cpdef reset(self)
    cpdef before(self, Timestep ts)
    cpdef commit(self, int scenario_index, double value)
    cpdef commit_all(self, double[:] value)
    cpdef after(self, Timestep ts)
    cpdef check(self,)

cdef class Node(AbstractNode):
    cdef double _cost
    cdef double _min_flow
    cdef double _max_flow
    cdef double _conversion_factor
    cdef object _min_flow_param
    cdef Parameter _max_flow_param

    cdef Parameter _conversion_factor_param
    cpdef get_cost(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_min_flow(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_max_flow(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_conversion_factor(self)
    cdef set_parameters(self, Timestep ts, ScenarioIndex scenario_index)

cdef class AggregatedNode(AbstractNode):
    cdef list _nodes
    # TODO Add min/max flow parameters, and add to solvers.

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
    cpdef get_cost(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_min_volume(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_max_volume(self, Timestep ts, ScenarioIndex scenario_index)
    cpdef get_level(self, Timestep ts, ScenarioIndex scenario_index)

cdef class AggregatedStorage(AbstractStorage):
    cdef list _storage_nodes
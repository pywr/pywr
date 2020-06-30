from .parameters._parameters cimport Parameter


cdef class Scenario:
    cdef basestring _name
    cdef int _size
    cdef public slice slice
    cdef object _ensemble_names

cdef class ScenarioCollection:
    cdef public object model
    cdef list _scenarios
    cdef readonly list combinations
    cdef int[:, :] _user_combinations
    cpdef int get_scenario_index(self, Scenario sc) except? -1
    cpdef add_scenario(self, Scenario sc)
    cpdef int ravel_indices(self, int[:] scenario_indices) except? -1

cdef class ScenarioCombinations:
    cdef ScenarioCollection _collection

cdef class ScenarioIndex:
    cdef readonly int global_id
    cdef int[:] _indices

cdef bint is_leap_year(int year)

cdef class Timestep:
    cdef readonly object period
    cdef readonly int index
    cdef readonly double days
    cdef readonly int dayofyear
    cdef readonly int dayofyear_index  # Day of the year for profiles
    cdef readonly bint is_leap_year
    cdef readonly int week_index  # Zero-based week
    cdef readonly int day
    cdef readonly int month
    cdef readonly int year

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
    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1

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
    cdef Parameter _min_flow_param
    cdef Parameter _max_flow_param

    cdef Parameter _conversion_factor_param
    cpdef double get_min_flow(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_max_flow(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_conversion_factor(self) except? -1
    cdef set_parameters(self, ScenarioIndex scenario_index)

cdef class AggregatedNode(AbstractNode):
    cdef list _nodes
    cdef double[:] _factors
    cdef double[:] _flow_weights
    cdef double _max_flow
    cdef double _min_flow
    cdef Parameter _min_flow_param
    cdef Parameter _max_flow_param

    cpdef double get_min_flow(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_max_flow(self, ScenarioIndex scenario_index) except? -1

cdef class BaseInput(Node):
    cdef object _licenses

cdef class AbstractStorage(AbstractNode):
    cdef public double[:] _volume
    cdef public double[:] _current_pc

cdef class Storage(AbstractStorage):
    cdef double _cost
    cdef double _initial_volume
    cdef double _initial_volume_pc
    cdef double _min_volume
    cdef double _max_volume
    cdef double _level
    cdef double _area
    cdef Parameter _min_volume_param
    cdef Parameter _max_volume_param
    cdef Parameter _level_param
    cdef Parameter _area_param
    cpdef _reset_storage_only(self, bint use_initial_volume = *)
    cpdef double get_min_volume(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_max_volume(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_level(self, ScenarioIndex scenario_index) except? -1
    cpdef double get_area(self, ScenarioIndex scenario_index) except? -1

cdef class AggregatedStorage(AbstractStorage):
    cdef list _storage_nodes

cdef class VirtualStorage(Storage):
    cdef list _nodes
    cdef double[:] _factors

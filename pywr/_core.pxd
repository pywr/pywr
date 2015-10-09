cdef class Scenario:
    cdef str _name
    cdef int _size

cdef class ScenarioCollection:
    cdef list _scenarios
    cdef public ScenarioCombinations combinations
    cpdef get_scenario_index(self, Scenario sc)
    cpdef add_scenario(self, Scenario sc)

cdef class ScenarioCombinations:
    cdef ScenarioCollection _collection

cdef class Timestep:
    cdef object _datetime
    cdef int _index
    cdef double _days

cdef class Parameter:
    cpdef setup(self, model)
    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1

cdef class ParameterArrayIndexed(Parameter):
    cdef double[:] values

cdef class ParameterConstantScenario(Parameter):
    cdef Scenario _scenario
    cdef double[:] _values
    cdef int _scenario_index

cdef class ParameterArrayIndexedScenarioMonthlyFactors(Parameter):
    cdef double[:] _values
    cdef double[:, :] _factors
    cdef Scenario _scenario
    cdef int _scenario_index

cdef class Recorder:
    cpdef setup(self, model)
    cpdef reset(self)
    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1
    cpdef int commit_all(self, Timestep ts, double[:] value) except -1

cdef class Domain:
    cdef object name

cdef class AbstractNode:
    cdef double[:] _prev_flow
    cdef double[:] _flow
    cdef double _cost
    cdef Recorder _recorder
    cdef Domain _domain
    cdef AbstractNode _parent
    cdef object _model
    cdef object _name

    cdef Parameter _cost_param
    cpdef get_cost(self, Timestep ts, int[:] scenario_indices=*)

    cpdef setup(self, model)
    cpdef reset(self)
    cpdef before(self, Timestep ts)
    cpdef commit(self, int scenario_index, double value)
    cpdef commit_all(self, double[:] value)
    cpdef after(self, Timestep ts)
    cpdef check(self,)

cdef class Node(AbstractNode):
    cdef double _min_flow
    cdef double _max_flow
    cdef double _conversion_factor
    cdef object _min_flow_param
    cdef Parameter _max_flow_param

    cdef Parameter _conversion_factor_param

    cpdef get_min_flow(self, Timestep ts, int[:] scenario_indices=*)
    cpdef get_max_flow(self, Timestep ts, int[:] scenario_indices=*)
    cpdef get_conversion_factor(self)
    cdef set_parameters(self, Timestep ts, int[:] scenario_indices=*)

cdef class Storage(AbstractNode):
    cdef public double[:] _volume
    cdef double _initial_volume
    cdef double _min_volume
    cdef double _max_volume
    cdef Parameter _min_volume_param
    cdef Parameter _max_volume_param

    cpdef get_min_volume(self, Timestep ts, int[:] scenario_indices=*)
    cpdef get_max_volume(self, Timestep ts, int[:] scenario_indices=*)

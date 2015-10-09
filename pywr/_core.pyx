# cython: profile=False
from pywr._core cimport *
import itertools
import numpy as np
cimport numpy as np

cdef double inf = float('inf')

def product(sizes):
    """ Wrapper for itertools.product to return a np.array """
    cdef int s
    for comb in itertools.product(*[range(s) for s in sizes]):
        yield np.array(comb, dtype=np.int32)

# TODO can this be a cdef class?
cdef class ScenarioCombinations:
    def __init__(self, ScenarioCollection collection):
        self._collection = collection

    def __iter__(self, ):
        cdef Scenario sc
        return product([sc._size for sc in self._collection._scenarios])

    def __len__(self, ):
        cdef Scenario sc
        if len(self._collection._scenarios) > 0:
            return np.prod([sc._size for sc in self._collection._scenarios])
        return 1


cdef class Scenario:
    def __init__(self, str name, int size):
        self._name = name
        self._size = size


cdef class ScenarioCollection:
    def __init__(self, ):
        self._scenarios = []
        self.combinations = ScenarioCombinations(self)

    cpdef get_scenario_index(self, Scenario sc):
        """Return the index of Scenario in this controller."""
        return self._scenarios.index(sc)

    cpdef add_scenario(self, Scenario sc):
        if sc in self._scenarios:
            raise ValueError("The same scenario can not be added twice.")
        self._scenarios.append(sc)

    def __len__(self):
        return len(self._scenarios)


cdef class Timestep:
    def __init__(self, object datetime, int index, double days):
        self._datetime = datetime
        self._index = index
        self._days = days

    property datetime:
        """Timestep representation as a `datetime.datetime` object"""
        def __get__(self, ):
            return self._datetime

    property index:
        """The index of the timestep for use in arrays"""
        def __get__(self, ):
            return self._index

    property days:
        def __get__(self, ):
            return self._days

cdef class Recorder:
    cpdef setup(self, model):
        pass

    cpdef reset(self):
        pass

    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1:
        return 0

    cpdef int commit_all(self, Timestep ts, double[:] value) except -1:
        return 0


cdef class NumpyArrayRecorder(Recorder):
    cdef double[:, :] _data
    cpdef setup(self, model):
        cdef int ncomb = len(model.scenarios.combinations)
        cdef int nts = len(model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1:
        self._data[ts._index, scenario_index] = value

    cpdef int commit_all(self, Timestep ts, double[:] value) except -1:
        cdef int i
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = value[i]

    property data:
        def __get__(self, ):
            return np.array(self._data)


cdef class Parameter:
    cpdef setup(self, model):
        pass
    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        return 0

cdef class ParameterArrayIndexed(Parameter):
    """Time varying parameter using an array and Timestep._index
    """
    def __cinit__(self, double[:] values):
        self.values = values

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts._index]

cdef class ParameterConstantScenario(Parameter):
    """A Scenario varying Parameter"""
    def __init__(self, Scenario scenario, values):
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_indices[self._scenario_index]]


cdef class ParameterArrayIndexedScenarioMonthlyFactors(Parameter):
    """Time varying parameter using an array and Timestep._index with
    multiplicative factors per Scenario
    """
    def __init__(self, Scenario scenario, double[:] values, double[:, :] factors):
        if scenario._size != factors.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if factors.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = values
        self._factors = factors

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.datetime.month-1
        return self._values[ts._index]*self._factors[scenario_indices[self._scenario_index], imth]

cdef class Domain:
    """ Domain class which all Node objects must have. """
    def __init__(self, name):
        self.name = name

cdef class AbstractNode:
    def __init__(self, model, name, **kwargs):
        self._model = model
        model.graph.add_node(self)
        model.dirty = True
        self.name = name

        self._parent = kwargs.pop('parent', None)
        self._domain = kwargs.pop('domain', None)
        # TODO fix this default hack
        self._recorder = kwargs.pop('recorder', NumpyArrayRecorder(len(model.timestepper)))

    property cost:
        """The cost per unit flow via the node

        The cost may be set to either a constant (i.e. a float) or a Parameter.

        The value returned can be positive (i.e. a cost), negative (i.e. a
        benefit) or netural. Typically supply nodes will have an associated
        cost and demands will provide a benefit.
        """
        def __set__(self, value):
            if isinstance(value, Parameter):
                self._cost_param = value
            else:
                self._cost_param = None
                self._cost = value

    cpdef get_cost(self, Timestep ts, int[:] scenario_indices=None):
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts, scenario_indices)

    property name:
        def __get__(self):
            return self._name

        def __set__(self, name):
            # check for name collision
            if name in self.model.node:
                raise ValueError('A node with the name "{}" already exists.'.format(name))
            # remove old name
            try:
                del(self.model.node[self._name])
            except KeyError:
                pass
            # apply new name
            self._name = name
            self.model.node[name] = self

    property model:
        """The recorder for the node, e.g. a NumpyArrayRecorder
        """
        def __get__(self):
            return self._model

        def __set__(self, value):
            self._model = value

    property recorder:
        """The recorder for the node, e.g. a NumpyArrayRecorder
        """
        def __get__(self):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    property domain:
        def __get__(self):
            if self._domain is None and self._parent is not None:
                return self._parent._domain
            return self._domain

        def __set__(self, value):
            if self._parent is not None:
                import warnings
                warnings.warn("Setting domain property of node with a parent.")
            self._domain = value

    property parent:
        """The parent Node/Storage of this object.
        """
        def __get__(self):
            return self._parent

        def __set__(self, value):
            self._parent = value


    cpdef setup(self, model):
        """Called before the first run of the model"""
        cdef int ncomb = len(model.scenarios.combinations)
        self._flow = np.empty(ncomb, dtype=np.float64)
        if self._recorder is not None:
            self._recorder.setup(model)
        if self._cost_param is not None:
            self._cost_param.setup(model)
        if self._recorder is not None:
            self._recorder.setup(model)

    cpdef reset(self):
        """Called at the beginning of a run"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0
        if self._recorder is not None:
            self._recorder.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

    cpdef commit(self, int scenario_index, double value):
        """Called once for each route the node is a member of"""
        self._flow[scenario_index] += value

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]

    cpdef after(self, Timestep ts):
        pass

    cpdef check(self,):
        pass

cdef class Node(AbstractNode):
    """Node class from which all others inherit
    """
    def __cinit__(self):
        """Initialise the node attributes
        """
        # Initialised attributes to zero
        self._min_flow = 0.0
        self._max_flow = inf
        self._cost = 0.0
        # Conversion is default to unity so that there is no loss
        self._conversion_factor = 1.0
        # Parameters are initialised to None which corresponds to
        # a static value
        self._min_flow_param = None
        self._max_flow_param = None
        self._cost_param = None
        self._conversion_factor_param = None
        self._recorder = None
        self._domain = None

    property prev_flow:
        """Total flow via this node in the previous timestep
        """
        def __get__(self):
            return np.array(self._prev_flow)

    property flow:
        """Total flow via this node in the current timestep
        """
        def __get__(self):
            return np.array(self._flow)

    property min_flow:
        """The minimum flow constraint on the node

        The minimum flow may be set to either a constant (i.e. a float) or a
        Parameter.
        """
        def __get__(self):
            return self._min_flow_param

        def __set__(self, value):
            if isinstance(value, Parameter):
                self._min_flow_param = value
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef get_min_flow(self, Timestep ts, int[:] scenario_indices=None):
        """Get the minimum flow at a given timestep
        """
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.value(ts, scenario_indices)

    property max_flow:
        """The maximum flow constraint on the node

        The maximum flow may be set to either a constant (i.e. a float) or a
        Parameter.
        """
        def __set__(self, value):
            if value is None:
                self._max_flow = inf
            elif isinstance(value, Parameter):
                self._max_flow_param = value
            else:
                self._max_flow_param = None
                self._max_flow = value

    cpdef get_max_flow(self, Timestep ts, int[:] scenario_indices=None):
        """Get the maximum flow at a given timestep
        """
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.value(ts, scenario_indices)

    property conversion_factor:
        """The conversion between inflow and outflow for the node

        The conversion factor may be set to either a constant (i.e. a float) or
        a Parameter.
        """
        def __set__(self, value):
            self._conversion_factor_param = None
            if isinstance(value, Parameter):
                raise ValueError("Conversion factor can not be a Parameter.")
            else:
                self._conversion_factor = value

    cpdef get_conversion_factor(self):
        """Get the conversion factor

        Note: the conversion factor must be a constant.
        """
        return self._conversion_factor

    cdef set_parameters(self, Timestep ts, int[:] scenario_indices=None):
        """Update the constant attributes by evaluating any Parameter objects

        This is useful when the `get_` functions need to be accessed multiple
        times and there is a benefit to caching the values.
        """
        if self._min_flow_param is not None:
            self._min_flow = self._min_flow_param.value(ts, scenario_indices)
        if self._max_flow_param is not None:
            self._max_flow = self._max_flow_param.value(ts, scenario_indices)
        if self._cost_param is not None:
            self._cost = self._cost_param.value(ts, scenario_indices)

    cpdef setup(self, model):
        """Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._flow = np.empty(ncomb, dtype=np.float64)
        self._prev_flow = np.empty(ncomb, dtype=np.float64)
        # Setup any Parameters and Recorders
        if self._min_flow_param is not None:
            self._min_flow_param.setup(model)
        if self._max_flow_param is not None:
            self._max_flow_param.setup(model)
        if self._cost_param is not None:
            self._cost_param.setup(model)
        if self._recorder is not None:
            self._recorder.setup(model)

    cpdef after(self, Timestep ts):
        """Called at the end of the timestep"""
        self._prev_flow[:] = self._flow[:]
        if self._recorder is not None:
            self._recorder.commit_all(ts, self._flow)

cdef class Storage(AbstractNode):
    def __cinit__(self, ):
        self._initial_volume = 0.0
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._cost_param = None
        self._recorder = None
        self._domain = None

    property volume:
        def __get__(self, ):
            return self._volume[0]

        def __set__(self, value):
            # This actually sets the initial volume
            # TODO provide a method to set the _volume for a given scenario(s).
            import warnings
            warnings.warn("Setting volume property directly is not supported. This method is updating the initial storage volume.")
            self._initial_volume = value

    property initial_volume:
        def __get__(self, ):
            return self._initial_volume

        def __set__(self, value):
            self._initial_volume

    property min_volume:
        def __set__(self, value):
            self._min_volume_param = None
            if isinstance(value, Parameter):
                self._min_volume_param = value
            else:
                self._min_volume = value

    cpdef get_min_volume(self, Timestep ts, int[:] scenario_indices=None):
        if self._min_volume_param is None:
            return self._min_volume
        return self._min_volume_param.value(ts, scenario_indices)

    property max_volume:
        def __set__(self, value):
            self._max_volume_param = None
            if isinstance(value, Parameter):
                self._max_volume_param = value
            else:
                self._max_volume = value

    cpdef get_max_volume(self, Timestep ts, int[:] scenario_indices=None):
        if self._max_volume_param is None:
            return self._max_volume
        return self._max_volume_param.value(ts, scenario_indices)

    property current_pc:
        " Current percentage full "
        def __get__(self, ):
            return np.array(self._volume[:]) / self._max_volume

    property recorder:
        def __get__(self, ):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    property domain:
        def __get__(self):
            return self._domain

        def __set__(self, value):
            self._domain = value

    cpdef setup(self, model):
        """Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._volume = np.empty(ncomb, dtype=np.float64)

    cpdef reset(self):
        """Called at the beginning of a run"""
        AbstractNode.reset(self)
        cdef int i
        for i in range(self._volume.shape[0]):
            self._volume[i] = self._initial_volume

    cpdef after(self, Timestep ts):
        AbstractNode.after(self, ts)
        cdef int i
        for i in range(self._flow.shape[0]):
            self._volume[i] += self._flow[i]*ts._days
        if self._recorder is not None:
            self._recorder.commit_all(ts, self._volume)

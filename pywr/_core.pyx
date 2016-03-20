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


cdef class ScenarioCombinations:
    def __init__(self, ScenarioCollection collection):
        self._collection = collection

    def __iter__(self, ):
        cdef Scenario sc
        cdef int i
        cdef int[:] indices
        for i, indices in enumerate(product([sc._size for sc in self._collection._scenarios])):
            yield ScenarioIndex(i, indices)

    def __len__(self, ):
        cdef Scenario sc
        if len(self._collection._scenarios) > 0:
            return np.prod([sc._size for sc in self._collection._scenarios])
        return 1


cdef class Scenario:
    def __init__(self, str name, int size):
        self._name = name
        self._size = size

    property size:
        def __get__(self, ):
            return self._size

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

    property shape:
        def __get__(self):
            if len(self._scenarios) == 0:
                return (1, )
            return tuple(sc.size for sc in self._scenarios)

    cpdef int ravel_indices(self, int[:] scenario_indices) except? -1:
        if scenario_indices is None:
            return 0
        # Case where scenario_indices is empty for no scenarios defined
        if scenario_indices.size == 0:
            return 0
        return np.ravel_multi_index(scenario_indices, np.array(self.shape))

cdef class ScenarioIndex:
    def __init__(self, int global_id, int[:] indices):
        self._global_id = global_id
        self._indices = indices

    property global_id:
        def __get__(self):
            return self._global_id

    property indices:
        def __get__(self):
            return np.array(self._indices)


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

cdef class Domain:
    """ Domain class which all Node objects must have. """
    def __init__(self, name):
        self.name = name

cdef class AbstractNode:
    def __cinit__(self):
        self._allow_isolated = False

    def __init__(self, model, name, **kwargs):
        self._model = model
        self.name = name

        self._parent = kwargs.pop('parent', None)
        self._domain = kwargs.pop('domain', None)
        self._recorders = []

        self._flow = np.empty([0,], np.float64)

        # there shouldn't be any unhandled keyword arguments by this point
        if kwargs:
            raise TypeError("__init__() got an unexpected keyword argument '{}'".format(list(kwargs.items())[0]))

    property allow_isolated:
        def __get__(self):
            return self._allow_isolated
        def __set__(self, value):
            self._allow_isolated = value

    property cost:
        """The cost per unit flow via the node

        The cost may be set to either a constant (i.e. a float) or a Parameter.

        The value returned can be positive (i.e. a cost), negative (i.e. a
        benefit) or netural. Typically supply nodes will have an associated
        cost and demands will provide a benefit.
        """
        def __get__(self):
            if self._cost_param is None:
                return self._cost
            return self._cost_param

        def __set__(self, value):
            if isinstance(value, Parameter):
                self._cost_param = value
                value.node = self
            else:
                self._cost_param = None
                self._cost = value

    cpdef get_cost(self, Timestep ts, ScenarioIndex scenario_index):
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts, scenario_index)

    property name:
        def __get__(self):
            return self._name

        def __set__(self, name):
            # check for name collision
            if name in self.model.nodes.keys():
                raise ValueError('A node with the name "{}" already exists.'.format(name))
            # apply new name
            self._name = name

    property recorders:
        def __get__(self):
            return self._recorders

    property model:
        """The recorder for the node, e.g. a NumpyArrayRecorder
        """
        def __get__(self):
            return self._model

        def __set__(self, value):
            self._model = value

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
        if self._cost_param is not None:
            self._cost_param.setup(model)

    cpdef reset(self):
        """Called at the beginning of a run"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

        if self._cost_param is not None:
            self._cost_param.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

        if self._cost_param is not None:
            self._cost_param.before(ts)

    cpdef commit(self, int scenario_index, double value):
        """Called once for each route the node is a member of"""
        self._flow[scenario_index] += value

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]

    cpdef after(self, Timestep ts):
        if self._cost_param is not None:
            self._cost_param.after(ts)

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
            if self._min_flow_param is None:
                return self._min_flow
            return self._min_flow_param

        def __set__(self, value):
            if isinstance(value, Parameter):
                self._min_flow_param = value
                value.node = self
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef get_min_flow(self, Timestep ts, ScenarioIndex scenario_index):
        """Get the minimum flow at a given timestep
        """
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.value(ts, scenario_index)

    property max_flow:
        """The maximum flow constraint on the node

        The maximum flow may be set to either a constant (i.e. a float) or a
        Parameter.
        """
        def __get__(self):
            if self._max_flow_param is None:
                return self._max_flow
            return self._max_flow_param

        def __set__(self, value):
            if value is None:
                self._max_flow = inf
            elif isinstance(value, Parameter):
                self._max_flow_param = value
                value.node = self
            else:
                self._max_flow_param = None
                self._max_flow = value

    cpdef get_max_flow(self, Timestep ts, ScenarioIndex scenario_index):
        """Get the maximum flow at a given timestep
        """
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.value(ts, scenario_index)

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

    property variables:
        """ Returns a list of any set Parameters with is_variable == True
        """
        def __get__(self):
            variables = []
            if self._cost_param is not None:
                if self._cost_param.is_variable:
                    variables.append(self._cost_param)
            if self._min_flow_param is not None:
                if self._min_flow_param.is_variable:
                    variables.append(self._min_flow_param)
            if self._max_flow_param is not None:
                if self._max_flow_param.is_variable:
                    variables.append(self._max_flow_param)
            return variables

    cpdef get_conversion_factor(self):
        """Get the conversion factor

        Note: the conversion factor must be a constant.
        """
        return self._conversion_factor

    cdef set_parameters(self, Timestep ts, ScenarioIndex scenario_index):
        """Update the constant attributes by evaluating any Parameter objects

        This is useful when the `get_` functions need to be accessed multiple
        times and there is a benefit to caching the values.
        """
        if self._min_flow_param is not None:
            self._min_flow = self._min_flow_param.value(ts, scenario_index)
        if self._max_flow_param is not None:
            self._max_flow = self._max_flow_param.value(ts, scenario_index)
        if self._cost_param is not None:
            self._cost = self._cost_param.value(ts, scenario_index)

    cpdef setup(self, model):
        """Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._flow = np.empty(ncomb, dtype=np.float64)
        self._prev_flow = np.zeros(ncomb, dtype=np.float64)
        # Setup any Parameters and Recorders
        if self._min_flow_param is not None:
            self._min_flow_param.setup(model)
        if self._max_flow_param is not None:
            self._max_flow_param.setup(model)

    cpdef reset(self):
        # Reset any Parameters and Recorders
        if self._min_flow_param is not None:
            self._min_flow_param.reset()
        if self._max_flow_param is not None:
            self._max_flow_param.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        AbstractNode.before(self, ts)

        # Complete any parameter calculations
        if self._max_flow_param is not None:
            self._max_flow_param.before(ts)
        if self._min_flow_param is not None:
            self._min_flow_param.before(ts)

    cpdef after(self, Timestep ts):
        """Called at the end of the timestep"""
        AbstractNode.after(self, ts)
        self._prev_flow[:] = self._flow[:]
        # Complete any parameter calculations
        if self._max_flow_param is not None:
            self._max_flow_param.after(ts)
        if self._min_flow_param is not None:
            self._min_flow_param.after(ts)

cdef class BaseLink(Node):
    pass


cdef class BaseInput(Node):
    pass


cdef class BaseOutput(Node):
    pass


cdef class StorageInput(BaseInput):
    cpdef commit(self, int scenario_index, double volume):
        BaseInput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, -volume)

    cpdef get_cost(self, Timestep ts, ScenarioIndex scenario_index):
        # Return negative of parent cost
        return -self.parent.get_cost(ts, scenario_index)

cdef class StorageOutput(BaseOutput):
    cpdef commit(self, int scenario_index, double volume):
        BaseOutput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, volume)

    cpdef get_cost(self, Timestep ts, ScenarioIndex scenario_index):
        # Return parent cost
        return self.parent.get_cost(ts, scenario_index)

cdef class Storage(AbstractNode):
    def __cinit__(self, ):
        self._initial_volume = 0.0
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._cost_param = None
        self._domain = None
        self._allow_isolated = True

    property volume:
        def __get__(self, ):
            return np.asarray(self._volume)

    property initial_volume:
        def __get__(self, ):
            return self._initial_volume

        def __set__(self, value):
            self._initial_volume = value

    property min_volume:
        def __get__(self):
            if self._min_volume_param is None:
                return self._min_volume
            return self._min_volume_param

        def __set__(self, value):
            self._min_volume_param = None
            if isinstance(value, Parameter):
                self._min_volume_param = value
                value.node = self
            else:
                self._min_volume = value

    cpdef get_min_volume(self, Timestep ts, ScenarioIndex scenario_index):
        if self._min_volume_param is None:
            return self._min_volume
        return self._min_volume_param.value(ts, scenario_index)

    property max_volume:
        def __get__(self):
            if self._max_volume_param is None:
                return self._max_volume
            return self._max_volume_param

        def __set__(self, value):
            self._max_volume_param = None
            if isinstance(value, Parameter):
                self._max_volume_param = value
                value.node = self
            else:
                self._max_volume = value

    cpdef get_max_volume(self, Timestep ts, ScenarioIndex scenario_index):
        if self._max_volume_param is None:
            return self._max_volume
        return self._max_volume_param.value(ts, scenario_index)

    property current_pc:
        " Current percentage full "
        def __get__(self, ):
            return np.asarray(self._current_pc)

    property domain:
        def __get__(self):
            return self._domain

        def __set__(self, value):
            self._domain = value

    property variables:
        """ Returns a list of any set Parameters with is_variable == True
        """
        def __get__(self):
            variables = []
            if self._cost_param is not None:
                if self._cost_param.is_variable:
                    variables.append(self._cost_param)
            if self._min_volume_param is not None:
                if self._min_volume_param.is_variable:
                    variables.append(self._min_volume_param)
            if self._max_volume_param is not None:
                if self._max_volume_param.is_variable:
                    variables.append(self._max_volume_param)
            return variables

    cpdef setup(self, model):
        """Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._volume = np.empty(ncomb, dtype=np.float64)
        self._current_pc = np.empty(ncomb, dtype=np.float64)

    cpdef reset(self):
        """Called at the beginning of a run"""
        AbstractNode.reset(self)
        cdef int i
        for i in range(self._volume.shape[0]):
            self._volume[i] = self._initial_volume
            # TODO fix this for variable max_volume
            self._current_pc[i] = self._volume[i] / self._max_volume

        if self._max_volume_param is not None:
            self._max_volume_param.reset()
        if self._min_volume_param is not None:
            self._min_volume_param.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        AbstractNode.before(self, ts)

        # Complete any parameter calculations
        if self._max_volume_param is not None:
            self._max_volume_param.before(ts)
        if self._min_volume_param is not None:
            self._min_volume_param.before(ts)

    cpdef after(self, Timestep ts):
        AbstractNode.after(self, ts)
        cdef int i
        for i in range(self._flow.shape[0]):
            self._volume[i] += self._flow[i]*ts._days
            # TODO fix this for variable max_volume
            self._current_pc[i] = self._volume[i] / self._max_volume

        if self._max_volume_param is not None:
            self._max_volume_param.after(ts)
        if self._min_volume_param is not None:
            self._min_volume_param.after(ts)

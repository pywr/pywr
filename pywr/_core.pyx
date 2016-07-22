from pywr._core cimport *
import itertools
import numpy as np
cimport numpy as np
import pandas as pd

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
    def __init__(self, model, str name, int size=1):
        self._name = name
        self._size = size
        model.scenarios.add_scenario(self)

    property size:
        def __get__(self, ):
            return self._size

    property name:
        def __get__(self):
            return self._name

cdef class ScenarioCollection:
    def __init__(self, ):
        self._scenarios = []
        self.combinations = ScenarioCombinations(self)

    property scenarios:
        def __get__(self):
            return self._scenarios

    cpdef int get_scenario_index(self, Scenario sc) except? -1:
        """Return the index of Scenario in this controller."""
        return self._scenarios.index(sc)

    cpdef add_scenario(self, Scenario sc):
        if sc in self._scenarios:
            raise ValueError("The same scenario can not be added twice.")
        self._scenarios.append(sc)

    property combination_names:
        def __get__(self):
            cdef ScenarioIndex si
            cdef Scenario sc
            cdef int i
            cdef list names
            for si in self.combinations:
                names = []
                for i, sc in enumerate(self._scenarios):
                    names.append('{}.{:03d}'.format(sc._name, si._indices[i]))
                yield '-'.join(names)

    def __len__(self):
        return len(self._scenarios)

    property shape:
        def __get__(self):
            if len(self._scenarios) == 0:
                return (1, )
            return tuple(sc.size for sc in self._scenarios)

    property multiindex:
        def __get__(self):
            cdef Scenario sc
            if len(self._scenarios) == 0:
                return pd.MultiIndex.from_product([range(1),], names=[''])
            else:
                ranges = [range(sc._size) for sc in self._scenarios]
                names = [sc._name for sc in self._scenarios]
                return pd.MultiIndex.from_product(ranges, names=names)

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
    def __init__(self, datetime, int index, double days):
        self._datetime = pd.Timestamp(datetime)
        self._index = index
        self._days = days
        self.dayofyear = self.datetime.timetuple().tm_yday

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
    """ Base class for all nodes in Pywr.

    This class is not intended to be used directly.
    """
    def __cinit__(self):
        self._allow_isolated = False
        self.virtual = False

    def __init__(self, model, name, comment=None, **kwargs):
        self._model = model
        self.name = name
        self.comment = comment

        self._parent = kwargs.pop('parent', None)
        self._domain = kwargs.pop('domain', None)
        self._recorders = []

        self._flow = np.empty([0,], np.float64)

        # there shouldn't be any unhandled keyword arguments by this point
        if kwargs:
            raise TypeError("__init__() got an unexpected keyword argument '{}'".format(list(kwargs.items())[0]))

    property allow_isolated:
        """ A property to flag whether this Node can be unconnected in a network. """
        def __get__(self):
            return self._allow_isolated
        def __set__(self, value):
            self._allow_isolated = value

    property name:
        """ Name of the node. """
        def __get__(self):
            return self._name

        def __set__(self, name):
            # check for name collision
            if name in self.model.nodes.keys():
                raise ValueError('A node with the name "{}" already exists.'.format(name))
            # apply new name
            self._name = name

    property recorders:
        """ Returns a list of `Recorder` objects attached to this node.

         See also
         --------
         `Recorder`
         """
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

    def __repr__(self):
        if self.name:
            # e.g. <Node "oxford">
            return '<{} "{}">'.format(self.__class__.__name__, self.name)
        else:
            return '<{} "{}">'.format(self.__class__.__name__, hex(id(self)))

    cpdef setup(self, model):
        """Called before the first run of the model"""
        cdef int ncomb = len(model.scenarios.combinations)
        self._flow = np.empty(ncomb, dtype=np.float64)
        self._prev_flow = np.zeros(ncomb, dtype=np.float64)

    cpdef reset(self):
        """Called at the beginning of a run"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

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
        self._prev_flow[:] = self._flow[:]

    cpdef check(self,):
        pass

cdef class Node(AbstractNode):
    """ Node class from which all others inherit
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
            else:
                self._cost_param = None
                self._cost = value

    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts, scenario_index)

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
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef double get_min_flow(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
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
            else:
                self._max_flow_param = None
                self._max_flow = value

    cpdef double get_max_flow(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
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
                variables.extend(self._cost_param.variables)
            if self._min_flow_param is not None:
                variables.extend(self._min_flow_param.variables)
            if self._max_flow_param is not None:
                variables.extend(self._max_flow_param.variables)
            return variables

    cpdef double get_conversion_factor(self) except? -1:
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
        # Setup any Parameters and Recorders
        if self._cost_param is not None:
            self._cost_param.setup(model)
        if self._min_flow_param is not None:
            self._min_flow_param.setup(model)
        if self._max_flow_param is not None:
            self._max_flow_param.setup(model)

    cpdef reset(self):
        # Reset any Parameters and Recorders
        if self._cost_param is not None:
            self._cost_param.reset()
        if self._min_flow_param is not None:
            self._min_flow_param.reset()
        if self._max_flow_param is not None:
            self._max_flow_param.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        AbstractNode.before(self, ts)

        # Complete any parameter calculations
        if self._cost_param is not None:
            self._cost_param.before(ts)
        if self._max_flow_param is not None:
            self._max_flow_param.before(ts)
        if self._min_flow_param is not None:
            self._min_flow_param.before(ts)

    cpdef after(self, Timestep ts):
        """Called at the end of the timestep"""
        AbstractNode.after(self, ts)

        # Complete any parameter calculations
        if self._cost_param is not None:
            self._cost_param.after(ts)
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


cdef class AggregatedNode(AbstractNode):
    """ Base class for a special type of node that is the aggregated sum of `Node` objects.

    This class is intended to be used isolated from the network.
    """
    def __cinit__(self, ):
        self._allow_isolated = True
        self.virtual = True
        self._factors = None
        self._min_flow = -inf
        self._max_flow = inf

    property nodes:
        def __get__(self):
            return self._nodes

        def __set__(self, value):
            self._nodes = list(value)
            self.model.dirty = True

    cpdef after(self, Timestep ts):
        AbstractNode.after(self, ts)
        cdef int i
        cdef Node n

        for i, si in enumerate(self.model.scenarios.combinations):
            self._flow[i] = 0.0
            for n in self._nodes:
                self._flow[i] += n._flow[i]

    property factors:
        def __get__(self):
            if self._factors is None:
                return None
            else:
                return np.asarray(self._factors, np.float64)
        def __set__(self, values):
            values = np.array(values, np.float64)
            self._factors = values
            self.model.dirty = True

    property max_flow:
        def __get__(self):
            return self._max_flow
        def __set__(self, value):
            if value is None:
                value = inf
            self._max_flow = value
            self.model.dirty = True

    property min_flow:
        def __get__(self):
            return self._min_flow
        def __set__(self, value):
            if value is None:
                value = -inf
            self._min_flow = value
            self.model.dirty = True

    @classmethod
    def load(cls, data, model):
        name = data["name"]
        nodes = [model._get_node_from_ref(model, node_name) for node_name in data["nodes"]]
        agg = cls(model, name, nodes)
        try:
            agg.factors = data["factors"]
        except KeyError: pass
        try:
            agg.min_flow = data["min_flow"]
        except KeyError: pass
        try:
            agg.max_flow = data["max_flow"]
        except KeyError: pass
        return agg

cdef class StorageInput(BaseInput):
    cpdef commit(self, int scenario_index, double volume):
        BaseInput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, -volume)

    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # Return negative of parent cost
        return -self.parent.get_cost(ts, scenario_index)

cdef class StorageOutput(BaseOutput):
    cpdef commit(self, int scenario_index, double volume):
        BaseOutput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, volume)

    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # Return parent cost
        return self.parent.get_cost(ts, scenario_index)


cdef class AbstractStorage(AbstractNode):

    property volume:
        def __get__(self, ):
            return np.asarray(self._volume)

    property current_pc:
        """ Current percentage full """
        def __get__(self, ):
            return np.asarray(self._current_pc)

    cpdef setup(self, model):
        """ Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._volume = np.zeros(ncomb, dtype=np.float64)
        self._current_pc = np.zeros(ncomb, dtype=np.float64)


cdef class Storage(AbstractStorage):
    """ Base class for all Storage objects.

    Notes
    -----
    Do not initialise this class directly. Use `pywr.core.Storage`.
    """
    def __cinit__(self, ):
        self._initial_volume = 0.0
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._level_param = None
        self._cost_param = None
        self._domain = None
        self._allow_isolated = True

    property cost:
        """The cost per unit increased in volume stored

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
            else:
                self._cost_param = None
                self._cost = value

    cpdef double get_cost(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts, scenario_index)

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
            else:
                self._min_volume = value

    cpdef double get_min_volume(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
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
            else:
                self._max_volume = value

    cpdef double get_max_volume(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        if self._max_volume_param is None:
            return self._max_volume
        return self._max_volume_param.value(ts, scenario_index)

    property level:
        def __get__(self):
            if self._level_param is None:
                return self._level
            return self._level_param

        def __set__(self, value):
            self._level_param = None
            if isinstance(value, Parameter):
                self._level_param = value
            else:
                self._level = value

    cpdef double get_level(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        if self._level_param is None:
            return self._level
        return self._level_param.value(ts, scenario_index)

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
                variables.extend(self._cost_param.variables)
            if self._min_volume_param is not None:
                variables.extend(self._min_volume_param.variables)
            if self._max_volume_param is not None:
                variables.extend(self._max_volume_param.variables)
            return variables

    cpdef setup(self, model):
        """Called before the first run of the model"""
        AbstractStorage.setup(self, model)
        # Setup any Parameters and Recorders
        if self._cost_param is not None:
            self._cost_param.setup(model)
        if self._min_volume_param is not None:
            self._min_volume_param.setup(model)
        if self._max_volume_param is not None:
            self._max_volume_param.setup(model)

    cpdef reset(self):
        """Called at the beginning of a run"""
        AbstractStorage.reset(self)
        cdef int i
        cdef double mxv = self._max_volume
        cdef ScenarioIndex si

        # Parameters reset first
        if self._cost_param is not None:
            self._cost_param.reset()
        if self._max_volume_param is not None:
            self._max_volume_param.reset()
        if self._min_volume_param is not None:
            self._min_volume_param.reset()

        for i, si in enumerate(self.model.scenarios.combinations):
            self._volume[i] = self._initial_volume
            # Ensure variable maximum volume is taken in to account
            if self._max_volume_param is not None:
                mxv = self._max_volume_param.value(self.model.timestepper.current, si)
            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        AbstractStorage.before(self, ts)

        # Complete any parameter calculations
        if self._cost_param is not None:
            self._cost_param.before(ts)
        if self._max_volume_param is not None:
            self._max_volume_param.before(ts)
        if self._min_volume_param is not None:
            self._min_volume_param.before(ts)

    cpdef after(self, Timestep ts):
        AbstractStorage.after(self, ts)
        cdef int i
        cdef double mxv = self._max_volume
        cdef ScenarioIndex si

        if self._cost_param is not None:
            self._cost_param.after(ts)
        if self._max_volume_param is not None:
            self._max_volume_param.after(ts)
        if self._min_volume_param is not None:
            self._min_volume_param.after(ts)

        for i, si in enumerate(self.model.scenarios.combinations):
            self._volume[i] += self._flow[i]*ts._days
            # Ensure variable maximum volume is taken in to account
            if self._max_volume_param is not None:
                mxv = self._max_volume_param.value(self.model.timestepper.current, si)
            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan


cdef class AggregatedStorage(AbstractStorage):
    """ Base class for a special type of storage node that is the aggregated sum of `Storage` objects.

    This class is intended to be used isolated from the network.
    """
    def __cinit__(self, ):
        self._allow_isolated = True
        self.virtual = True

    property storage_nodes:
        def __get__(self):
            return self._storage_nodes

        def __set__(self, value):
            self._storage_nodes = list(value)

    property initial_volume:
        def __get__(self, ):
            cdef Storage s
            return np.sum([s._initial_volume for s in self._storage_nodes])

    cpdef reset(self):
        cdef int i
        cdef double mxv = 0.0
        cdef ScenarioIndex si

        for i, si in enumerate(self.model.scenarios.combinations):
            for s in self._storage_nodes:
                mxv += s.get_max_volume(self.model.timestepper.current, si)

            self._volume[i] = self.initial_volume
            # Ensure variable maximum volume is taken in to account
            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan

    cpdef after(self, Timestep ts):
        AbstractStorage.after(self, ts)
        cdef int i
        cdef Storage s
        cdef double mxv

        for i, si in enumerate(self.model.scenarios.combinations):
            self._flow[i] = 0.0
            mxv = 0.0
            for s in self._storage_nodes:
                self._flow[i] += s._flow[i]
                mxv += s.get_max_volume(ts, si)
            self._volume[i] += self._flow[i]*ts._days

            # Ensure variable maximum volume is taken in to account
            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan

    @classmethod
    def load(cls, data, model):
        name = data["name"]
        nodes = [model._get_node_from_ref(model, node_name) for node_name in data["storage_nodes"]]
        agg = cls(model, name, nodes)
        return agg


cdef class VirtualStorage(Storage):
    def __cinit__(self, ):
        self.virtual = True

    property nodes:
        def __get__(self):
            return self._nodes

        def __set__(self, value):
            self._nodes = list(value)
            self.model.dirty = True

    cpdef after(self, Timestep ts):
        cdef int i
        cdef ScenarioIndex si
        cdef AbstractNode n

        for i, si in enumerate(self.model.scenarios.combinations):
            self._flow[i] = 0.0
            for n in self._nodes:
                self._flow[i] -= n._flow[i]
        Storage.after(self, ts)
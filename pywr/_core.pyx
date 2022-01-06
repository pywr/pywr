from pywr._core cimport *
from pywr._component cimport Component
import itertools
from libc.math cimport isnan
import numpy as np
cimport numpy as np
import pandas as pd
import warnings

cdef double inf = float('inf')


cdef class Scenario:
    """Represents a scenario in the model.

    Typically a scenario will be used to run many similar models simultaneously. A small
    number of `Parameter` objects in the model will return different values depending
    on the scenario, but many will not. Multiple scenarios can be defined such that
    some `Parameter` values vary with one scenario, but not another. Scenarios are defined
    with a size that represents the number of ensembles in that scenario.

    Parameters
    ----------
    model : `pywr.core.Model`
        The model instance to attach the scenario to.
    name : str
        The name of the scenario.
    size : int, optional
        The number of ensembles in the scenario. The default value is 1.
    slice : slice, optional
        If given this defines the subset of the ensembles that are actually run
         in the model. This is useful if a large number of ensembles is defined, but
         certain analysis (e.g. optimisation) can only be done on a small subset.
    ensemble_names : iterable of str, optional
        User defined names describing each ensemble.

    See Also
    --------
    ScenarioCollection
    ScenarioIndex
    """
    def __init__(self, model, name, int size=1, slice slice=None, ensemble_names=None):
        self._name = name
        if size < 1:
            raise ValueError("Size must be greater than or equal to 1.")
        self._size = size
        self.slice = slice
        self.ensemble_names = ensemble_names
        # Do this last so only set on the model if no error is raised.
        model.scenarios.add_scenario(self)

    property size:
        def __get__(self, ):
            return self._size

    property name:
        def __get__(self):
            return self._name

    property ensemble_names:
        def __get__(self):
            if self._ensemble_names is None:
                return list(range(self._size))
            return self._ensemble_names

        def __set__(self, names):
            if names is None:
                self._ensemble_names = None
                return
            if len(names) != self._size:
                raise ValueError("The length of ensemble_names ({}) must be equal to the size of the scenario ({})".format(len(names), self.size))  # noqa
            self._ensemble_names = names

cdef class ScenarioCollection:
    """ Represents a collection of `Scenario` objects.

    This class is used by a `Model` instance to hold the defined scenarios and control
    which combinations of ensembles are used during model execution. By default the
    product of all scenario ensembles (i.e. all possible combinations of ensembles) is
    executed. However user defined slices can be set on individual `Scenario` instances
    to restrict the number of ensembles executed from that scenario. Alternatively
    the user may provide an array of the specific ensemble combinations (indices) that
    should be run. The latter approach takes precedent over the former per `Scenario`
    slices.

    See Also
    --------
    Scenario
    ScenarioIndex
    """
    def __init__(self, model):
        self.model = model
        self._scenarios = []
        self.combinations = None
        self.user_combinations = None

    property scenarios:
        def __get__(self):
            return self._scenarios

    def __getitem__(self, name):
        cdef Scenario sc
        for sc in self._scenarios:
            if sc._name == name:
                return sc
        raise KeyError("Scenario with name '{}' not found.".format(name))

    def get_combinations(self):
        """Returns a list of ScenarioIndices for every combination of Scenarios
        """
        cdef Scenario scenario
        cdef int i
        if len(self._scenarios) == 0:
            # model has no scenarios defined, implicitly has 1 scenario of size 1
            combinations = [ScenarioIndex(0, np.array([0], dtype=np.int32))]
        elif self._user_combinations is not None:
            # use combinations given by user
            combinations = list([ScenarioIndex(i, self._user_combinations[i, :])
                                 for i in range(self._user_combinations.shape[0])])
        else:
            # product of all scenarios, taking into account Scenario.slice
            iter = itertools.product(*[range(scenario._size)[scenario.slice]
                                     if scenario.slice else range(scenario._size)
                                     for scenario in self._scenarios])
            combinations = list([ScenarioIndex(i, np.array(x, dtype=np.int32)) for i, x in enumerate(iter)])
        if not combinations:
            raise ValueError("No scenarios were selected to be run")
        return combinations

    def setup(self):
        self.combinations = self.get_combinations()

    property user_combinations:
        def __get__(self, ):
            return self._user_combinations

        def __set__(self, values):
            if values is None:
                self._user_combinations = None
                return
            cdef Scenario sc
            values = np.asarray(values, dtype=np.int32)
            if values.ndim != 2:
                raise ValueError('A 2-dimensional array of scenario indices must be provided.')
            if values.shape[1] != len(self._scenarios):
                raise ValueError('User defined combinations must have shape (N, S) where S in number of Scenarios')
            # Check maximum values
            for sc, v in zip(self._scenarios, values.max(axis=0)):
                if v >= sc._size:
                    raise ValueError('Given ensemble index for scenario "{}" out of range.'.format(sc.name))
            if np.any(values.min(axis=0) < 0):
                raise ValueError('Ensemble index less than zero is invalid.')

            self._user_combinations = values

    cpdef int get_scenario_index(self, Scenario sc) except? -1:
        """Return the index of Scenario in this controller."""
        return self._scenarios.index(sc)

    cpdef add_scenario(self, Scenario sc):
        if sc in self._scenarios:
            raise ValueError("The same scenario can not be added twice.")
        self.model.dirty = True
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
            if self._user_combinations is not None:
                # raise ValueError("ScenarioCollection.shape is undefined if user_combinations is defined.")
                return (len(self._user_combinations), )
            if len(self._scenarios) == 0:
                return (1, )
            return tuple(len(range(sc.size)[sc.slice]) if sc.slice is not None else sc.size for sc in self._scenarios)

    property multiindex:
        def __get__(self):
            cdef Scenario sc
            if len(self._scenarios) == 0:
                return pd.MultiIndex.from_product([range(1)], names=[''])
            else:
                ensemble_names = [scenario.ensemble_names for scenario in self._scenarios]
                indices = [[ensemble_names[n][i] for n, i in enumerate(scenario_index.indices)]
                           for scenario_index in self.model.scenarios.get_combinations()]
                names = [sc._name for sc in self._scenarios]
                return pd.MultiIndex.from_tuples(indices, names=names)

    cpdef int ravel_indices(self, int[:] scenario_indices) except? -1:
        if scenario_indices is None:
            return 0
        # Case where scenario_indices is empty for no scenarios defined
        if scenario_indices.size == 0:
            return 0
        return np.ravel_multi_index(scenario_indices, np.array(self.shape))

cdef class ScenarioIndex:
    """A ScenarioIndex is an indexer for a given combination of scenarios.

    The ScenarioIndex is used to represent a given combination of scenarios
    during a simulation. It is given to Parameter instances to allow them
    to alter their calculations based on each scenario. Instances of this
    class are generated by Pywr, and users are unlikely to need to
    instantiate this class.

    Attributes
    ----------
    global_id : int
        Read-only global scenario index that this ScenarioIndex refers to.
    indices : np.array
        The indices for each of the Scenarios in this model.

    See Also
    --------
    Scenario
    ScenarioCollection
    """
    def __init__(self, int global_id, int[:] indices):
        self.global_id = global_id
        self._indices = indices

    property indices:
        def __get__(self):
            return np.array(self._indices)

    def __repr__(self):
        return "<ScenarioIndex gid={:d} indices={}>".format(self.global_id, tuple(np.asarray(self._indices)))


cdef bint is_leap_year(int year):
    # http://stackoverflow.com/a/11595914/1300519
    return ((year & 3) == 0 and ((year % 25) != 0 or (year & 15) == 0))


cdef class Timestep:
    """A representation of a simulation time-step.

    Attributes
    ----------
    period : pandas.Period
        Read-only global scenario index that this ScenarioIndex refers to.
    index : int
        The time-step index of the simulation.
    days : int
        The time-step length in days.
    dayofyear : int
        The day of the year (from `pandas.Period.dayofyear`).
    dayofyear_index : int
        A zero-based index for the day of the year. This begins at 0 and ends at 366. The 29th of February
        is skipped if it is not a leap year.
    day : int
        The day of the month.
    week_index : int
        A zero-based index for the week of the year. The last week of the year (index 51) is always slightly
        longer than 7 days.
    month : int
        The month number (1 - 12).
    year : int
        The year.
    is_leap_year : bool
        True if the current year is a leap year; false otherwise.
    """
    def __init__(self, period, int index, double days):
        self.period = period
        self.index = index
        if days <= 0:
            raise ValueError("The days argument must be > 0.")
        self.days = days
        self.dayofyear = period.dayofyear
        self.day = period.day
        self.month = period.month
        self.year = period.year
        self.is_leap_year = is_leap_year(self.year)
        self.end_year = self.period.end_time.year

        # Calculate day of year index (zero based)
        cdef int i = self.dayofyear - 1
        if not self.is_leap_year:
            if i > 58:  # 28th Feb
                i += 1
        self.dayofyear_index = i

        # Calculate week of year
        if self.dayofyear_index >= 364:
            # last week of year is slightly longer than 7 days
            self.week_index = 51
        else:
            self.week_index = self.dayofyear_index // 7

    property datetime:
        """Timestep representation as a `datetime.datetime` object"""
        def __get__(self, ):
            return self.period.to_timestamp()

    def __repr__(self):
        return "<Timestep date=\"{}\">".format(self.period.strftime("%Y-%m-%d"))

    cpdef double days_in_current_year(self):
        """Returns the number of days of the current timestep that fall in the current year"""
        cdef double year_end, ts_start
        if self.year != self.end_year:
            # end time of period is in the next year
            year_end = pd.Timestamp(f"{self.end_year}-01-01").value
            ts_start = self.period.start_time.value
            return (year_end - ts_start) / 8.64e+13
        return self.days

    cpdef double days_in_next_year(self):
        """Returns the number of days of the current timestep that fall in the next year"""
        cdef double year_end, ts_end
        if self.year != self.end_year:
            year_end = pd.Timestamp(f"{self.end_year}-01-01").value
            ts_end = self.period.end_time.value
            return (ts_end - year_end) / 8.64e+13
        return 0

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

        self._flow = np.empty([0], np.float64)

        # there shouldn't be any unhandled keyword arguments by this point
        if kwargs:
            raise TypeError("__init__() got an unexpected keyword argument '{}'".format(list(kwargs.items())[0]))

    component_attrs = []  # redefined by subclasses
    property components:
        """Generator that returns all of the Components attached to the Node

        This is used by Model.find_orphaned_parameters and isn't performance
        critical.
        """
        def __get__(self):
            for attr in self.component_attrs:
                try:
                    component = getattr(self, attr)
                except AttributeError:
                    pass
                else:
                    if isinstance(component, Component):
                        yield component

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

    property fully_qualified_name:
        def __get__(self):
            if self._parent is not None:
                return '{}.{}'.format(self._parent.fully_qualified_name, self.name)
            return self.name

    property recorders:
        """ Returns a list of `pywr.recorders.Recorder` objects attached to this node.

         See also
         --------
         pywr.recorders.Recorder
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
            self._prev_flow[i] = 0.0

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

    cpdef finish(self):
        pass

    cpdef check(self,):
        pass

    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1:
        return 0.0

    cpdef double[:] get_all_cost(self, double[:] out=None):
        if out is None:
            out = np.zeros(len(self.model.scenarios.combinations))
        else:
            out[:] = 0.0
        return out


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

    component_attrs = ["min_flow", "max_flow", "cost", "conversion_factor"]

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

    property has_fixed_cost:
        """Returns true if cost is not a Parameter."""
        def __get__(self):
            return self._cost_param is None

    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1:
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.get_value(scenario_index)

    cpdef double[:] get_all_cost(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._cost_param is None:
            out[:] = self._cost
        else:
            out[:] = self._cost_param.get_all_values()
        return out

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
            if value is None:
                self._min_flow = 0
                self._min_flow_param = None
            elif isinstance(value, Parameter):
                self._min_flow_param = value
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef double get_fixed_min_flow(self):
        """Returns min_flow value if it is a fixed value otherwise returns NaN."""
        if self.has_fixed_flows:
            return self._min_flow
        return float('nan')

    cpdef double get_constant_min_flow(self):
        """Returns min_flow value if it is a constant parameter or fixed value otherwise returns NaN."""
        if self._min_flow_param is None:
            return self._min_flow
        elif self._min_flow_param.is_constant:
            return self._min_flow_param.get_constant_value()
        return float('nan')

    cpdef double get_min_flow(self, ScenarioIndex scenario_index) except? -1:
        """Get the minimum flow at a given timestep
        """
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.get_value(scenario_index)

    cpdef double[:] get_all_min_flow(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._min_flow_param is None:
            out[:] = self._min_flow
        else:
            out[:] = self._min_flow_param.get_all_values()
        return out

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
                self._max_flow_param = None
            elif isinstance(value, Parameter):
                self._max_flow_param = value
            else:
                self._max_flow_param = None
                self._max_flow = value

    property has_fixed_flows:
        """Returns true if both min_flow and max_flow are not Parameters."""
        def __get__(self):
            return self._max_flow_param is None and self._min_flow_param is None

    property has_constant_flows:
        """Returns true if both min_flow and max_flow are literal constants or "constant" Parameters."""
        def __get__(self):
            return (self._max_flow_param is None or self._max_flow_param.is_constant) and \
                   (self._min_flow_param is None or self._min_flow_param.is_constant)

    cpdef double get_fixed_max_flow(self):
        """Returns max_flow value if it is fixed value otherwise returns NaN."""
        if self.has_fixed_flows:
            return self._max_flow
        return float('nan')

    cpdef double get_constant_max_flow(self):
        """Returns max_flow value if it is a constant parameter or fixed value otherwise returns NaN."""
        if self._max_flow_param is None:
            return self._max_flow
        elif self._max_flow_param.is_constant:
            return self._max_flow_param.get_constant_value()
        return float('nan')

    cpdef double get_max_flow(self, ScenarioIndex scenario_index) except? -1:
        """Get the maximum flow at a given timestep
        """
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.get_value(scenario_index)

    cpdef double[:] get_all_max_flow(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._max_flow_param is None:
            out[:] = self._max_flow
        else:
            out[:] = self._max_flow_param.get_all_values()
        return out

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

    cpdef double get_conversion_factor(self) except? -1:
        """Get the conversion factor

        Note: the conversion factor must be a constant.
        """
        return self._conversion_factor

    cdef set_parameters(self, ScenarioIndex scenario_index):
        """Update the constant attributes by evaluating any Parameter objects

        This is useful when the `get_` functions need to be accessed multiple
        times and there is a benefit to caching the values.
        """
        if self._min_flow_param is not None:
            self._min_flow = self._min_flow_param.get_value(scenario_index)
        if self._max_flow_param is not None:
            self._max_flow = self._max_flow_param.get_value(scenario_index)
        if self._cost_param is not None:
            self._cost = self._cost_param.get_value(scenario_index)


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
        self._flow_weights = None
        self._min_flow = 0.0
        self._max_flow = inf
        self._min_flow_param = None
        self._max_flow_param = None

    component_attrs = ["min_flow", "max_flow"]

    property nodes:
        def __get__(self):
            return self._nodes

        def __set__(self, value):
            self._nodes = list(value)
            self.model.dirty = True

    cpdef after(self, Timestep ts):
        AbstractNode.after(self, ts)
        cdef int i, j
        cdef Node n

        cdef double[:] weights
        if self._flow_weights is not None:
            weights = self._flow_weights
        else:
            weights = np.ones(len(self._nodes))

        for i, si in enumerate(self.model.scenarios.combinations):
            self._flow[i] = 0.0
            for j, n in enumerate(self._nodes):
                self._flow[i] += n._flow[i]*weights[j]

    property factors:
        def __get__(self):
            return self._factors

        def __set__(self, values):
            from pywr.parameters import ConstantParameter

            # remove existing factors (if any)
            if self._factors is not None:
                for factor in self._factors:
                    factor.parents.remove(self)

            if values is None:
                factors = None
            else:
                factors = []
                for val in values:
                    if isinstance(val, (int, float)):
                        factors.append(ConstantParameter(self.model, val))
                    else:
                        factors.append(val)

            self._factors = factors
            self.model.dirty = True

    property has_fixed_factors:
        """Returns true if all factors are of type `ConstantParameter`"""
        def __get__(self):
            from pywr.parameters import ConstantParameter
            return all([isinstance(p, ConstantParameter) for p in self.factors])

    property has_constant_factors:
        """Returns true if all factors are `is_constant==True`"""
        def __get__(self):
            return all([p.is_constant for p in self.factors])

    property flow_weights:
        def __get__(self):
            if self._flow_weights is None:
                return None
            else:
                return np.asarray(self._flow_weights, np.float64)

        def __set__(self, values):
            if values is None:
                self._flow_weights = None
            else:
                values = np.array(values, np.float64)
                if np.any(np.abs(values) < 1e-6):
                    warnings.warn("Very small flow_weights in AggregateNode result in ill-conditioned matrix")
                self._flow_weights = values
            self.model.dirty = True

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
            if value is None:
                self._min_flow = -inf
                self._min_flow_param = None
            elif isinstance(value, Parameter):
                self._min_flow_param = value
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef double get_min_flow(self, ScenarioIndex scenario_index) except? -1:
        """Get the minimum flow at a given timestep
        """
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.get_value(scenario_index)

    cpdef double[:] get_all_min_flow(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._min_flow_param is None:
            out[:] = self._min_flow
        else:
            out[:] = self._min_flow_param.get_all_values()
        return out

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
                self._max_flow_param = None
            elif isinstance(value, Parameter):
                self._max_flow_param = value
            else:
                self._max_flow_param = None
                self._max_flow = value

    cpdef double get_max_flow(self, ScenarioIndex scenario_index) except? -1:
        """Get the maximum flow at a given timestep
        """
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.get_value(scenario_index)

    cpdef double[:] get_all_max_flow(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._max_flow_param is None:
            out[:] = self._max_flow
        else:
            out[:] = self._max_flow_param.get_all_values()
        return out

    cpdef double[:] get_factors(self, ScenarioIndex scenario_index):
        """Get node factors for the current timestep and given scenario index.
        """
        cdef Parameter p
        return np.array([p.get_value(scenario_index) for p in self.factors], np.float64)

    cpdef double[:] get_constant_factors(self):
        """Get constant factors.

        Will return an array of `NaN` if the factors are no `is_constant`. 
        """
        cdef Parameter p
        return np.array([p.get_constant_value() for p in self.factors], np.float64)

    cpdef double[:] get_factors_norm(self, ScenarioIndex scenario_index):
        """Get node factors normalised by the factor of the first node.
        
        If `scenario_index` is `None` assumed to be constant factors.
        """
        cdef double f0, f
        cdef int i
        cdef double[:] factors_norm, factors

        if scenario_index is None:
            factors = self.get_constant_factors()
        else:
            factors = self.get_factors(scenario_index)
        f0 = factors[0]
        factors_norm = np.empty(len(factors), np.float64)

        for i in range(len(factors)):
            factors_norm[i] = f0/factors[i]
        return factors_norm


cdef class StorageInput(BaseInput):
    cpdef commit(self, int scenario_index, double volume):
        BaseInput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, -volume)

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]
        self._parent.commit_all(-np.array(value))

    property has_fixed_cost:
        """Returns true if cost is not a Parameter."""
        def __get__(self):
            return self.parent.has_fixed_cost

    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1:
        # Return negative of parent cost
        return -self.parent.get_cost(scenario_index)

    cpdef double[:] get_all_cost(self, double[:] out=None):
        """Get the cost at a given timestep for all scenario combinations
        """
        cdef int i
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        self.parent.get_all_cost(out=out)
        for i in range(len(self.model.scenarios.combinations)):
            out[i] *= -1.0
        return out


cdef class StorageOutput(BaseOutput):
    cpdef commit(self, int scenario_index, double volume):
        BaseOutput.commit(self, scenario_index, volume)
        self._parent.commit(scenario_index, volume)

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]
        self._parent.commit_all(value)

    property has_fixed_cost:
        """Returns true if cost is not a Parameter."""
        def __get__(self):
            return self.parent.has_fixed_cost

    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1:
        # Return parent cost
        return self.parent.get_cost(scenario_index)

    cpdef double[:] get_all_cost(self, double[:] out=None):
        """Get the cost at a given timestep for all scenario combinations
        """
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        self.parent.get_all_cost(out=out)
        return out


cdef class AbstractStorage(AbstractNode):
    """ Base class for all Storage objects.

    Notes
    -----
    Do not initialise this class directly. Use `pywr.core.Storage`.
    """
    property volume:
        def __get__(self, ):
            return np.asarray(self._volume)

    property current_pc:
        """ Current proportion full.
         
        Note that this property is the raw internal value of the current_pc and may contain `NaN` values. Prefer
        use of the `get_current_pc` method to return a guaranteed finite value between 0.0 and 1.0.
        """
        def __get__(self, ):
            return np.asarray(self._current_pc)

    cpdef double get_current_pc(self, ScenarioIndex scenario_index):
        """Return the current proportion of full of the storage node.
        
        This method will always return a finite value between 0.0 and 1.0. If the current proportion is `NaN` 
        (usually because max_volume is zero) it is assumed full (i.e. returns 1.0). It is preferable to use
        this method in Parameter calculations to avoid dealing with NaN or out of range values.
        """
        cdef double current_pc = self._current_pc[scenario_index.global_id]
        if current_pc > 1.0 or isnan(current_pc):
            current_pc = 1.0
        elif current_pc < 0.0:
            current_pc = 0.0
        return current_pc

    cpdef setup(self, model):
        """ Called before the first run of the model"""
        AbstractNode.setup(self, model)
        cdef int ncomb = len(model.scenarios.combinations)
        self._volume = np.zeros(ncomb, dtype=np.float64)
        self._current_pc = np.zeros(ncomb, dtype=np.float64)


cdef class Storage(AbstractStorage):
    def __cinit__(self, ):
        self.initial_volume = None
        self.initial_volume_pc = None
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._level_param = None
        self._area_param = None
        self._cost_param = None
        self._domain = None
        self._allow_isolated = True

    component_attrs = ["cost", "min_volume", "max_volume", "level", "area"]

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

    property has_fixed_cost:
        """Returns true if cost is not a Parameter."""
        def __get__(self):
            return self._cost_param is None

    cpdef double get_cost(self, ScenarioIndex scenario_index) except? -1:
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.get_value(scenario_index)

    cpdef double[:] get_all_cost(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._cost_param is None:
            out[:] = self._cost
        else:
            out[:] = self._cost_param.get_all_values()
        return out

    property initial_volume:
        def __get__(self, ):
            return self._initial_volume

        def __set__(self, value):
            if value is None:
                self._initial_volume = np.nan
            else:
                self._initial_volume = value

    property initial_volume_pc:
        def __get__(self, ):
            return self._initial_volume_pc

        def __set__(self, value):
            if value is None:
                self._initial_volume_pc = np.nan
            else:
                self._initial_volume_pc = value

    cpdef double get_initial_volume(self) except? -1:
        """Returns the absolute initial volume. """
        cdef double mxv = self._max_volume

        if self._max_volume_param is not None:
            # Max volume is a parameter; require both initial_volume and initial_volume_pc be given.
            # The parameter will not be evaluated at the beginning of the model run.
            if not np.isfinite(self._initial_volume_pc) or not np.isfinite(self._initial_volume):
                raise RuntimeError('Both `initial_volume` and `initial_volume_pc` must be supplied if'
                                   '`max_volume` is defined as a parameter.')

            initial_volume = self._initial_volume
        else:
            # User only has to supply absolute or relative initial volume
            if np.isfinite(self._initial_volume_pc):
                initial_volume = self._initial_volume_pc * mxv
            elif np.isfinite(self._initial_volume):
                initial_volume = self._initial_volume
            else:
                raise RuntimeError('Initial volume must be set as either a percentage or absolute volume.')
        return initial_volume

    cpdef double get_initial_pc(self) except? -1:
        """Returns the initial volume as a proportion. """
        cdef double mxv = self._max_volume

        if self._max_volume_param is not None:
            # Max volume is a parameter; require both initial_volume and initial_volume_pc be given.
            # The parameter will not be evaluated at the beginning of the model run.
            if not np.isfinite(self._initial_volume_pc) or not np.isfinite(self._initial_volume):
                raise RuntimeError('Both `initial_volume` and `initial_volume_pc` must be supplied if'
                                   '`max_volume` is defined as a parameter.')
            initial_pc = self._initial_volume_pc
        else:
            # User only has to supply absolute or relative initial volume
            if np.isfinite(self._initial_volume_pc):
                initial_pc = self._initial_volume_pc
            elif np.isfinite(self._initial_volume):
                try:
                    initial_pc = self._initial_volume / mxv
                except ZeroDivisionError:
                    initial_pc = np.nan
            else:
                raise RuntimeError('Initial volume must be set as either a percentage or absolute volume.')
        return initial_pc

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

    cpdef double get_min_volume(self, ScenarioIndex scenario_index) except? -1:
        if self._min_volume_param is None:
            return self._min_volume
        return self._min_volume_param.get_value(scenario_index)

    cpdef double[:] get_all_min_volume(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._min_volume_param is None:
            out[:] = self._min_volume
        else:
            out[:] = self._min_volume_param.get_all_values()
        return out

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

    cpdef double get_max_volume(self, ScenarioIndex scenario_index) except? -1:
        if self._max_volume_param is None:
            return self._max_volume
        return self._max_volume_param.get_value(scenario_index)

    cpdef double[:] get_all_max_volume(self, double[:] out=None):
        if out is None:
            out = np.empty(len(self.model.scenarios.combinations))

        if self._max_volume_param is None:
            out[:] = self._max_volume
        else:
            out[:] = self._max_volume_param.get_all_values()
        return out

    property level:
        def __get__(self):
            if self._level_param is None:
                return self._level
            return self._level_param

        def __set__(self, value):
            self._level_param = None
            if value is None:
                self._level_param = None
                self._level = 0.0
            elif isinstance(value, Parameter):
                self._level_param = value
            else:
                self._level = value

    cpdef double get_level(self, ScenarioIndex scenario_index) except? -1:
        if self._level_param is None:
            return self._level
        return self._level_param.get_value(scenario_index)

    property area:
        def __get__(self):
            if self._area_param is None:
                return self._area
            return self._area_param

        def __set__(self, value):
            self._area_param = None
            if value is None:
                self._area_param = None
                self._area = 0.0
            elif isinstance(value, Parameter):
                self._area_param = value
            else:
                self._area = value

    cpdef double get_area(self, ScenarioIndex scenario_index) except? -1:
        if self._area_param is None:
            return self._area
        return self._area_param.get_value(scenario_index)

    property domain:
        def __get__(self):
            return self._domain

        def __set__(self, value):
            self._domain = value

    cpdef reset(self):
        """Called at the beginning of a run"""
        AbstractStorage.reset(self)
        self._reset_storage_only()

    cpdef _reset_storage_only(self, bint use_initial_volume=True):
        """Reset the current volume of the storage node.

        Parameters
        ==========
        use_initial_volume : bool (default: True)
            Reset the volume to the initial volume of the storage node. If false the volume is reset to max_volume.
        """
        cdef int i
        cdef double mxv = self._max_volume
        cdef double reset_volume, reset_pc

        for i, si in enumerate(self.model.scenarios.combinations):

            if use_initial_volume:
                reset_volume = self.get_initial_volume()
                reset_pc = self.get_initial_pc()
            else:
                if self._max_volume_param is not None:
                    # If it's not the first time-step, and we want to reset to max capacity then the
                    # parameter should already have been evaluated.
                    mxv = self._max_volume_param.get_value(si)
                    # Reset to max capacity, accounting for variable volume
                    reset_volume = mxv
                    # We do this division (rather than setting reset_pc = 1.0) to be consistent with the case below
                    # when mxv is zero.
                    try:
                        reset_pc = reset_volume / mxv
                    except ZeroDivisionError:
                        reset_pc = np.nan
                else:
                    reset_volume = mxv
                    # Compute the proportional volume.
                    try:
                        reset_pc = reset_volume / mxv
                    except ZeroDivisionError:
                        reset_pc = np.nan

            self._volume[i] = reset_volume
            self._current_pc[i] = reset_pc

    cpdef after(self, Timestep ts, double[:] adjustment=None):
        AbstractStorage.after(self, ts)
        cdef int i
        cdef double mxv, mnv
        cdef ScenarioIndex si

        for i, si in enumerate(self.model.scenarios.combinations):
            self._volume[i] += self._flow[i]*ts.days
            # Ensure variable maximum volume is taken in to account
            mxv = self.get_max_volume(si)
            mnv = self.get_min_volume(si)

            # Apply any storage adjustment if given
            if adjustment is not None:
                self._volume[i] += adjustment[i]
                # Ensure volume stays within bounds.
                self._volume[i] = min(mxv, max(mnv, self._volume[i]))

            if abs(self._volume[i] - mxv) < 1e-6:
                self._volume[i] = mxv
            if abs(self._volume[i] - mnv) < 1e-6:
                self._volume[i] = mnv

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
            return np.sum([s.get_initial_volume() for s in self._storage_nodes])

    cpdef reset(self):
        cdef int i
        cdef double mxv = 0.0
        cdef ScenarioIndex si

        for i, si in enumerate(self.model.scenarios.combinations):
            mxv = 0.0
            for s in self._storage_nodes:
                mxv += s.get_max_volume(si)

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
                mxv += s.get_max_volume(si)
            self._volume[i] += self._flow[i]*ts.days

            # Ensure variable maximum volume is taken in to account
            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan


cdef class VirtualStorage(Storage):
    def __cinit__(self, ):
        self._allow_isolated = True
        self.virtual = True
        self.active = True

    cpdef reset(self):
        self.active = True
        Storage.reset(self)

    property nodes:
        def __get__(self):
            return self._nodes

        def __set__(self, value):
            self._nodes = list(value)
            self.model.dirty = True

    property factors:
        def __get__(self):
            return np.array(self._factors)

        def __set__(self, value):
            self._factors = np.array(value, dtype=np.float64)

    cpdef after(self, Timestep ts, double[:] adjustment=None):
        cdef int i
        cdef ScenarioIndex si
        cdef AbstractNode n

        if self.active:
            for i, si in enumerate(self.model.scenarios.combinations):
                self._flow[i] = 0.0
                for n, f in zip(self._nodes, self._factors):
                    self._flow[i] -= f*n._flow[i]
            Storage.after(self, ts, adjustment=adjustment)


cdef class RollingVirtualStorage(VirtualStorage):
    def __cinit__(self):
        self._memory_pointer = 0

    cpdef setup(self, model):
        super(RollingVirtualStorage, self).setup(model)
        cdef int ncomb = len(model.scenarios.combinations)
        if self.timesteps < 2:
            raise ValueError('The number of time-steps for a RollingVirtualStorage node must be greater than one.')
        self._memory = np.zeros((self.timesteps-1, ncomb), dtype=np.float64)
        self._memory_pointer = 0

    cpdef reset(self):
        VirtualStorage.reset(self)
        self._memory[:] = 0.0
        self._memory_pointer = 0

    cpdef after(self, Timestep ts, double[:] adjustment=None):
        cdef int i
        cdef ScenarioIndex si

        assert adjustment is None

        # Update the storage volumes by applying an adjustment
        VirtualStorage.after(self, ts, adjustment=self._memory[self._memory_pointer, :])

        # Store today's flow in the memory and increment the memory pointer
        for i, si in enumerate(self.model.scenarios.combinations):
            # Flow is negative in VirtualStorage to remove from the store, save the +ve number here for
            # returning to the store later
            self._memory[self._memory_pointer, i] = -self._flow[i] * ts.days
        self._memory_pointer = (self._memory_pointer + 1) % (self.timesteps - 1)

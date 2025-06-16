import numpy as np
cimport numpy as np
from scipy.stats import percentileofscore
import pandas as pd
import warnings


recorder_registry = {}

cdef enum AggFuncs:
    SUM = 0
    MIN = 1
    MAX = 2
    MEAN = 3
    MEDIAN = 4
    PRODUCT = 5
    CUSTOM = 6
    PERCENTILE = 7
    PERCENTILEOFSCORE = 8
    COUNT_NONZERO = 9
_agg_func_lookup = {
    "sum": AggFuncs.SUM,
    "min": AggFuncs.MIN,
    "max": AggFuncs.MAX,
    "mean": AggFuncs.MEAN,
    "median": AggFuncs.MEDIAN,
    "product": AggFuncs.PRODUCT,
    "custom": AggFuncs.CUSTOM,
    "percentile": AggFuncs.PERCENTILE,
    "percentileofscore": AggFuncs.PERCENTILEOFSCORE,
    "count_nonzero": AggFuncs.COUNT_NONZERO
}
_agg_func_lookup_reverse = {v: k for k, v in _agg_func_lookup.items()}

cdef enum ObjDirection:
    NONE = 0
    MAXIMISE = 1
    MINIMISE = 2
_obj_direction_lookup = {
    "maximize": ObjDirection.MAXIMISE,
    "maximise": ObjDirection.MAXIMISE,
    "max": ObjDirection.MAXIMISE,
    "minimise": ObjDirection.MINIMISE,
    "minimize": ObjDirection.MINIMISE,
    "min": ObjDirection.MINIMISE,
}

cdef class Aggregator:
    """Utility class for computing aggregate values.

    Users are unlikely to use this class directly. Instead [pywr.recorders.Recorder][] subclasses will use this functionality
    to aggregate their results across different dimensions (e.g., time, scenarios, etc.).

    Examples
    -------
    ```python
    a = Aggregator("sum")
    a.aggregated_1d([1.0, 4.0, 9.0]) # this returns 14

    Aggregator({"func": "percentile", "args": [95],"kwargs": {}})
    Aggregator({"func": "percentileofscore", "kwargs": {"score": 0.5, "kind": "rank"}})
    ```

    Attributes
    ----------
    func : str | dict | Callable
        The aggregation function to use. 

    """
    def __init__(self, func):
        """Initialise the class.
        
        Parameters
        ----------
        func : str | dict | Callable
            The aggregation function to use. This can be a string or dict defining aggregation functions, or a callable
            custom function that performs aggregation.

            When a string it can be one of: "sum", "min", "max", "mean", "median", "product", or "count_nonzero". These
            strings map to and cause the aggregator to use the corresponding [numpy functions](https://numpy.org/doc/stable/reference/routines.statistics.html).

            A dict can be provided containing a "func" key, and optional "args" and "kwargs" keys. The value of "func"
            should be a string corresponding to the aforementioned numpy function names with the additional options of
            "percentile" and "percentileofscore". These latter two functions require additional arguments (the percentile
            and score) to function and must be provided as the values in either the "args" or "kwargs" keys of the
            dictionary. Please refer to the corresponding numpy (or scipy) function definitions for documentation on these
            arguments.

            Finally, a callable function can be given. This function must accept either a 1D or 2D numpy array as the
            first argument, and support the "axis" keyword as integer value that determines which axis over which the
            function should apply aggregation. The axis keyword is only supplied when a 2D array is given. Therefore,`
            the callable function should behave in a similar fashion to the numpy functions.
        
        """
        self.func = func

    property func:
        def __get__(self):
            if self._func == AggFuncs.CUSTOM:
                return self._user_func
            return _agg_func_lookup_reverse[self._func]
        def __set__(self, func):
            self._user_func = None
            func_args = []
            func_kwargs = {}
            if isinstance(func, str):
                func_type = _agg_func_lookup[func.lower()]
            elif isinstance(func, dict):
                func_type = _agg_func_lookup[func['func']]
                func_args = func.get('args', [])
                func_kwargs = func.get('kwargs', {})
            elif callable(func):
                self._user_func = func
                func_type = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(func))
            self._func = func_type
            self.func_args = func_args
            self.func_kwargs = func_kwargs

    cpdef double aggregate_1d(self, double[:] data, ignore_nan=False) except *:
        """Compute an aggregated value across 1D array.

        Parameters
        ---------
        data : Iterable[float]
            The 1D array to aggregated from.
        ignore_nan : Optional[bool], default=False
            Remove NaNs before aggregating.

        Returns
        -------
        float
            The result of the aggregation.

        Raises
        ------
        ValueError
            If the aggregation function does not exist.
        """
        cdef double[:] values = data

        if ignore_nan:
            values = np.array(values)[~np.isnan(values)]
            if len(values) == 0:
                return np.nan

        if self._func == AggFuncs.PRODUCT:
            return np.prod(values)
        elif self._func == AggFuncs.SUM:
            return np.sum(values)
        elif self._func == AggFuncs.MAX:
            return np.max(values)
        elif self._func == AggFuncs.MIN:
            return np.min(values)
        elif self._func == AggFuncs.MEAN:
            return np.mean(values)
        elif self._func == AggFuncs.MEDIAN:
            return np.median(values)
        elif self._func == AggFuncs.CUSTOM:
            return self._user_func(np.array(values))
        elif self._func == AggFuncs.PERCENTILE:
            return np.percentile(values, *self.func_args, **self.func_kwargs)
        elif self._func == AggFuncs.PERCENTILEOFSCORE:
            return percentileofscore(values, *self.func_args, **self.func_kwargs)
        elif self._func == AggFuncs.COUNT_NONZERO:
            return np.count_nonzero(values)
        else:
            raise ValueError('Aggregation function code "{}" not recognised.'.format(self._func))

    cpdef double[:] aggregate_2d(self, double[:, :] data, axis=0, ignore_nan=False) except *:
        """Compute an aggregated value along an axis of a 2D array.

        Parameters
        ---------
        data : Iterable[float]
            The 2D array to aggregated from.
        axis : Optional[int], default=0
            The dimension or axis along which the aggregation is computed. 
        ignore_nan : Optional[bool], default=False
            Remove NaNs before aggregating.

        Returns
        -------
        Iterable[float]
            The result of the aggregation for each other dimension.

        Raises
        ------
        ValueError
            If the aggregation function or axis do not exist.
        """
        cdef double[:, :] values = data
        cdef Py_ssize_t i

        if ignore_nan:
            values = np.array(values)[~np.isnan(values)]

        if self._func == AggFuncs.PRODUCT:
            return np.prod(values, axis=axis)
        elif self._func == AggFuncs.SUM:
            return np.sum(values, axis=axis)
        elif self._func == AggFuncs.MAX:
            return np.max(values, axis=axis)
        elif self._func == AggFuncs.MIN:
            return np.min(values, axis=axis)
        elif self._func == AggFuncs.MEAN:
            return np.mean(values, axis=axis)
        elif self._func == AggFuncs.MEDIAN:
            return np.median(values, axis=axis)
        elif self._func == AggFuncs.CUSTOM:
            return self._user_func(np.array(values), axis=axis)
        elif self._func == AggFuncs.PERCENTILE:
            return np.percentile(values, *self.func_args, axis=axis, **self.func_kwargs)
        elif self._func == AggFuncs.PERCENTILEOFSCORE:
            # percentileofscore doesn't support the axis argument
            # we must therefore iterate over the array
            if axis == 0:
                out = np.empty(data.shape[1])
                for i in range(data.shape[1]):
                    out[i] = percentileofscore(values[:, i], *self.func_args, **self.func_kwargs)
            elif axis == 1:
                out = np.empty(data.shape[0])
                for i in range(data.shape[0]):
                    out[i] = percentileofscore(values[i, :], *self.func_args, **self.func_kwargs)
            else:
                raise ValueError('Axis "{}" not recognised for "percentileofscore" function.'.format(axis))
            return out
        elif self._func == AggFuncs.COUNT_NONZERO:
            return np.count_nonzero(values, axis=axis).astype(np.float64)
        else:
            raise ValueError('Aggregation function code "{}" not recognised.'.format(self._func))


cdef class Recorder(Component):
    """Base class for recording information from a [pywr.core.Model][].

    Recorder components are used to calculate, aggregate and save data from a simulation. This
    base class provides the basic functionality for all recorders. The recorder has two key methods:

    - `.values()` which is implemented by a recorder inherited from this object.
    - `.aggregated_value()` which aggregates the array of numbers returned by `.values()`
        using `agg_func` (usually these contain a number for each model scenario).
        This method returns just one number.

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.


    !!!info "Optimisation bounds"
        -  The constraint bounds are not used during model simulation. Instead, they are intended for use by optimisation
            wrappers (or other external tools) to define constrained optimisation problems.
        - The bound can be `None` (the default) to disable the respective bound. If both bounds are `None` then the
            `is_constraint` property will return `False`. The lower bound must
            be strictly less than the upper bound. An equality constraint can be created by setting both bounds to the
            same value.
    """
    def __init__(self, model, agg_func="mean", ignore_nan=False, is_objective=None, epsilon=1.0,
                 name=None, constraint_lower_bounds=None, constraint_upper_bounds=None, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.

        Other Parameters
        ----------------
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Default the constraints internal values to be +/- inf.
        # This ensures the bounds checking works later in the init.
        self._constraint_lower_bounds = -np.inf
        self._constraint_upper_bounds = np.inf
        if name is None:
            name = self.__class__.__name__.lower()
        super(Recorder, self).__init__(model, name=name, **kwargs)
        self.ignore_nan = ignore_nan
        self.is_objective = is_objective
        self.epsilon = epsilon
        self.constraint_lower_bounds = constraint_lower_bounds
        self.constraint_upper_bounds = constraint_upper_bounds
        # Create the aggregator for scenarios
        self._scenario_aggregator = Aggregator(agg_func)

    property agg_func:
        """The aggregation function.
        
        **Setter:** set the aggregation function.
        """
        def __get__(self):
            return self._scenario_aggregator.func
        def __set__(self, agg_func):
            self._scenario_aggregator.func = agg_func

    property is_objective:
        """Whether the recorder is set as objective.
        
        **Setter:** the objective direction.
        """
        def __set__(self, value):
            if value is None:
                self._is_objective = ObjDirection.NONE
            else:
                self._is_objective = _obj_direction_lookup[value]
        def __get__(self):
            if self._is_objective == ObjDirection.NONE:
                return None
            elif self._is_objective == ObjDirection.MAXIMISE:
                return 'maximise'
            elif self._is_objective == ObjDirection.MINIMISE:
                return 'minimise'
            else:
                raise ValueError("Objective direction type not recognised.")

    property constraint_lower_bounds:
        """The lower bound value(s).
        
        **Setter:** set new bounds.
        """
        def __set__(self, value):
            if value is None:
                self._constraint_lower_bounds = -np.inf
            else:
                if self.constraint_upper_bounds is not None and value > self.constraint_upper_bounds:
                    raise ValueError('Lower bounds can not be larger than the upper bounds.')
                self._constraint_lower_bounds = value
        def __get__(self):
            if np.isneginf(self._constraint_lower_bounds):
                return None
            else:
                return self._constraint_lower_bounds

    property constraint_upper_bounds:
        """The upper bound value(s).
        
        **Setter:** set new bounds.
        """
        def __set__(self, value):
            if value is None:
                self._constraint_upper_bounds = np.inf
            else:
                if self.constraint_lower_bounds is not None and value < self.constraint_lower_bounds:
                    raise ValueError('Upper bounds can not be smaller than the lower bounds.')
                self._constraint_upper_bounds = value
        def __get__(self):
            if np.isinf(self._constraint_upper_bounds):
                return None
            else:
                return self._constraint_upper_bounds

    @property
    def is_equality_constraint(self):
        """This returns true if upper and lower constraint bounds are both defined and equal to one another."""
        return self.constraint_upper_bounds is not None and self.constraint_lower_bounds is not None and \
               self.constraint_lower_bounds == self.constraint_upper_bounds

    @property
    def is_double_bounded_constraint(self):
        """This returns true if upper and lower constraint bounds are both defined and not-equal to one another."""
        return self.constraint_upper_bounds is not None and self.constraint_lower_bounds is not None and \
               self.constraint_lower_bounds != self.constraint_upper_bounds

    @property
    def is_lower_bounded_constraint(self):
        """This returns true if lower constraint bounds is defined and upper constraint bounds is not."""
        return self.constraint_upper_bounds is None and self.constraint_lower_bounds is not None

    @property
    def is_upper_bounded_constraint(self):
        """This returns true if upper constraint bounds is defined and lower constraint bounds is not."""
        return self.constraint_upper_bounds is not None and self.constraint_lower_bounds is None

    @property
    def is_constraint(self):
        """This returns true if either upper or lower constraint bounds is defined."""
        return self.constraint_upper_bounds is not None or self.constraint_lower_bounds is not None

    def is_constraint_violated(self):
        """This returns true if the value from this Recorder violates its constraint bounds.

        Raises
        -------
        ValueError
            If no constraint bounds are defined (i.e. self.is_constraint == False)
        """
        value = self.aggregated_value()
        if self.is_equality_constraint:
            feasible = value == self.constraint_lower_bounds
        elif self.is_double_bounded_constraint:
            feasible = self.constraint_lower_bounds <= value <= self.constraint_upper_bounds
        elif self.is_lower_bounded_constraint:
            feasible = self.constraint_lower_bounds <= value
        elif self.is_upper_bounded_constraint:
            feasible = value <= self.constraint_upper_bounds
        else:
            raise ValueError(f'Recorder "{self.name}" has no constraint bounds defined.')
        return not feasible

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self.name)

    cpdef double aggregated_value(self) except *:
        """
        Aggregate the recorder value using `agg_func`. This returns one number to expose to an optimisation algorithm.
        
        Returns
        -------
        float
            The aggregated number.
        """
        cdef double[:] values = self.values()
        return self._scenario_aggregator.aggregate_1d(values, ignore_nan=self.ignore_nan)

    cpdef double[:] values(self) except *:
        """
        Get the values stored by the recorder.
        
        Returns
        -------
        Iterable[float]
            A memory view of the values.
        
        Raises
        ------
        NotImplementedError
            This must be implemented by a recorder inheriting from this class.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        Recorder
            The loaded class.
        """
        try:
            node_name = data["node"]
        except KeyError:
            pass
        else:
            data["node"] = model.nodes[node_name]
        return cls(model, **data)

    @classmethod
    def register(cls):
        """Register the recorder in the global register."""
        recorder_registry[cls.__name__.lower()] = cls

    @classmethod
    def unregister(cls):
        """De-register the recorder in the global register."""
        del(recorder_registry[cls.__name__.lower()])

cdef class AggregatedRecorder(Recorder):
    """This recorder provides a method to produce a complex aggregated recorder by taking
    the results of other records.

    When:

    - the `.values()` method is called, this collects the values from the provided
    recorders and aggregates the results using the `recorder_agg_func` parameter. When
    `recorder_agg_func` is omitted, this defaults to `agg_func`. This allows to, for example,
    sum all the recorder values or take the maximum values. The aggregation is done
    by scenario, which means that this recorder returns an array whose size equals the
    number of model scenarios.
    - the `.aggregated_value()` method is called, this aggregates the array of numbers
    using `agg_func`, returning one number.

    Examples
    -------
    In the model below, the recorder aggregates the arrays from the `NumpyArrayNodeRecorder`,
    which stored the time series of the data.

    ```python
    import numpy as np
    from pywr.nodes import Input, Output
    from pywr.core import Model, Scenario
    from pywr.recorders import AggregatedRecorder, NumpyArrayNodeRecorder

    model = Model()
    i = Input(model, "A", max_flow=10)
    o1 = Output(model, "B", max_flow=2, cost=-10)
    o2 = Output(model, "C", max_flow=3, cost=-10)
    i.connect(o1)
    i.connect(o2)

    node1 = NumpyArrayNodeRecorder(model, o1, temporal_agg_func="sum")
    node2 = NumpyArrayNodeRecorder(model, o2, temporal_agg_func="sum")
    rec = AggregatedRecorder(
        model=model,
        recorders=[node1, node2],
        recorder_agg_func="sum",
        agg_func="mean",
        name="Combined flow"
    )

    model.run()

    # each NumpyArrayNodeRecorder.values() returns the total delivered flow at the end of
    # the simulation (`temporal_agg_func="sum")`. The aggregated recorder then sums
    # these two numbers (`recorder_agg_func="sum"`) and returns an array with one
    # number (as the scenario size is 1).
    print(np.asarray(rec.values()))  # >> [1825.]

    # this averages the numbers in the array above (`agg_func="mean",`) and returns
    # one number. This the same number as above as the model contains one scenario.
    print(rec.aggregated_value())  # >> 1825.
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    recorders: Iterable[Recorder]
        The other `Recorder` instances to perform the aggregation over.
    recorder_agg_func : Optional[str | Callable]
        Recorder aggregation function to use when `value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, recorders, **kwargs):
        """
        Initialize the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        recorders: Iterable[Recorder]
            The other `Recorder` instances to perform the aggregation over.

        Other Parameters
        ----------------
        recorder_agg_func : Optional[str | Callable], default=agg_func
            Recorder aggregation function to use when `value()` is called.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across self.recorders scenarios
        agg_func = kwargs.pop('recorder_agg_func', kwargs.get('agg_func'))

        if isinstance(agg_func, str):
            agg_func = _agg_func_lookup[agg_func.lower()]
        elif callable(agg_func):
            self.recorder_agg_func = agg_func
            agg_func = AggFuncs.CUSTOM
        else:
            raise ValueError("Unrecognised recorder aggregation function: \"{}\".".format(agg_func))
        self._recorder_agg_func = agg_func

        super(AggregatedRecorder, self).__init__(model, **kwargs)
        self.recorders = list(recorders)

        for rec in self.recorders:
            self.children.add(rec)

    cpdef double[:] values(self) except *:
        """
        Aggregate the values from the recorders. 
        
        This collects the values from the `recorders` and aggregates the results using the 
        `_recorder_agg_func` attribute. This allows to, for example,
        sum all the recorder values or take the maximum values. The aggregation is done
        by scenario, which means that this recorder returns an array whose size equals the 
        number of model scenarios.
        
        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        cdef Recorder recorder
        cdef double[:] value, value2
        assert(len(self.recorders))
        cdef int n = len(self.model.scenarios.combinations)
        cdef int i

        if self._recorder_agg_func == AggFuncs.PRODUCT:
            value = np.ones(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] *= value2[i]
        elif self._recorder_agg_func == AggFuncs.SUM:
            value = np.zeros(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] += value2[i]
        elif self._recorder_agg_func == AggFuncs.MAX:
            value = np.empty(n)
            value[:] = -np.inf
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    if value2[i] > value[i]:
                        value[i] = value2[i]
        elif self._recorder_agg_func == AggFuncs.MIN:
            value = np.empty(n)
            value[:] = np.inf
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    if value2[i] < value[i]:
                        value[i] = value2[i]
        elif self._recorder_agg_func == AggFuncs.MEAN:
            value = np.zeros(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] += value2[i]
            for i in range(n):
                value[i] /= len(self.recorders)
        else:
            value = self.recorder_agg_func([recorder.values() for recorder in self.recorders], axis=0)
        return value

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        AggregatedRecorder
            The loaded class.
        """
        recorder_names = data.pop("recorders")
        recorders = [load_recorder(model, name) for name in recorder_names]
        rec = cls(model, recorders, **data)
        return rec

AggregatedRecorder.register()


cdef class NodeRecorder(Recorder):
    """This recorder records the flow of a node. Note that this does not return a timeseries
    but store the latest flow(s) when `.value()` is called.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NodeRecorder(
        model=model,
        name="Demanded flow",
        node=demand
    )
    ```

    JSON
    ======
    ```json
    {
        "Demanded flow": {
            "type": "NodeRecorder",
            "node": "Demand"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, AbstractNode node, name=None, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            The node instance to recorder the flow of.
        name : Optional[str], default=None
            Name of the recorder.

        Other Parameters
        ----------------
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        super(NodeRecorder, self).__init__(model, name=name, **kwargs)
        self._node = node
        node._recorders.append(self)

    cpdef double[:] values(self) except *:
        """
        Get the node's flow.
        
        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._node._flow

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

NodeRecorder.register()


cdef class StorageRecorder(Recorder):
    """This recorder records the absolute volume of a storage node. Note that this does not return a timeseries
    but store the latest storage(s) when `.value()` is called.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import StorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    StorageRecorder(
        model=model,
        name="Last volume",
        node=storage
    )
    ```

    JSON
    ======
    ```json
    {
        "Last volume": {
            "type": "StorageRecorder",
            "node": "Reservoir"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, AbstractStorage node, name=None, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            The storage instance to recorder the volume of.
        name : Optional[str], default=None
            Name of the recorder.

        Other Parameters
        ----------------
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """

        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        super(StorageRecorder, self).__init__(model, name=name, **kwargs)
        self._node = node
        node._recorders.append(self)

    cpdef double[:] values(self) except *:
        """
        Get the node's absolute volume.
        
        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._node._volume

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

StorageRecorder.register()


cdef class ParameterRecorder(Recorder):
    """Base class for recorders that track `Parameter` values.

    Parameters
    ----------
    model : `pywr.core.Model`
    param : `pywr.parameters.Parameter`
        The parameter to record.
    name : str (optional)
        The name of the recorder
    """
    def __init__(self, model, Parameter param, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        super(ParameterRecorder, self).__init__(model, name=name, **kwargs)
        self._param = param
        param.parents.add(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)

    @classmethod
    def load(cls, model, data):
        # when the parameter being recorder is defined inline (i.e. not in the
        # parameters section, but within the node) we need to make sure the
        # node has been loaded first
        try:
            node_name = data["node"]
        except KeyError:
            node = None
        else:
            del(data["node"])
            node = model.nodes[node_name]
        from pywr.parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)

ParameterRecorder.register()


cdef class IndexParameterRecorder(Recorder):
    def __init__(self, model, IndexParameter param, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        super(IndexParameterRecorder, self).__init__(model, name=name, **kwargs)
        self._param = param
        param.parents.add(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)

IndexParameterRecorder.register()


cdef class NumpyArrayNodeRecorder(NodeRecorder):
    """This recorder recorders a timeseries of a [pywr.core.Node]'s flow
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeRecorder(
        model=model,
        name="Demanded flow",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Demanded flow": {
            "type": "NumpyArrayNodeRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.

    See also
    --------
    [pywr.recorders.NumpyArrayNodeDeficitRecorder][]
    [pywr.recorders.NumpyArrayNodeSuppliedRatioRecorder][]
    [pywr.recorders.NumpyArrayNodeCurtailmentRatioRecorder][]
    """
    def __init__(self, model, AbstractNode node, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the median flow for a scenario.
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        factor = kwargs.pop('factor', 1.0)
        super(NumpyArrayNodeRecorder, self).__init__(model, node, **kwargs)
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.factor = factor

    property temporal_agg_func:
        """The temporal aggregation function used in `values()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variables."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        """Reset the internal variables."""
        self._data[:, :] = 0.0

    cpdef after(self):
        """Calculate the total flow and scale it with `factor`."""
        cdef int i
        cdef Timestep ts = self.model.timestepper.current
        for i in range(self._data.shape[1]):
            self._data[ts.index, i] = self._node._flow[i]*self.factor
        return 0

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the node's flow."""
        def __get__(self, ):
            return np.array(self._data)

    cpdef double[:] values(self) except *:
        """
        Compute a value for each scenario using `temporal_agg_func`.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Get a `pandas.DataFrame` of the recorder data.

        Returns
        -------
        pandas DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The model
            timesteps are in the row index.
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

NumpyArrayNodeRecorder.register()


cdef class NumpyArrayNodeDeficitRecorder(NumpyArrayNodeRecorder):
    """This recorder recorders a timeseries of a [pywr.core.Node]'s deficit
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    The deficit is calculated as the difference between the value in the node's `max_flow`
    attribute and the flow allocated during the time-step in `flow`:

        deficit = max_flow - actual_flow

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeDeficitRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeDeficitRecorder(
        model=model,
        name="Demand deficit",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Demand deficit": {
            "type": "NumpyArrayNodeDeficitRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a

    See also
    --------
    NumpyArrayNodeRecorder
    NumpyArrayNodeSuppliedRatioRecorder
    NumpyArrayNodeCurtailmentRatioRecorder
    """
    cpdef after(self):
        """Calculate the deficit."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Node node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            self._data[ts.index,scenario_index.global_id] = max_flow - node._flow[scenario_index.global_id]
        return 0
NumpyArrayNodeDeficitRecorder.register()


cdef class NumpyArrayNodeSuppliedRatioRecorder(NumpyArrayNodeRecorder):
    """This recorder recorders a timeseries of a [pywr.core.Node]'s supply ratio
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    The supply ratio is calculated as the ratio of the flow allocated during the time-step in `flow`
    and the node's `max_flow` attribute:

        supply_ratio = actual_flow / max_flow

    If the node's max_flow returns zero, then the ratio is recorded as 1.0.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeSuppliedRatioRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeSuppliedRatioRecorder(
        model=model,
        name="Demand supply ratio",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Demand supply ratio": {
            "type": "NumpyArrayNodeSuppliedRatioRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a

    See also
    --------
    NumpyArrayNodeRecorder
    NumpyArrayNodeDeficitRecorder
    NumpyArrayNodeCurtailmentRatioRecorder
    """
    cpdef after(self):
        """Calculate the supply ratio."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Node node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            try:
                self._data[ts.index,scenario_index.global_id] = node._flow[scenario_index.global_id] / max_flow
            except ZeroDivisionError:
                self._data[ts.index,scenario_index.global_id] = 1.0
        return 0
NumpyArrayNodeSuppliedRatioRecorder.register()


cdef class NumpyArrayNodeCurtailmentRatioRecorder(NumpyArrayNodeRecorder):
    """This recorder recorders a timeseries of a [pywr.core.Node]'s curtailment ratio
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    The curtailment ratio is calculated as one minus the ratio of the flow allocated during the time-step in `flow`
    and the `max_flow` attribute:

        curtailment_ratio = 1 - actual_flow / max_flow

    If the node's `max_flow` returns zero, then the curtailment ratio is recorded as 0.0.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeCurtailmentRatioRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeCurtailmentRatioRecorder(
        model=model,
        name="Demand curtailment ratio",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Demand curtailment ratio": {
            "type": "NumpyArrayNodeCurtailmentRatioRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a

    See also
    --------
    NumpyArrayNodeRecorder
    NumpyArrayNodeDeficitRecorder
    NumpyArrayNodeSuppliedRatioRecorder
    """
    cpdef after(self):
        """Calculate the curtailment ratio."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Node node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            try:
                self._data[ts.index,scenario_index.global_id] = 1.0 - node._flow[scenario_index.global_id] / max_flow
            except ZeroDivisionError:
                self._data[ts.index,scenario_index.global_id] = 0.0
NumpyArrayNodeCurtailmentRatioRecorder.register()


cdef class NumpyArrayNodeCostRecorder(NumpyArrayNodeRecorder):
    """This recorder stores the timeseries of unit cost from a `Node`. The data is
    saved internally using a memory view. The data can be accessed through the `data` attribute or
    `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeCostRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeCostRecorder(
        model=model,
        name="Output cost",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Output cost": {
            "type": "NumpyArrayNodeCostRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.

    See also
    --------
    NumpyArrayNodeRecorder
    """
    cpdef after(self):
        """Calculate the cost."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Node node = self._node
        for scenario_index in self.model.scenarios.combinations:
            self._data[ts.index, scenario_index.global_id] = node.get_cost(scenario_index)
NumpyArrayNodeCostRecorder.register()


cdef class FlowDurationCurveRecorder(NumpyArrayNodeRecorder):
    """This recorder calculates a flow duration curve for each scenario from a node's flow.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import FlowDurationCurveRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    FlowDurationCurveRecorder(
        model=model,
        name="FDC demand",
        node=demand,
        percentiles=np.arange(1, 101, 0.5),
        temporal_agg_func="mean"
    )
    ```

    JSON
    ======
    ```json
    {
        "FDC demand": {
            "type": "FlowDurationCurveRecorder",
            "node": "Demand",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "temporal_agg_func": "mean"
        }
    }
    ```

    !!!note "Aggregation"
        When you call the `value()` method, the recorder aggregates the flow duration curve over
        the percentiles using the function specified in the `temporal_agg_func` attribute.

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the
        `value()` method.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, AbstractNode node, percentiles, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.
        percentiles : Iterable[float]
            The percentiles to use in the calculation of the flow duration curve.
            Values must be in the range 0-100.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the median exceeded flow for a scenario.
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across percentiles
        if 'fdc_agg_func' in kwargs:
            # Support previous behaviour
            warnings.warn('The "fdc_agg_func" key is deprecated for defining the temporal '
                          'aggregation in {}. Please "temporal_agg_func" instead.'
                          .format(self.__class__.__name__))
            if "temporal_agg_func" in kwargs:
                raise ValueError('Both "fdc_agg_func" and "temporal_agg_func" keywords given.'
                                 'This is ambiguous. Please use "temporal_agg_func" only.')
            kwargs["temporal_agg_func"] = kwargs.pop("fdc_agg_func")

        super(FlowDurationCurveRecorder, self).__init__(model, node, **kwargs)
        self._percentiles = np.asarray(percentiles, dtype=np.float64)

    cpdef finish(self):
        """Calculate the flow duration curve."""
        self._fdc = np.percentile(np.asarray(self._data), np.asarray(self._percentiles), axis=0)

    property fdc:
        """Get the flow duration curve."""
        def __get__(self, ):
            return np.array(self._fdc)

    cpdef double[:] values(self) except *:
        """
        Compute a value for each scenario by aggregating over the percentiles using the `temporal_agg_func`.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._temporal_aggregator.aggregate_2d(self._fdc, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data.

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = np.array(self._percentiles)
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self.fdc), index=index, columns=sc_index)

FlowDurationCurveRecorder.register()


cdef class SeasonalFlowDurationCurveRecorder(FlowDurationCurveRecorder):
    """This recorder calculates a flow duration curve for each scenario for a given season
    specified as months.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import SeasonalFlowDurationCurveRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    SeasonalFlowDurationCurveRecorder(
        model=model,
        name="FDC demand",
        node=demand,
        percentiles=np.arange(1, 101, 0.5),
        months=[7, 8, 9],
        temporal_agg_func="mean"
    )
    ```

    JSON
    ======
    ```json
    {
        "FDC demand": {
            "type": "SeasonalFlowDurationCurveRecorder",
            "node": "Demand",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "months": [7, 8, 9],
            "temporal_agg_func": "mean"
        }
    }
    ```

    !!!note "Aggregation"
        When you call the `value()` method, the recorder aggregates the flow duration curve over
        the percentiles using the function specified in the `temporal_agg_func` attribute.

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the
        `value()` method.
    months: Iterable[int]
        The values of the months the flow duration curve should be calculated for.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.

    Parameters
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    percentiles : array
        The percentiles to use in the calculation of the flow duration curve.
        Values must be in the range 0-100.
    agg_func: str, optional
        function used for aggregating the FDC across percentiles.
        Numpy style functions that support an axis argument are supported.
    fdc_agg_func: str, optional
        optional different function for aggregating across scenarios.
    """

    def __init__(self, model, AbstractNode node, percentiles, months, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.
        percentiles : Iterable[float]
            The percentiles to use in the calculation of the flow duration curve.
            Values must be in the range 0-100.
        months: Iterable[int]
            The values of the months the flow duration curve should be calculated for.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the median exceeded flow for a scenario.
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        super(SeasonalFlowDurationCurveRecorder, self).__init__(model, node, percentiles, **kwargs)
        self._months = set(months)

    cpdef finish(self):
        """Calculate the flow duration curve."""
        # this is a def method rather than cpdef because closures inside cpdef functions are not supported yet.
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        df = pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
        mask = np.asarray(df.index.map(self.is_season))
        self._fdc = np.percentile(df.loc[mask, :], np.asarray(self._percentiles), axis=0)

    def is_season(self, x):
        """Whether the given month is in `self._months`

        Parameters
        ----------
        x : int
            The month.

        Returns
        -------
        bool
        """
        return x.month in self._months

SeasonalFlowDurationCurveRecorder.register()

cdef class FlowDurationCurveDeviationRecorder(FlowDurationCurveRecorder):
    """This recorder calculates a flow duration curves (FDC) for each scenario and then
    calculates their deviation from an upper and lower target FDCs using the following steps:

    For each percentile, the recorder calculates the difference between the flow duration curve
    of a node and a user-defined upper (`upper_target_fdc`) and/or lower target (`lower_target_fdc`) curves divided
    by the target.

    For the upper target, the deviation for one scenario is calculated as:

        (fdc[k] - upper_target_fdc[k]) / upper_target_fdc[k]

    where `k` is the percentile.

    For the upper target, the deviation for one scenario is calculated as:

        (lower_target_fdc[k] - fdc[k]) / lower_target_fdc[k]

    The shape of the target arrays depend whether you provide `scenario`:

    - When `scenario` is `None`, this can either be a 1D array of size equal to `percentiles`
      or a 2D array where the shape is (scenario_size, percentile_size).
    - If `scenario` is given, then this must be a 2D array where the shape is (scenario_size,
      percentile_size).

    If you provide one target curve, the deviation is calculated only using the provided target. If you provide both the lower and upper
    target curves, the overall deviation is the worst of the upper and lower difference.
    The deviation is positive if the node's FDC is above the upper target or below the lower
    target. If the FDC falls between the upper and lower targets, the deviation is zero.

    The 2nd dimension of the target duration curves must equal the size of the percentile list
    have the same order (high to low values or low to high values).

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import FlowDurationCurveDeviationRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    FlowDurationCurveDeviationRecorder(
        model=model,
        name="FDC deviation",
        node=demand,
        percentiles=np.array([1, 5, 20, 40, 60, 80, 100]),
        lower_target_fdc=np.array([100, 80, 56, 51, 43, 23, 12]),
        temporal_agg_func="mean"
    )
    ```

    JSON
    ======
    ```json
    {
        "FDC deviation": {
            "type": "FlowDurationCurveDeviationRecorder",
            "node": "Demand",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "lower_target_fdc": [100, 80, 56, 51, 43, 23, 12],
            "temporal_agg_func": "mean"
        }
    }
    ```

    !!!note "Aggregation"
        When you call the `value()` method, the recorder aggregates the flow duration curve over
        the percentiles using the function specified in the `temporal_agg_func` attribute.

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate the FDC deviations over the percentiles when computing a value per scenario in the
        `value()` method.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, AbstractNode node, percentiles, lower_target_fdc=None, upper_target_fdc=None, scenario=None, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.
        percentiles : Iterable[float]
            The percentiles to use in the calculation of the flow duration curve.
            Values must be in the range 0-100.
        lower_target_fdc : Optional[Iterable[float]], default=None
            The lower FDC against which the scenario FDCs are compared. When `scenario` is `None`, this can be
            a 1D array of size equal to `percentiles` or a 2D array where the shape is (scenario_size, percentile_size).
            If `scenario` is given, then this must be a 2D array where the shape is (scenario_size, percentile_size).
            If this is not provided, then deviations from a lower target FDC are recorded as 0.0. If targets are loaded from an external file, this needs to be indexed using
            the percentile values.
        upper_target_fdc : Optional[Iterable[float]], default=None
            The upper FDC against which the scenario FDCs are compared.  When `scenario` is `None`, this can be
            a 1D array of size equal to `percentiles` or a 2D array where the shape is (scenario_size, percentile_size).
            If `scenario` is given, then this must be a 2D array where the shape is (scenario_size, percentile_size).
            If values are not provided, then deviations from a upper target FDC are recorded as 0.0. If targets are loaded from an external file this needs to be indexed using
            the percentile values.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate the FDC deviations over the percentiles when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean deviation for a scenario.
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.

        Raises
        ------
        ValueError
            If `lower_target_fdc` or `upper_target_fdc` is not provided or when thw two arguments
            do not match the length of `percentiles`. When the first dimension of
            `lower_target_fdc` or `upper_target_fdc` does not match the scenario size (when `scenario` is
             given) or the model scenario combinations.
        """
        super(FlowDurationCurveDeviationRecorder, self).__init__(model, node, percentiles, **kwargs)

        if lower_target_fdc is None and upper_target_fdc is None:
            raise ValueError("At least one of the 'lower_target_fdc' and 'lower_target_fdc' arguments should be provided")

        if lower_target_fdc is None:
            lower_target = None
        else:
            lower_target = np.array(lower_target_fdc, dtype=np.float64)
            if lower_target.ndim < 2:
                lower_target = lower_target[:, np.newaxis]

        if upper_target_fdc is None:
            upper_target = None
        else:
            upper_target = np.array(upper_target_fdc, dtype=np.float64)
            if upper_target.ndim < 2:
                upper_target = upper_target[:, np.newaxis]

        self._lower_target_fdc = lower_target
        self._upper_target_fdc = upper_target
        self.scenario = scenario
        if self._lower_target_fdc is not None:
            if len(self._percentiles) != self._lower_target_fdc.shape[0]:
                raise ValueError("The lengths of the lower target FDC and the percentiles list do not match")
        if self._upper_target_fdc is not None:
            if len(self._percentiles) != self._upper_target_fdc.shape[0]:
                raise ValueError("The lengths of the upper target FDC and the percentiles list do not match")

    cpdef setup(self):
        """Check the size of the FDC targets.
        
        Raises
        ------
        ValueError
            If the first dimension of the targets does not match the scenario size, when
            `scenario` is given. Or when `scenario` is `None`, but the target is a 2D array,
            the first dimension of the targets does not match the scenario combination length.
        """
        super(FlowDurationCurveDeviationRecorder, self).setup()
        # Check target FDC is the correct size; this is done in setup rather than __init__
        # because the scenarios might change after the Recorder is created.
        if self.scenario is not None:
            if self._lower_target_fdc is not None:
                if self._lower_target_fdc.shape[1] != self.scenario.size:
                    raise ValueError('The number of lower target FDCs does not match the size ({}) of scenario "{}"'.format(self.scenario.size, self.scenario.name))
            if self._upper_target_fdc is not None:
                if self._upper_target_fdc.shape[1] != self.scenario.size:
                    raise ValueError('The number of upper target FDCs does not match the size ({}) of scenario "{}"'.format(self.scenario.size, self.scenario.name))
        else:
            if self._lower_target_fdc is not None:
                if self._lower_target_fdc.shape[1] > 1 and \
                        self._lower_target_fdc.shape[1] != len(self.model.scenarios.combinations):
                    raise ValueError("The number of lower target FDCs does not match the number of scenarios")
            if self._upper_target_fdc is not None:
                if self._upper_target_fdc.shape[1] > 1 and \
                        self._upper_target_fdc.shape[1] != len(self.model.scenarios.combinations):
                    raise ValueError("The number of upper target FDCs does not match the number of scenarios")

    cpdef finish(self):
        """Calculate the deviations."""
        super(FlowDurationCurveDeviationRecorder, self).finish()

        cdef int i, j, jl, ju, k, sc_index
        cdef ScenarioIndex scenario_index
        cdef double[:] utrgt_fdc, ltrgt_fdc
        cdef double udev, ldev

        # We have to do this the slow way by iterating through all scenario combinations
        if self.scenario is not None:
            sc_index = self.model.scenarios.get_scenario_index(self.scenario)

        self._fdc_deviations = np.empty((len(self._percentiles), len(self.model.scenarios.combinations)), dtype=np.float64)
        for i, scenario_index in enumerate(self.model.scenarios.combinations):

            if self.scenario is not None:
                # Get the scenario specific ensemble id for this combination
                j = scenario_index._indices[sc_index]
            else:
                j = scenario_index.global_id

            if self._lower_target_fdc is not None:
                if self._lower_target_fdc.shape[1] == 1:
                    jl = 0
                else:
                    jl = j
                # Cache the target FDC to use in this combination
                ltrgt_fdc = self._lower_target_fdc[:, jl]


            if self._upper_target_fdc is not None:
                if self._upper_target_fdc.shape[1] == 1:
                    ju = 0
                else:
                    ju = j
                # Cache the target FDC to use in this combination
                utrgt_fdc = self._upper_target_fdc[:, ju]

            # Finally calculate deviation
            for k in range(len(self._percentiles)):
                try:
                    # upper deviation (+ve when flow higher than upper target)
                    if self._upper_target_fdc is not None:
                        udev = (self._fdc[k, i] - utrgt_fdc[k])  / utrgt_fdc[k]
                    else:
                        udev = 0.0
                    # lower deviation (+ve when flow less than lower target)
                    if self._lower_target_fdc is not None:
                        ldev = (ltrgt_fdc[k] - self._fdc[k, i])  / ltrgt_fdc[k]
                    else:
                        ldev = 0.0
                    # Overall deviation is the worst of upper and lower, but if both
                    # are negative (i.e. FDC is between upper and lower) there is zero deviation
                    self._fdc_deviations[k, i] = max(udev, ldev, 0.0)
                except ZeroDivisionError:
                    self._fdc_deviations[k, i] = np.nan

    property fdc_deviations:
        """Get the flow duration curve deviations as numpy array."""
        def __get__(self, ):
            return np.array(self._fdc_deviations)


    cpdef double[:] values(self) except *:
        """
        Compute a value for each scenario by aggregating over the percentiles using the `temporal_agg_func`.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._temporal_aggregator.aggregate_2d(self._fdc_deviations, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self, return_fdc=False):
        """Return a `pandas.DataFrame` of the deviations from the target FDCs.

        Parameters
        ----------
        return_fdc : Optional[bool], default=False
            If True returns a tuple of two dataframes. The first is the deviations, the second
            is the actual FDC. Otherwise, this returns the DataFrame.

        Returns
        -------
        pandas.DataFrame | tuple[pandas.DataFrame, pandas.DataFrame]
            The DataFrame with the FDC deviations or a tuple with the deviation and actual FDC
            DataFrames, if `return_fdc` is `True`.
        """
        index = np.array(self._percentiles)
        sc_index = self.model.scenarios.multiindex

        df = pd.DataFrame(data=np.array(self._fdc_deviations), index=index, columns=sc_index)
        if return_fdc:
            return df, super(FlowDurationCurveDeviationRecorder, self).to_dataframe()
        else:
            return df

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        FlowDurationCurveDeviationRecorder
            The loaded class.
        """
        node = model.nodes[data.pop("node")]
        percentiles = data.pop("percentiles")
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        from pywr.parameters import load_parameter_values
        upper_target_fdc = data.pop("upper_target_fdc", None)
        if isinstance(upper_target_fdc, dict):
            upper_target_fdc.update({"indexes": percentiles})
            upper_target_fdc = load_parameter_values(model, upper_target_fdc)
        lower_target_fdc = data.pop("lower_target_fdc", None)
        if isinstance(lower_target_fdc, dict):
            lower_target_fdc.update({"indexes": percentiles})
            lower_target_fdc = load_parameter_values(model, lower_target_fdc)
        return cls(model, node, percentiles, lower_target_fdc=lower_target_fdc, upper_target_fdc=upper_target_fdc,
                   **data)

FlowDurationCurveDeviationRecorder.register()


cdef class NumpyArrayAbstractStorageRecorder(StorageRecorder):
    def __init__(self, model, AbstractStorage node, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage instance to record.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean storage for a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, node, **kwargs)

        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variable."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        """Reset the internal variable."""
        self._data[:, :] = 0.0

    cpdef after(self):
        raise NotImplementedError()

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the storage's data."""
        def __get__(self, ):
            return np.array(self._data)

    cpdef double[:] values(self) except *:
        """Compute a value for each scenario using `temporal_agg_func`."""
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the timesteps.
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)


cdef class NumpyArrayStorageRecorder(NumpyArrayAbstractStorageRecorder):
    """This recorder recorders a timeseries of a [pywr.nodes.Storage]'s volume
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayStorageRecorder(
        model=model,
        name="Relative storage",
        node=storage,
        proportional=True,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Relative storage": {
            "type": "NumpyArrayStorageRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
            "proportional": true,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    proportional : bool
        Whether to record proportional [0, 1.0] or absolute storage volumes.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage instance to record.
        proportional : Optional[bool], default=False
            Whether to record proportional [0, 1.0] or absolute storage volumes.
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean storage for a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        self.proportional = kwargs.pop('proportional', False)
        super().__init__(*args, **kwargs)

    cpdef after(self):
        """Calculate the storage."""
        cdef int i
        cdef Timestep ts = self.model.timestepper.current
        for i in range(self._data.shape[1]):
            if self.proportional:
                self._data[ts.index, i] = self._node._current_pc[i]
            else:
                self._data[ts.index, i] = self._node._volume[i]
        return 0
NumpyArrayStorageRecorder.register()


cdef class NumpyArrayNormalisedStorageRecorder(NumpyArrayAbstractStorageRecorder):
    """This recorder stores the storage's volume and normalised the volume relative to a user-defined
    control curve. The data is normalised such that values of 1, 0 and -1 align with full, at control
    curve and empty volumes respectively. The data is saved internally using a
    memory view. The data can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ConstantParameter
    from pywr.recorders import NumpyArrayNormalisedStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    NumpyArrayNormalisedStorageRecorder(
        model=model,
        name="Normalised 80% storage",
        node=storage,
        parameter=ConstantParameter(model, 0.8),
    )
    ```

    JSON
    ======
    ```json
    {
        "Normalised 80% storage": {
            "type": "NumpyArrayNormalisedStorageRecorder",
            "node": "Reservoir",
            "parameter": 0.8
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    parameter : Parameter
        The control curve parameter to use to normalise the storage between -1.0 and 1.0.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):

        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage instance to record.
        parameter : Parameter
            The control curve parameter to use to normalise the storage between -1.0 and 1.0.
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean normalised storage for a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """

        # Optional different method for aggregating across time.
        self.parameter = kwargs.pop('parameter')
        super().__init__(*args, **kwargs)
        self.children.add(self.parameter)

    cpdef after(self):
        """Calculate the normalised storage."""
        cdef int i
        cdef Timestep ts = self.model.timestepper.current
        cdef double[:] values = self.parameter.get_all_values()
        cdef double vol, cc, norm_vol

        for i in range(self._data.shape[1]):
            vol = self._node._current_pc[i]
            cc = values[i]

            if vol < cc:
                # Lower than control curve; value between -1.0 and 0.0
                norm_vol = vol / cc - 1.0
            else:
                # At or above control curve; value between 0.0 and 1.0
                norm_vol = (vol - cc) / (1.0 - cc)

            self._data[ts.index, i] = norm_vol

        return 0

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        NumpyArrayNormalisedStorageRecorder
            The loaded class.
        """
        from pywr.parameters import load_parameter
        node = model.nodes[data.pop("node")]
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, node=node, parameter=parameter, **data)
NumpyArrayNormalisedStorageRecorder.register()


cdef class StorageDurationCurveRecorder(NumpyArrayStorageRecorder):
    """This recorder calculates a storage duration curve for each scenario.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import StorageDurationCurveRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    StorageDurationCurveRecorder(
        model=model,
        name="FDC storage",
        node=storage,
        percentiles=np.arange(1, 101, 0.5),
        temporal_agg_func="mean"
    )
    ```

    JSON
    ======
    ```json
    {
        "FDC storage": {
            "type": "StorageDurationCurveRecorder",
            "node": "Reservoir",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "temporal_agg_func": "mean"
        }
    }
    ```

    !!!note "Aggregation"
        When you call the `value()` method, the recorder aggregates the storage duration curve over
        the percentiles using the function specified in the `temporal_agg_func` attribute.

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage node instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate the SDCs over the percentiles when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def __init__(self, model, AbstractStorage node, percentiles, **kwargs):

        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage node instance to record.
        percentiles : Iterable[float]
            The percentiles to use in the calculation of the storage duration curve.
            Values must be in the range 0-100.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate the SDCs over the percentiles when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean storage over a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        if "sdc_agg_func" in kwargs:
            # Support previous behaviour
            warnings.warn('The "sdc_agg_func" key is deprecated for defining the temporal '
                          'aggregation in {}. Please "temporal_agg_func" instead.'
                          .format(self.__class__.__name__))
            if "temporal_agg_func" in kwargs:
                raise ValueError('Both "sdc_agg_func" and "temporal_agg_func" keywords given.'
                                 'This is ambiguous. Please use "temporal_agg_func" only.')
            kwargs["temporal_agg_func"] = kwargs.pop("sdc_agg_func")

        super(StorageDurationCurveRecorder, self).__init__(model, node, **kwargs)
        self._percentiles = np.asarray(percentiles, dtype=np.float64)


    cpdef finish(self):
        """Calculate the storage duration curve."""
        self._sdc = np.percentile(np.asarray(self._data), np.asarray(self._percentiles), axis=0)

    property sdc:
        """Get the storage duration curve."""
        def __get__(self, ):
            return np.array(self._sdc)

    cpdef double[:] values(self) except *:
        """
        Compute a value for each scenario by aggregating over the percentiles using the `temporal_agg_func`.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._temporal_aggregator.aggregate_2d(self._sdc, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the percentiles.
        """
        index = np.array(self._percentiles)
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=self.sdc, index=index, columns=sc_index)

StorageDurationCurveRecorder.register()

cdef class NumpyArrayLevelRecorder(NumpyArrayAbstractStorageRecorder):
    """This recorder recorders a timeseries of a [pywr.nodes.Storage]'s level
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayLevelRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayLevelRecorder(
        model=model,
        name="Level",
        node=storage,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Level": {
            "type": "NumpyArrayLevelRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    cpdef after(self):
        """Calculate the level."""
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Storage node = self._node
        for i, scenario_index in enumerate(self.model.scenarios.combinations):
            self._data[ts.index, i] = node.get_level(scenario_index)
        return 0
NumpyArrayLevelRecorder.register()


cdef class NumpyArrayAreaRecorder(NumpyArrayAbstractStorageRecorder):
    """This recorder recorders a timeseries of a [pywr.nodes.Storage]'s area
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayAreaRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayAreaRecorder(
        model=model,
        name="Area",
        node=storage,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Area": {
            "type": "NumpyArrayAreaRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    cpdef after(self):
        """Calculate the level."""
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef Storage node = self._node
        for i, scenario_index in enumerate(self.model.scenarios.combinations):
            self._data[ts.index, i] = node.get_area(scenario_index)
        return 0
NumpyArrayAreaRecorder.register()


cdef class NumpyArrayParameterRecorder(ParameterRecorder):
    """This recorder recorders a timeseries of a parameter's value
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveParameter, ConstantParameter
    from pywr.recorders import NumpyArrayParameterRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    parameter = ControlCurveParameter(
        model=model,
        name="Rule curve position",
        storage_node=storage,
        values=[1.0, 45.0, 90.0],
        control_curves=[ConstantParameter(model, 0.76), ConstantParameter(model, 0.56)],
    )
    NumpyArrayParameterRecorder(
        model=model,
        name="Control curve value",
        param=parameter,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Control curve value": {
            "type": "NumpyArrayParameterRecorder",
            "param": "Rule curve position",
            "temporal_agg_func": "sum",
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    param : Parameter
        Parameter instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, Parameter param, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            Parameter instance to record.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean parameter value for a scenario,
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super(NumpyArrayParameterRecorder, self).__init__(model, param, **kwargs)

        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variable."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        """Reset the internal variable."""
        self._data[:, :] = 0.0

    cpdef after(self):
        """Save the data."""
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        self._data[ts.index, :] = self._param.get_all_values()
        return 0

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the parameter's values."""
        def __get__(self, ):
            return np.array(self._data)

    cpdef double[:] values(self) except *:
        """Compute a value for each scenario using `temporal_agg_func`."""
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the timesteps.
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
NumpyArrayParameterRecorder.register()


cdef class NumpyArrayDailyProfileParameterRecorder(ParameterRecorder):
    """This recorder recorders an annual profile parameter and stores 366 values per scenario.

    For each day of the year, it stores the value encountered for that day during a simulation. This recorder is
    useful for returning the daily profile that may result from the combination of one or more parameters. For
    example, during optimisation of new profiles non-daily parameters (e.g. [pywr.parameters.RbfProfileParameter][]) and/or
    aggregations of several parameters might be used. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    !!!info
        This differs from a [pywr.recorders.NumpyArrayParameterRecorder][] as it does not
        store the profile for each timestep, but it stores 366 values only.

    Attributes
    ----------
    model : Model
        The model instance.
    param : Parameter
        Parameter instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, Parameter param, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            Parameter instance to record.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the median value for a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, param, **kwargs)

        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variable."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        self._data = np.zeros((366, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef after(self):
        """Save the data."""
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef int i = ts.dayofyear_index
        self._data[i, :] = self._param.get_all_values()
        return 0

    property data:
        """This contains an array with shape (366, number_of_scenarios) with
        the profile data."""
        def __get__(self, ):
            return np.array(self._data)

    cpdef double[:] values(self) except *:
        """Compute a value for each scenario using `temporal_agg_func`."""
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains a number between 1 and 366.
        """
        index = np.arange(1, 367)
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
NumpyArrayDailyProfileParameterRecorder.register()


cdef class NumpyArrayIndexParameterRecorder(IndexParameterRecorder):
    """This recorder recorders a timeseries of a parameter's index
    for each time-step and scenario. The data is saved internally using a
    memory view and can be accessed through the `data` attribute or `to_dataframe()` method.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveIndexParameter, ConstantParameter
    from pywr.recorders import NumpyArrayIndexParameterRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    parameter = ControlCurveIndexParameter(
        model=model,
        name="Rule curve index",
        storage_node=storage,
        control_curves=[ConstantParameter(model, 0.76), ConstantParameter(model, 0.56)],
    )
    NumpyArrayIndexParameterRecorder(
        model=model,
        name="Control curve value",
        param=parameter,
        temporal_agg_func="sum"
    )
    ```

    JSON
    ======
    ```json
    {
        "Control curve value": {
            "type": "NumpyArrayIndexParameterRecorder",
            "param": "Rule curve index",
            "temporal_agg_func": "sum",
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    param : IndexParameter
        IndexParameter instance to record.
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, IndexParameter param, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : IndexParameter
            IndexParameter instance to record.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the mean index for a scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super(NumpyArrayIndexParameterRecorder, self).__init__(model, param, **kwargs)

        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variable."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb), dtype=np.int32)

    cpdef reset(self):
        """Reset the internal variable."""
        self._data[:, :] = 0

    cpdef after(self):
        """Save the data."""
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        self._data[ts.index, :] = self._param.get_all_indices()
        return 0

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the parameter's index."""
        def __get__(self, ):
            return np.array(self._data)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the timesteps.
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
NumpyArrayIndexParameterRecorder.register()


cdef class RollingWindowParameterRecorder(ParameterRecorder):
    """This recorder calculates the rolling value of a [pywr.parameters.Parameter][] for the
    last N timesteps. The metric to use over the rolling window can be provided in `temporal_agg_func`.


    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import RollingWindowParameterRecorder

    model = Model()
    river = Link(model=model, name="River")
    parameter = NodeThresholdParameter(
        model=model,
        node=river,
        predicate="LT",
        values=[10, 2],
        threshold=1000,
        name="Daily river-dependant license"
    )
    RollingWindowParameterRecorder(
        model=model,
        name="Mean max abstraction",
        param=parameter,
        window=4
    )
    ```

    JSON
    ======
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean max abstraction": {
            "type": "RollingWindowParameterRecorder",
            "parameter": "Daily river-dependant license",
            "window": 4
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    window : int
        The number of timestep to use to calculate the rolling window.
    temporal_agg_func : str | Callable
        The function to use to aggregate the values of `param` over the rolling window.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def __init__(self, model, Parameter param, int window, *args, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            The parameter instance to use to get the value.
        window : int
            The number of timestep to use to calculate the rolling window.

        Other parameters
        ----------------
        temporal_agg_func : Optional[str | Callable], default="mean"
            The function to use to aggregate the values of `param` over the rolling window. For example,
            "mean" to calculate the rolling mean.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        if "agg_func" in kwargs and "temporal_agg_func" not in kwargs:
            # Support previous behaviour
            warnings.warn('The "agg_func" key is deprecated for defining the temporal '
                          'aggregation in {}. Please "temporal_agg_func" instead.'
                          .format(self.__class__.__name__))
            temporal_agg_func = kwargs.get("agg_func")
        else:
            temporal_agg_func = kwargs.pop("temporal_agg_func", "mean")

        super(RollingWindowParameterRecorder, self).__init__(model, param, *args, **kwargs)
        self.window = window
        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.after()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variable."""
        cdef int ncomb = len(self.model.scenarios.combinations)
        cdef int nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb,), np.float64)
        self._memory = np.empty((nts, ncomb,), np.float64)
        self.position = 0

    cpdef reset(self):
        """Reset the internal variable."""
        self._data[...] = 0
        self.position = 0

    cpdef after(self):
        """Calculate the rolling mean after aggregating with `temporal_agg_func`"""
        cdef int i, n
        cdef double[:] value
        cdef ScenarioIndex scenario_index
        cdef Timestep timestep = self.model.timestepper.current

        for i, scenario_index in enumerate(self.model.scenarios.combinations):
            self._memory[self.position, i] = self._param.get_value(scenario_index)

        if timestep.index < self.window:
            n = timestep.index + 1
        else:
            n = self.window

        value = self._temporal_aggregator.aggregate_2d(self._memory[0:n, :], axis=0)
        self._data[timestep.index, :] = value

        self.position += 1
        if self.position >= self.window:
            self.position = 0

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the data."""
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the timesteps.
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=self.data, index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        RollingWindowParameterRecorder
            The loaded class.
        """
        from pywr.parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        window = int(data.pop("window"))
        return cls(model, parameter, window, **data)

RollingWindowParameterRecorder.register()

cdef class RollingMeanFlowNodeRecorder(NodeRecorder):
    """This records the mean flow of a Node for previous `N` timesteps or number of days.

    !!!danger "Deprecated"
        This recorder has been deprecated in favour of [pywr.parameters.RollingMeanFlowNodeParameter][].
        If you need to record the value, use a recorder capable of recording an arbitrary parameter, such as
        [pywr.recorders.NumpyArrayParameterRecorder][].
    """
    def __init__(self, model, node, timesteps=None, days=None, name=None, **kwargs):
        warnings.warn("`RollingMeanFlowNodeRecorder` has been deprecated in favour of `RollingMeanFlowNodeParameter`."
                      " If you need to record the value use a recorder capable of recording an arbitrary parameter"
                      " (e.g. `NumpyArrayParameterRecorder`)", DeprecationWarning, stacklevel=2)

        super(RollingMeanFlowNodeRecorder, self).__init__(model, node, name=name, **kwargs)
        self.model = model
        if not timesteps and not days:
            raise ValueError("Either `timesteps` or `days` must be specified.")
        if timesteps:
            self.timesteps = int(timesteps)
        else:
            self.timesteps = 0
        if days:
            self.days = int(days)
        else:
            self.days = 0
        self._data = None
        self._memory = None
        self.position = 0

    cpdef setup(self):
        """Setup the internal variable."""
        super(RollingMeanFlowNodeRecorder, self).setup()
        self._data = np.empty([len(self.model.timestepper), len(self.model.scenarios.combinations)])
        if self.days > 0:
            try:
                self.timesteps = self.days // self.model.timestepper.delta
            except TypeError:
                raise TypeError('A rolling window defined as a number of days is only valid with daily time-steps.')
        if self.timesteps == 0:
            raise ValueError("Timesteps property of MeanFlowRecorder is less than 1.")
        self._memory = np.zeros([len(self.model.scenarios.combinations), self.timesteps])

    cpdef reset(self):
        super(RollingMeanFlowNodeRecorder, self).reset()
        self.position = 0

    cpdef after(self):
        cdef Timestep timestep
        cdef int i, n
        cdef double[:] mean_flow
        # save today's flow
        for i in range(0, self._memory.shape[0]):
            self._memory[i, self.position] = self._node._flow[i]
        # calculate the mean flow
        timestep = self.model.timestepper.current
        if timestep.index < self.timesteps:
            n = timestep.index + 1
        else:
            n = self.timesteps
        # save the mean flow
        mean_flow = np.mean(self._memory[:, 0:n], axis=1)
        self._data[<int>(timestep.index), :] = mean_flow
        # prepare for the next timestep
        self.position += 1
        if self.position >= self.timesteps:
            self.position = 0

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the flow data."""
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    @classmethod
    def load(cls, model, data):
        name = data.get("name")
        node = model.nodes[data["node"]]
        if "timesteps" in data:
            timesteps = int(data["timesteps"])
        else:
            timesteps = None
        if "days" in data:
            days = int(data["days"])
        else:
            days = None
        return cls(model, node, timesteps=timesteps, days=days, name=name)

RollingMeanFlowNodeRecorder.register()

cdef class BaseConstantNodeRecorder(NodeRecorder):
    """
    Base class for NodeRecorder classes with a single value for each scenario combination
    """

    cpdef setup(self):
        """Setup the internal variable."""
        self._values = np.zeros(len(self.model.scenarios.combinations))

    cpdef reset(self):
        """Reset the internal variable."""
        self._values[...] = 0.0

    cpdef after(self):
        raise NotImplementedError()

    cpdef double[:] values(self) except *:
        """Return the internal values."""
        return self._values


cdef class TotalDeficitNodeRecorder(BaseConstantNodeRecorder):
    """This recorder calculates a node's deficit at each timestep and returns the total
    deficit at the end of the run.

    The deficit is calculated as the difference between the value in the node's `max_flow`
    attribute and the flow allocated in `flow`:

        deficit = max_flow - actual_flow

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import TotalDeficitNodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    TotalDeficitNodeRecorder(
        model=model,
        name="Demand total deficit",
        node=demand
    )
    ```

    JSON
    ======
    ```json
    {
        "Demand total deficit": {
            "type": "TotalDeficitNodeRecorder",
            "node": "Demand"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    factor : int
        The value to use to scale the flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    cpdef after(self):
        """Calculate the deficit for the timestep."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = self.model.timestepper.current.days
        cdef AbstractNode node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            self._values[scenario_index.global_id] += (max_flow - node._flow[scenario_index.global_id])*days

        return 0
TotalDeficitNodeRecorder.register()


cdef class TotalFlowNodeRecorder(BaseConstantNodeRecorder):
    """
    This recorder returns the total flow for a Node at the end of the simulation for each
    scenario. A factor can also be provided to scale the total flow (for example, to calculate
    the operational costs).

    Examples
    -------
    In the following example, the total operational cost of supply is recorded:

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import TotalFlowNodeRecorder

    model = Model()
    node = Output(model, name="Demand", max_flow=5)
    TotalFlowNodeRecorder(
        model=model,
        node=node,
        factor=100,
        name="Total cost"
    )
    ```

    JSON
    ======
    ```json
    {
        "Total cost": {
            "type": "TotalFlowNodeRecorder",
            "factor": 100,
            "node": "Demand"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    factor : int
        The value to use to scale the flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            The node instance to recorder the flow of.
        factor : Optional[int], default=1
            The value to use to scale the flow.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        self.factor = kwargs.pop('factor', 1.0)
        super(TotalFlowNodeRecorder, self).__init__(*args, **kwargs)

    cpdef after(self):
        """Calculate the total flow for each scenario."""
        cdef ScenarioIndex scenario_index
        cdef int i
        cdef double days = self.model.timestepper.current.days
        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            self._values[i] += self._node._flow[i]*self.factor*days
        return 0
TotalFlowNodeRecorder.register()


cdef class MeanFlowNodeRecorder(BaseConstantNodeRecorder):
    """
    This recorder returns the mean flow for a Node at the end of the simulation for each
    scenario. A factor can also be provided to scale the mean flow (for example, to calculate
    the operational costs).

    Examples
    -------
    In the following example, the mean operational cost of supply is recorded:

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import MeanFlowNodeRecorder

    model = Model()
    node = Output(model, name="Demand", max_flow=5)
    MeanFlowNodeRecorder(
        model=model,
        node=node,
        factor=100,
        name="Total cost"
    )
    ```

    JSON
    ======
    ```json
    {
        "Total cost": {
            "type": "MeanFlowNodeRecorder",
            "factor": 100,
            "node": "Demand"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    factor : int
        The value to use to scale the flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            The node instance to recorder the flow of.
        factor : Optional[int], default=1
            The value to use to scale the flow.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        self.factor = kwargs.pop('factor', 1.0)
        super(MeanFlowNodeRecorder, self).__init__(*args, **kwargs)

    cpdef after(self):
        """Calculate the total flow for each scenario."""
        cdef ScenarioIndex scenario_index
        cdef int i
        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            self._values[i] += self._node._flow[i]*self.factor
        return 0

    cpdef finish(self):
        """Scale the total flow by the number of steps to calculate the average flow."""
        cdef int i
        cdef int nt = self.model.timestepper.current.index
        for i in range(self._values.shape[0]):
            self._values[i] /= nt
MeanFlowNodeRecorder.register()


cdef class DeficitFrequencyNodeRecorder(BaseConstantNodeRecorder):
    """This recorder returns the frequency of timesteps with a failure to meet a node's `max_flow`.

    The deficit is calculated as the difference between the value in the node's `max_flow`
    attribute and the flow allocated in `flow` at each timestep:

        deficit = max_flow - actual_flow

    When this is not zero, the recorder internal counter increases by one (timestep). At the end
    of the run this number is divided by the total number of timesteps to return a frequency.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import DeficitFrequencyNodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    DeficitFrequencyNodeRecorder(
        model=model,
        name="Demand deficit frequency",
        node=demand
    )
    ```

    JSON
    ======
    ```json
    {
        "Demand deficit frequency": {
            "type": "DeficitFrequencyNodeRecorder",
            "node": "Demand"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint.
    """
    cpdef after(self):
        """Count the number of timesteps where there is a deficit for each scenario."""
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef AbstractNode node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            if abs(node._flow[scenario_index.global_id] - max_flow) > 1e-6:
                self._values[scenario_index.global_id] += 1.0

    cpdef finish(self):
        """Calculate the frequency for each scenario."""
        cdef int i
        cdef int nt = self.model.timestepper.current.index
        for i in range(self._values.shape[0]):
            self._values[i] /= nt
DeficitFrequencyNodeRecorder.register()

cdef class BaseConstantStorageRecorder(StorageRecorder):
    """
    Base class for StorageRecorder classes with a single value for each scenario combination
    """

    cpdef setup(self):
        """Setup the internal variable."""
        self._values = np.zeros(len(self.model.scenarios.combinations))

    cpdef reset(self):
        self._values[...] = 0.0

    cpdef after(self):
        raise NotImplementedError()

    cpdef double[:] values(self) except *:
        return self._values
BaseConstantStorageRecorder.register()

cdef class MinimumVolumeStorageRecorder(BaseConstantStorageRecorder):
    """This recorder stores the minimum volume in a `Storage` node during a simulation.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import MinimumVolumeStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    MinimumVolumeStorageRecorder(
        model=model,
        name="Min volume",
        node=storage
    )
    ```

    JSON
    ======
    ```json
    {
        "Min volume": {
            "type": "MinimumVolumeStorageRecorder",
            "node": "Reservoir"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    cpdef reset(self):
        """Reset the internal variables."""
        self._values[...] = np.inf

    cpdef after(self):
        """Calculate the min storage per scenario."""
        cdef int i
        for i in range(self._values.shape[0]):
            self._values[i] = np.min([self._node._volume[i], self._values[i]])
        return 0
MinimumVolumeStorageRecorder.register()

cdef class MinimumThresholdVolumeStorageRecorder(BaseConstantStorageRecorder):
    """This recorder checks whether the absolute volume in a `Storage` node during a simulation
    falls below a particular volume threshold It returns a value
    of `1.0`, for each scenario, when the absolute volume is less than or equal to the threshold
    at any time-step during the simulation. Otherwise, it will return zero.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import MinimumThresholdVolumeStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    MinimumThresholdVolumeStorageRecorder(
        model=model,
        name="Threshold",
        threshold=10,
        node=storage
    )
    ```

    JSON
    ======
    ```json
    {
        "Threshold": {
            "type": "MinimumThresholdVolumeStorageRecorder",
            "node": "Reservoir",
            "threshold": 10
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float
        The storage threshold.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, node, threshold, *args, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            The storage instance to recorder the volume of.
        threshold : float
            The storage threshold.

        Other Parameters
        ----------------
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        name : Optional[str], default=None
            Name of the recorder.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        self.threshold = threshold
        super(MinimumThresholdVolumeStorageRecorder, self).__init__(model, node, *args, **kwargs)

    cpdef reset(self):
        """Reset the internal variables."""
        self._values[...] = 0.0

    cpdef after(self):
        """Calculate the min storage per scenario if the threshold is exceeded."""
        cdef int i
        for i in range(self._values.shape[0]):
            if self._node._volume[i] <= self.threshold:
                self._values[i] = 1.0
        return 0
MinimumThresholdVolumeStorageRecorder.register()


cdef class TimestepCountIndexParameterRecorder(IndexParameterRecorder):
    """Record the number of times an index parameter exceeds a threshold for each scenario.

    This recorder will count the number of timesteps so will be a daily count when running on a
    daily timestep.

    Parameters
    ----------
    model : `pywr.core.Model`
    parameter : `pywr.core.IndexParameter`
        The parameter to record
    threshold : int
        The threshold to compare the parameter to
    """
    def __init__(self, model, IndexParameter parameter, int threshold, *args, **kwargs):
        super().__init__(model, parameter, *args, **kwargs)
        self.threshold = threshold

    cpdef setup(self):
        """Setup the internal variable."""
        self._count = np.zeros(len(self.model.scenarios.combinations), np.int32)

    cpdef reset(self):
        self._count[...] = 0

    cpdef after(self):
        cdef Timestep ts = self.model.timestepper.current
        cdef int value
        cdef ScenarioIndex scenario_index

        for scenario_index in self.model.scenarios.combinations:
            value = self._param.get_index(scenario_index)
            if value >= self.threshold:
                # threshold achieved, increment count
                self._count[scenario_index.global_id] += 1

    cpdef double[:] values(self) except *:
        return np.asarray(self._count).astype(np.float64)
TimestepCountIndexParameterRecorder.register()


cdef class AnnualCountIndexThresholdRecorder(Recorder):
    """For each scenario, count the number of times a list of parameters exceeds a threshold in each year.

    If multiple parameters exceed in one timestep then it is only counted once. The recorder also allows
    for exclusion of months and for the inclusion of a range of dates within a calendar year to which
    the parameter exceedence is counted. Both the exclusion of months and the inclusion of dates can
    simultaneously be provided, where the intersection of excluded months with a range of dates will result
    in the day not counting any exceedences.

    Output from data property has shape: (years, scenario combinations)

    Parameters
    ----------
    model : `pywr.core.Model`
    parameters : list
        List of `pywr.core.IndexParameter` to record against
    name : str
        The name of the recorder
    threshold : int
        Threshold to compare parameters against
    exclude_months : list or None
        Optional list of month numbers to exclude from the count.
    include_from_month, include_from_day : int or None
        Optional start date to specify a range of dates to include in the count. If intended to be used,
        both arguments must be supplied, otherwise the recorder will assume that this is not used and default
        to the 1st Jan. Period to count is inclusive of the start date.
    include_to_month, include_to_day : int or None
        Optional end date to specify a range of dates to include in the count. If intended to be used,
        both arguments must be supplied, otherwise the recorder will assume that this is not used and default
        to the 31st Dec. Period to count is inclusive of the end date.
    """
    def __init__(self, model, list parameters, str name, int threshold, *args, **kwargs):
        self.exclude_months = kwargs.pop('exclude_months', None)
        self.include_from_month = kwargs.pop('include_from_month', None)
        self.include_from_day = kwargs.pop('include_from_day', None)
        self.include_to_month = kwargs.pop('include_to_month', None)
        self.include_to_day = kwargs.pop('include_to_day', None)
        # Optional different method for aggregating across time.
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'sum')
        super(AnnualCountIndexThresholdRecorder, self).__init__(model, name=name, *args, **kwargs)
        self.parameters = parameters
        self.threshold = threshold
        for parameter in self.parameters:
            self.children.add(parameter)
        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    cpdef setup(self):
        """Setup the internal variables."""
        super(AnnualCountIndexThresholdRecorder, self).setup()
        self._num_years = self.model.timestepper.end.year - self.model.timestepper.start.year + 1
        self._ncomb = len(self.model.scenarios.combinations)
        self._data = np.empty([self._num_years, self._ncomb])
        self._data_this_year = np.zeros([len(self.parameters), self._ncomb])

    cpdef reset(self):
        self._data[...] = 0
        self._current_year = -1
        self._start_year = self.model.timestepper.start.year

    cpdef after(self):
        cdef Timestep ts = self.model.timestepper.current
        cdef int idx = self._current_year - self._start_year
        cdef int p
        cdef int include_from
        cdef int include_to
        cdef Py_ssize_t i
        cdef int value
        cdef ScenarioIndex scenario_index
        cdef IndexParameter parameter

        if ts.year != self._current_year:
            # A new year
            if self._current_year != -1:
                # As long as at least one year has been run
                # then save data for previous year
                for i in range(self._ncomb):
                    self._data[idx, i] = np.sum(self._data_this_year[:, i])

            self._data_this_year[...] = 0
            self._current_year = ts.year

        if self.exclude_months is not None and ts.month in self.exclude_months:
            return

        # include a range of dates within a year
        if self.include_from_month is None or self.include_from_day is None:
            include_from = 1
        else:
            include_from = pd.Timestamp(self._current_year, self.include_from_month, self.include_from_day).dayofyear
        if self.include_to_month is None or self.include_to_day is None:
            include_to = 366
        else:
            include_to = pd.Timestamp(self._current_year, self.include_to_month, self.include_to_day).dayofyear
        if not (include_from <= ts.dayofyear <= include_to):
            return

        for scenario_index in self.model.scenarios.combinations:
            for p, parameter in enumerate(self.parameters):
                value = parameter.get_index(scenario_index)
                if value >= self.threshold:
                    self._data_this_year[p, scenario_index.global_id] += 1
                    break  # if multiple parameters exceed, only count once

    cpdef finish(self):
        cdef int idx = self._current_year - self._start_year
        cdef Py_ssize_t i
        for i in range(self._ncomb):
            self._data[idx, i] = np.sum(self._data_this_year[:, i])

    cpdef double[:] values(self) except *:
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the years.
        """
        start_year = self.model.timestepper.start.year
        end_year = self.model.timestepper.end.year
        index = pd.period_range(f'{start_year}-01-01', f'{end_year}-01-01', freq='A')
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=np.array(self._data, dtype=int), index=index, columns=sc_index)

    property data:
        """This contains an array with shape (total_timesteps, number_of_scenarios) with
        the counters."""
        def __get__(self):
            return np.array(self._data, dtype=np.int16)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        parameters = [load_parameter(model, p) for p in data.pop("parameters")]
        return cls(model, parameters=parameters, **data)
AnnualCountIndexThresholdRecorder.register()


cdef class AnnualTotalFlowRecorder(Recorder):
    """
    This recorder calculates the total flow in each year across a list of nodes
    for each scenario. The output is saved in the `data` property which returns
    an array with shape: (years, scenario combinations).

    A list of factors can be provided to scale the total flow (e.g. for calculating the operational costs).

    Examples
    -------
    In the following example, the total operational cost of supply is recorded:

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import AnnualTotalFlowRecorder

    model = Model()
    node1 = Output(model, name="Demand 1", max_flow=5)
    node2 = Output(model, name="Demand 2", max_flow=3.4)
    AnnualTotalFlowRecorder(
        model=model,
        nodes=[node1, node2],
        factors=[100, 12],
        name="Total costs"
    )
    ```

    JSON
    ======
    ```json
    {
        "Total costs": {
            "type": "AnnualTotalFlowRecorder",
            "factors": [100, 12],
            "nodes": ["Demand 1, "Demand 2"]
        }
    }
    ```


    Attributes
    ----------
    model : Model
        The model instance.
    nodes : list[Node]
        The list of nodes to calculate the annual total flow of.
    factors : list[float]
        Scale each flow by the given factors.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, model, str name, list nodes, *args, **kwargs):
        """Initialise the recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        name : Optional[str], default=None
            Name of the recorder.
        nodes : list[Node]
            The list of nodes to calculate the annual total flow of.

        Other parameters
        ----------------
        factors : Optional[list[float]], default=None
            Scale each flow by the given factors.
        temporal_agg_func : Optional[str | Callable], default="sum"
            When `.values()` is called, temporally aggregate the data using the given function for each scenario.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'sum')
        factors = kwargs.pop('factors', None)
        super().__init__(model, name=name, *args, **kwargs)
        self.nodes = nodes
        self.factors = factors
        self._temporal_aggregator = Aggregator(temporal_agg_func)

    property temporal_agg_func:
        """The temporal aggregation function used in `.value()`.
        
        **Setter:** set the aggregation function.
        """
        def __set__(self, agg_func):
            self._temporal_aggregator.func = agg_func

    property factors:
        """The factors used to scale the total flow.
        
        **Setter:** set the factors.
        """
        # Property provides np.array style access to the internal memoryview.
        def __get__(self):
            return np.array(self._factors)
        def __set__(self, factors):
            if factors is None:
                factors = np.array([1.0 for n in self.nodes])
            self._factors = np.array(factors)

    cpdef setup(self):
        """Setup the interval variables."""
        super(AnnualTotalFlowRecorder, self).setup()
        self._num_years = self.model.timestepper.end.year - self.model.timestepper.start.year + 1
        self._ncomb = len(self.model.scenarios.combinations)
        self._data = np.empty([self._num_years, self._ncomb])

    cpdef reset(self):
        """Reset the interval variables."""
        self._data[...] = 0
        self._current_year = -1
        self._start_year = self.model.timestepper.start.year

    cpdef after(self):
        """Calculate the total annual flows."""
        cdef int i, j
        cdef Timestep ts = self.model.timestepper.current
        cdef int idx = ts.year - self._start_year
        cdef AbstractNode node
        cdef double days_in_current_year = ts.days_in_current_year()
        cdef double days_in_next_year = ts.days_in_next_year()
        cdef double ts_days = ts.days

        for i in range(self._ncomb):
            for j, node in enumerate(self.nodes):
                self._data[idx, i] += node._flow[i] * days_in_current_year * self._factors[j]
                if days_in_current_year != ts.days and idx+1 < self._data.shape[0]:
                    # Timestep cross into the next year.
                    self._data[idx + 1, i] += node._flow[i] * days_in_next_year * self._factors[j]

    cpdef double[:] values(self) except *:
        """Compute a value for each scenario using `temporal_agg_func`.
        
        Returns
        numpy.typing.NDArray[numpy.number]
            An array with the aggregated values. This has a size equal to the number of scenarios.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)

    property data:
        """This contains an array with shape (total_years, number_of_scenarios) with
        the total scaled flow in each year and scenario."""
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    def to_dataframe(self):
        """Convert the data to a `pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains a MultiIndex for the columns with the recorder name
            as the first level and scenario combination names as the second level. The row index
            contains the years.
        """
        start_year = self.model.timestepper.start.year
        end_year = self.model.timestepper.end.year
        index = pd.period_range(f'{start_year}-01-01', f'{end_year}-01-01', freq='A')
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        AnnualTotalFlowRecorder
            The loaded class.
        """
        nodes = [model.nodes[n] for n in data.pop("nodes")]
        return cls(model, nodes=nodes, **data)
AnnualTotalFlowRecorder.register()


cdef class AnnualCountIndexParameterRecorder(IndexParameterRecorder):
    """ Record the number of years where an IndexParameter is greater than or equal to a threshold """
    def __init__(self, model, IndexParameter param, int threshold, *args, **kwargs):
        super(AnnualCountIndexParameterRecorder, self).__init__(model, param, *args, **kwargs)
        self.threshold = threshold

    cpdef setup(self):
        """Setup the internal variables."""
        self._count = np.zeros(len(self.model.scenarios.combinations), np.int32)
        self._current_max = np.zeros_like(self._count)

    cpdef reset(self):
        self._count[...] = 0
        self._current_max[...] = 0
        self._current_year = -1

    cpdef after(self):
        cdef int i, ncomb, value
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current

        ncomb = len(self.model.scenarios.combinations)

        if ts.year != self._current_year:
            # A new year
            if self._current_year != -1:
                # As long as at least one year has been run
                # then update the count if threshold equal to or exceeded
                for i in range(ncomb):
                    if self._current_max[i] >= self.threshold:
                        self._count[i] += 1

            # Finally reset current maximum and update current year
            self._current_max[...] = 0
            self._current_year = ts.year

        for scenario_index in self.model.scenarios.combinations:
            # Get current parameter value
            value = self._param.get_index(scenario_index)

            # Update annual max if a new maximum is found
            if value > self._current_max[scenario_index.global_id]:
                self._current_max[scenario_index.global_id] = value

        return 0

    cpdef finish(self):
        cdef int i
        cdef int ncomb = len(self.model.scenarios.combinations)
        # Complete the current year by updating the count if threshold equal to or exceeded
        for i in range(ncomb):
            if self._current_max[i] >= self.threshold:
                self._count[i] += 1

    cpdef double[:] values(self) except *:
        return np.asarray(self._count).astype(np.float64)
AnnualCountIndexParameterRecorder.register()


def load_recorder(model, data, recorder_name=None):
    recorder = None

    if isinstance(data, str):
        recorder_name = data

    # check if recorder has already been loaded
    for rec in model.recorders:
        if rec.name == recorder_name:
            recorder = rec
            break

    if recorder is None and isinstance(data, str):
        # recorder was requested by name, but hasn't been loaded yet
        if hasattr(model, "_recorders_to_load"):
            # we're still in the process of loading data from JSON and
            # the parameter requested hasn't been loaded yet - do it now
            try:
                data = model._recorders_to_load.pop(recorder_name)
            except KeyError:
                raise KeyError("Unknown recorder: '{}'".format(data))
            recorder = load_recorder(model, data)
        else:
            raise KeyError("Unknown recorder: '{}'".format(data))

    if recorder is None:
        recorder_type = data['type']

        name = recorder_type.lower()
        try:
            cls = recorder_registry[name]
        except KeyError:
            if name.endswith("recorder"):
                name = name.replace("recorder", "")
            else:
                name += "recorder"
            try:
                cls = recorder_registry[name]
            except KeyError:
                raise NotImplementedError('Unrecognised recorder type "{}"'.format(recorder_type))

        del(data["type"])
        recorder = cls.load(model, data)

    return recorder


cdef class BaseConstantParameterRecorder(ParameterRecorder):
    """Base class for `ParameterRecorder` classes with a single value for each scenario combination
    """
    cpdef setup(self):
        """Setup the internal variable."""
        self._values = np.zeros(len(self.model.scenarios.combinations))

    cpdef reset(self):
        """Reset the internal variable."""
        self._values[...] = 0.0

    cpdef after(self):
        raise NotImplementedError()

    cpdef double[:] values(self) except *:
        """
        Get the values stored by the recorder.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._values


cdef class TotalParameterRecorder(BaseConstantParameterRecorder):
    """The recorder calculates the total value of a [pywr.parameters.Parameter][]
    over a simulation. An optional `factor` can be provided to
    apply a linear scaling of the values. If the parameter represents a flux
    the `integrate` keyword argument can be used to multiply the values by the time-step
    length in days.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import TotalParameterRecorder

    model = Model()
    river = Link(model=model, name="River")
    parameter = NodeThresholdParameter(
        model=model,
        node=river,
        predicate="LT",
        values=[10, 2],
        threshold=1000,
        name="Daily river-dependant license"
    )
    TotalParameterRecorder(
        model=model,
        name="Mean max abstraction",
        param=parameter
    )
    ```

    JSON
    ======
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean max abstraction": {
            "type": "TotalParameterRecorder",
            "parameter": "Daily river-dependant license"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    factor : float
        The scaling factor for the values of `param`.
    integrate : bool
        Whether to multiply the value by the time-step length in days during the summation.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            The parameter instance to use to get the value.
        factor : Optional[float], default=1.0
            The scaling factor for the values of `param`.
        integrate : Optional[bool], default=False
            Whether to multiply the value by the time-step length in days during the summation.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        self.factor = kwargs.pop('factor', 1.0)
        self.integrate = kwargs.pop('integrate', False)
        super(TotalParameterRecorder, self).__init__(*args, **kwargs)

    cpdef after(self):
        """Add the value of the current timestep."""
        cdef ScenarioIndex scenario_index
        cdef int i
        cdef double[:] values
        cdef factor = self.factor

        if self.integrate:
            factor *= self.model.timestepper.current.days

        values = self._param.get_all_values()
        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            self._values[i] += values[i]*factor
        return 0
TotalParameterRecorder.register()


cdef class MeanParameterRecorder(BaseConstantParameterRecorder):
    """This recorder calculates the mean value of a [pywr.parameters.Parameter][]
    over a simulation. An optional `factor` can be provided to apply a linear
    scaling of the values.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import MeanParameterRecorder

    model = Model()
    river = Link(model=model, name="River")
    parameter = NodeThresholdParameter(
        model=model,
        node=river,
        predicate="LT",
        values=[10, 2],
        threshold=1000,
        name="Daily river-dependant license"
    )
    MeanParameterRecorder(
        model=model,
        name="Mean max abstraction",
        param=parameter
    )
    ```

    JSON
    ======
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean max abstraction": {
            "type": "MeanParameterRecorder",
            "parameter": "Daily river-dependant license"
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    factor : float
        The scaling factor for the values of `param`.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            The parameter instance to use to get the value.
        factor : Optional[float], default=1.0
            The scaling factor for the values of `param`.
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        self.factor = kwargs.pop('factor', 1.0)
        super(MeanParameterRecorder, self).__init__(*args, **kwargs)

    cpdef after(self):
        """Add the value of the current timestep."""
        cdef ScenarioIndex scenario_index
        cdef int i
        cdef double[:] values
        cdef factor = self.factor

        values = self._param.get_all_values()
        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index.global_id
            self._values[i] += values[i]*factor
        return 0

    cpdef finish(self):
        """Calculate the mean value from the total."""
        cdef int i
        cdef int nt = self.model.timestepper.current.index
        for i in range(self._values.shape[0]):
            self._values[i] /= nt
MeanParameterRecorder.register()

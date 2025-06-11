from ._parameters import load_parameter
cimport numpy as np
import numpy as np

cdef enum Predicates:
    """The enum predicate."""
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4
_predicate_lookup = {
    "LT": Predicates.LT, "<": Predicates.LT,
    "GT": Predicates.GT, ">": Predicates.GT,
    "EQ": Predicates.EQ, "=": Predicates.EQ,
    "LE": Predicates.LE, "<=": Predicates.LE,
    "GE": Predicates.GE, ">=": Predicates.GE,
}

cdef class AbstractThresholdParameter(IndexParameter):
    """ Base class for parameters returning one of two values depending on another state.

    Parameters
    ----------
    threshold : double or Parameter
        Threshold to compare the value of the recorder to
    values : iterable of doubles
        If the predicate evaluates False the zeroth value is returned,
        otherwise the first value is returned.
    predicate : string
        One of {"LT", "GT", "EQ", "LE", "GE"}.
    ratchet : bool
        If true the parameter behaves like a ratchet. Once it is triggered,
        it stays in the triggered position (default=False).

    Notes
    -----
    On the first day of the model runs, the recorder will not have a value for
    the previous day. In this case, the predicate evaluates to True.

    """
    def __init__(self, model, threshold, *args, values=None, predicate=None, ratchet=False, **kwargs):
        super(AbstractThresholdParameter, self).__init__(model, *args, **kwargs)
        self.threshold = threshold
        if values is None:
            self.values = None
        else:
            self.values = np.array(values, np.float64)
        if predicate is None:
            predicate = Predicates.LT
        elif isinstance(predicate, str):
            predicate = _predicate_lookup[predicate.upper()]
        self.predicate = predicate
        self.ratchet = ratchet

    cpdef setup(self):
        """Setup the parameter state."""
        super(AbstractThresholdParameter, self).setup()
        cdef int ncomb = len(self.model.scenarios.combinations)
        self._triggered = np.empty(ncomb, dtype=np.uint8)

    cpdef reset(self):
        """Reset the parameter state."""
        super(AbstractThresholdParameter, self).reset()
        self._triggered[...] = 0

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        raise NotImplementedError()

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Get the value from the values attribute, using the index for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The value in `values`.
        """
        cdef int ind = self.get_index(scenario_index)
        cdef double v
        if self.values is not None:
            v = self.values[ind]
        else:
            return np.nan
        return v

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evalutes True, else 0
        
        Parameters
        ----------
        timestep : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The value.
        """
        cdef double x
        cdef bint ind, triggered

        triggered = self._triggered[scenario_index.global_id]

        # Return triggered state if ratchet is enabled.
        if self.ratchet and triggered:
            return triggered

        x = self._value_to_compare(timestep, scenario_index)

        cdef double threshold
        if self._threshold_parameter is not None:
            threshold = self._threshold_parameter.get_value(scenario_index)
        else:
            threshold = self._threshold

        if self.predicate == Predicates.LT:
            ind = x < threshold
        elif self.predicate == Predicates.GT:
            ind = x > threshold
        elif self.predicate == Predicates.LE:
            ind = x <= threshold
        elif self.predicate == Predicates.GE:
            ind = x >= threshold
        else:
            ind = x == threshold

        self._triggered[scenario_index.global_id] = max(ind, triggered)
        return ind

    property threshold:
        """The threshold to compare against the values.
        
        **Setter**: set the threshold as `float` or [pywr.parameters.Parameter][].
        """
        def __get__(self):
            if self._threshold_parameter is not None:
                return self._threshold_parameter
            else:
                return self._threshold

        def __set__(self, value):
            if self._threshold_parameter is not None:
                self.children.remove(self._threshold_parameter)
                self._threshold_parameter = None
            if isinstance(value, Parameter):
                self._threshold_parameter = value
                self.children.add(self._threshold_parameter)
            else:
                self._threshold = value

cdef class StorageThresholdParameter(AbstractThresholdParameter):
    """This parameter returns one of two values depending on the current volume in a
    Storage node by assessing the following predicate:

        storage predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.

    Notes
    -----
    On the first day of the model run, the calculated storage will not have a value for
    the previous day. In this case, the predicate evaluates to True.

    Examples
    -------
    In the example below the storage in the node "Reservoir" is compared against the constant threshold `0.5`. When
    the storage is less than `0.5`, `0` is returned, otherwise `10` is returned. 

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageThresholdParameter

    model = Model()
    node = Storage(model=model, max_volume=100, name="Reservoir", initial_volume=100)
    StorageThresholdParameter(
        model=model, 
        storage=node, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="My parameter"
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "StorageThresholdParameter",
            "storage":"Reservoir",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare the value of the storage to
    storage : Storage
        The storage node to use as value to compare against the `threshold`.
    ratchet : bool
        If true the parameter behaves like a ratchet. Once it is triggered,
        it stays in the triggered position.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.


    """
    def __init__(self, model, AbstractStorage storage, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        storage : AbstracStorage
            The storage node whose volume is compared against the threshold.
       
        Other Parameters
        ----------------
        threshold : double or Parameter
            Threshold to compare the value of the recorder to.
        values : iterable of doubles
            If the predicate evaluates False the zeroth value is returned,
            otherwise the first value is returned.
        predicate : Optional[Literal["LT", "GT", "EQ", "LE", "GE"]], default="LT"
            Compare the threshold against the storage using the given predicate.
        ratchet : Optional[bool], default=False
            If true the parameter behaves like a ratchet. Once it is triggered,
            it stays in the triggered position.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(StorageThresholdParameter, self).__init__(model, *args, **kwargs)
        self.storage = storage

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.storage._volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        StorageThresholdParameter
            The loaded class.
        """
        node = model.nodes[data.pop("storage_node")]
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, node, threshold, values=values, predicate=predicate, **data)
StorageThresholdParameter.register()


cdef class NodeThresholdParameter(AbstractThresholdParameter):
    """This parameter returns one of two values depending on the previous flow in a node 
    by assessing the following predicate:

        flow predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.
    
    Notes
    -----
    On the first day of the model run, the calculated flow will not have a value for
    the previous day. In this case, the predicate evaluates to True.

    Examples
    -------
    In the example below the flow of the "WTW" node is compared against the constant threshold `0.5`. When
    the previous flow is less than `0.1`, `0` is returned, otherwise `10` is returned. 

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageThresholdParameter

    model = Model()
    node = Link(model=model, max_flow=10, name="WTW")
    NodeThresholdParameter(
        model=model, 
        node=node, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.1,
        name="My parameter"
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "NodeThresholdParameter",
            "node":"WTW",
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.1,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare the value of the flow to
    storage : Storage
        The storage node to use as value to compare against the `threshold`.
    ratchet : bool
        If true the parameter behaves like a ratchet. Once it is triggered,
        it stays in the triggered position.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    """
    def __init__(self, model, AbstractNode node, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : AbstractNode
            The node whose previous flow is compared against the threshold.
       
        Other Parameters
        ----------------
        threshold : double or Parameter
            Threshold to compare the value of the recorder to.
        values : iterable of doubles
            If the predicate evaluates False the zeroth value is returned,
            otherwise the first value is returned.
        predicate : Optional[Literal["LT", "GT", "EQ", "LE", "GE"]], default="LT"
            Compare the threshold against the storage using the given predicate.
        ratchet : Optional[bool], default=False
            If true the parameter behaves like a ratchet. Once it is triggered,
            it stays in the triggered position.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(NodeThresholdParameter, self).__init__(model, *args, **kwargs)
        self.node = node

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.node._prev_flow[scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        if timestep.index == 0:
            # previous flow on initial timestep is undefined
            return 0
        return AbstractThresholdParameter.index(self, timestep, scenario_index)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        NodeThresholdParameter
            The loaded class.
        """
        node = model.nodes[data.pop("node")]
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, node, threshold, values=values, predicate=predicate, **data)
NodeThresholdParameter.register()


cdef class MultipleThresholdIndexParameter(IndexParameter):
    """This parameter returns an index based on the previous days flow in a node 
    against multiple given thresholds. 

    The index is zero-based. For example, if only one threshold is
    supplied, then the index is either 0 (above) or 1 (below). For two
    thresholds, the index is either 0 (above both), 1 (in between), or 2 (below
    both), and so on.

    The parameter always returns the index of the first threshold the node flow is above.

    Examples
    -------

    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Input
    from pywr.parameters import ArrayIndexedParameter, MultipleThresholdIndexParameter

    model = Model()
    node = Input(
        model,
        max_flow=ArrayIndexedParameter(
            model, np.arange(0, 20)
        ), 
        name="input"
    )
    parameter = MultipleThresholdIndexParameter(
        model=model,
        name="My parameter",
        node=node,
        thresholds=[10, 5, 2]
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "MultipleThresholdIndexParameter",
            "node": "input",    
            "thresholds": [10, 5, 2]
        }
    }
    ```
    
    In the example, the parameter will return the following array if recorder and when
    the model runs for 13 timesteps:
        
        [3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0]

    Attributes
    ----------
    model : Model
        The model instance.
    node : AbstractNode
        The node used for the flow
    thresholds : Iterable[Parameter | float]
        The thresholds.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, node, thresholds, use_max_flow=False, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : AbstractNode
            The node whose flow is compared against the thresholds.
        thresholds : Iterable[Parameter | float]
            The thresholds.
       
        Other Parameters
        ----------------
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(MultipleThresholdIndexParameter, self).__init__(model, **kwargs)
        self.node = node

        self.thresholds = []
        for threshold in thresholds:
            if not isinstance(threshold, Parameter):
                from pywr.parameters import ConstantParameter
                threshold = ConstantParameter(model, threshold)
            self.thresholds.append(threshold)

        for threshold in self.thresholds:
            self.children.add(thresholds)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """This returns the index of the first threshold the node flow is above.

        The index is zero-based. For example, if only one threshold is
        supplied then the index is either 0 (above) or 1 (below). For two
        thresholds the index is either 0 (above both), 1 (in between), or 2 (below
        both), and so on.

        Parameters
        ----------
        timestep : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The parameter index.
        """
        cdef double flow
        cdef int index, j
        cdef double target_threshold
        cdef Parameter threshold

        flow = self.node._prev_flow[scenario_index.global_id]
        index = len(self.thresholds)
        for j, threshold in enumerate(self.thresholds):
            target_threshold = threshold.get_value(scenario_index)
            if flow >= target_threshold:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        MultipleThresholdIndexParameter
            The loaded class.
        """
        node = model.nodes[data.pop("node")]
        thresholds = [load_parameter(model, d) for d in data.pop("thresholds")]
        return cls(model, node, thresholds, **data)
MultipleThresholdIndexParameter.register()


cdef class MultipleThresholdParameterIndexParameter(IndexParameter):
    """This parameter returns an index based on the value in the parameter against 
    multiple given thresholds.
    
    The index is zero-based. For example, if only one threshold is
    supplied, then the index is either 0 (above) or 1 (below). For two
    thresholds, the index is either 0 (above both), 1 (in between), or 2 (below
    both), and so on.

    The parameter always returns the index of the first threshold the node flow is above.

    Examples
    -------

    Python
    ======
    ```python
    from pywr.core import Model
    import numpy as np
    from pywr.nodes import Input
    from pywr.parameters import ArrayIndexedParameter, MultipleThresholdParameterIndexParameter

    model = Model()
    parameter = ArrayIndexedParameter(
        model,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        name="Index"
    )
    MultipleThresholdParameterIndexParameter(
        model=model,
        name="My parameter",
        parameter=parameter,
        thresholds=[10, 5, 2]
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "MultipleThresholdParameterIndexParameter",
            "parameter": "Index",    
            "thresholds": [10, 5, 2]
        }
    }
    ```
    
    In the example, the parameter will return the following arra if recorder:
        
        [3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1] 

    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter instance whose value is compared against the thresholds.
    thresholds : Iterable[Parameter | float]
            The thresholds.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, parameter, thresholds, use_max_flow=False, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter instance whose value is compared against the thresholds.
        thresholds : Iterable[Parameter | float]
            The thresholds.
       
        Other Parameters
        ----------------
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(MultipleThresholdParameterIndexParameter, self).__init__(model, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)

        self.thresholds = []
        for threshold in thresholds:
            if not isinstance(threshold, Parameter):
                from pywr.parameters import ConstantParameter
                threshold = ConstantParameter(model, threshold)
            self.thresholds.append(threshold)

        for threshold in self.thresholds:
            self.children.add(thresholds)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """This returns the index of the first threshold the parameter value is above.

        The index is zero-based. For example, if only one threshold is
        supplied then the index is either 0 (above) or 1 (below). For two
        thresholds the index is either 0 (above both), 1 (in between), or 2 (below
        both), and so on.

        Parameters
        ----------
        timestep : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The parameter index.
        """
        cdef double value
        cdef int index, j
        cdef Parameter threshold

        value = self.parameter.get_value(scenario_index)
        index = len(self.thresholds)
        for j, threshold in enumerate(self.thresholds):
            target_threshold = threshold.get_value(scenario_index)
            if value >= target_threshold:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        MultipleThresholdParameterIndexParameter
            The loaded class.
        """
        parameter = load_parameter(model, data.pop("parameter"))
        thresholds = [load_parameter(model, d) for d in data.pop("thresholds")]
        return cls(model, parameter, thresholds, **data)
MultipleThresholdParameterIndexParameter.register()


cdef class ParameterThresholdParameter(AbstractThresholdParameter):
    """This parameter returns one of two values depending on a parameter value:

        parameter_value predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.

    Examples
    -------
    In the example below the parameter "My parameter" is compared against the constant 
    threshold `0.5`. When its value is less than `0.5`, `0` is returned, otherwise `10` is returned. 

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import ConstantParameter, ParameterThresholdParameter

    model = Model()
    parameter = ConstantParameter(model=model, value=100, name="My parameter")
    ParameterThresholdParameter(
        model=model, 
        param=parameter, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="Threshold"
    )
    ```

    JSON
    ======
    ```json
    {
        "Threshold": {
            "type": "ParameterThresholdParameter",
            "parameter": "My parameter",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare the value of the storage to
    param : Parameter
        The parameter whose value is compared against the `threshold`.
    ratchet : bool
        If true the parameter behaves like a ratchet. Once it is triggered first
        it stays in the triggered position.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, Parameter param, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        param : Parameter
            The parameter whose value is compared against the `threshold`.
       
        Other Parameters
        ----------------
        threshold : double or Parameter
            Threshold to compare the value of the recorder to.
        values : iterable of doubles
            If the predicate evaluates False the zeroth value is returned,
            otherwise the first value is returned.
        predicate : Optional[Literal["LT", "GT", "EQ", "LE", "GE"]], default="LT"
            Compare the threshold against the storage using the given predicate.
        ratchet : Optional[bool], default=False
            If true the parameter behaves like a ratchet. Once it is triggered,
            it stays in the triggered position.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(ParameterThresholdParameter, self).__init__(model, *args, **kwargs)
        self.param = param
        self.children.add(param)

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return self.param.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        ParameterThresholdParameter
            The loaded class.
        """
        param = load_parameter(model, data.pop('parameter'))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, param, threshold, values=values, predicate=predicate, **data)
ParameterThresholdParameter.register()


cdef class RecorderThresholdParameter(AbstractThresholdParameter):
    """This parameter returns one of two values depending on a recorder value at the previous timestep:

        recorder_value predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.

    Examples
    -------
    In the example below the recorder "My recorder" is compared against the constant 
    threshold `0.5`. When its value is less than `0.5`, `0` is returned, otherwise `10` is returned. 

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.recorders import RollingMeanFlowNodeRecorder
    from pywr.parameters import RecorderThresholdParameter

    model = Model()
    parameter = RollingMeanFlowNodeRecorder(
        model=model, 
        node=model.nodes["Link"],
        timesteps=5,
        name="My recorder"
    )
    RecorderThresholdParameter(
        model=model, 
        param=parameter, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="Threshold"
    )
    ```

    JSON
    ======
    ```json
    {
        "Threshold": {
            "type": "RecorderThresholdParameter",
            "parameter": "My recorder",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare the value of the storage to
    recorder : Recorder
        The recorder whose value is compared against the `threshold`.
    initial_value : float
        The value to use when the model starts, and the recorder has no value.
    ratchet : bool
        If true the recorder behaves like a ratchet. Once it is triggered,
        it stays in the triggered position.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(self,  model, Recorder recorder, *args, initial_value=1, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        recorder : Recorder
            The recorder whose value is compared against the `threshold`.
       
        Other Parameters
        ----------------
        initial_value : float
            The value to use when the model starts and the recorder has no value.
        threshold : double or Parameter
            Threshold to compare the value of the recorder to.
        values : iterable of doubles
            If the predicate evaluates False the zeroth value is returned,
            otherwise the first value is returned.
        predicate : Optional[Literal["LT", "GT", "EQ", "LE", "GE"]], default="LT"
            Compare the threshold against the storage using the given predicate.
        ratchet : Optional[bool], default=False
            If true the parameter behaves like a ratchet. Once it is triggered,
            it stays in the triggered position.
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
        super(RecorderThresholdParameter, self).__init__(model, *args, **kwargs)
        self.recorder = recorder
        self.recorder.parents.add(self)
        self.initial_value = initial_value

    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # TODO Make this a more general API on Recorder
        return self.recorder.data[timestep.index - 1, scenario_index.global_id]

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns 1 if the predicate evaluates True, else 0
        
        Parameters
        ----------
        timestep : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        int
            The value.
        """
        cdef int index = timestep.index
        cdef int ind
        if index == 0:
            # on the first day the recorder doesn't have a value so we have no
            # threshold to compare to
            ind = self.initial_value
        else:
            ind = super(RecorderThresholdParameter, self).index(timestep, scenario_index)
        return ind

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        RecorderThresholdParameter
            The loaded class.
        """
        from pywr.recorders._recorders import load_recorder  # delayed to prevent circular reference
        recorder = load_recorder(model, data.pop("recorder"))
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, recorder, threshold, values=values, predicate=predicate, **data)
RecorderThresholdParameter.register()


cdef class CurrentYearThresholdParameter(AbstractThresholdParameter):
    """
    This parameter returns one of two values depending on the year of the current timestep:

        current_year predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.

    Examples
    -------
    In the example below the parameter "Threshold" returns `0.1` when the year is above
    1967, `0.6` otherwise.

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import CurrentYearThresholdParameter

    model = Model()
    CurrentYearThresholdParameter(
        model=model,
        predicate=">",
        values=[0.6, 0.1],
        threshold=1967,
        name="Threshold"
    )
    ```

    JSON
    ======
    ```json
    {
        "Threshold": {
            "type": "CurrentYearThresholdParameter",
            "predicate": "GT",
            "values": [0.6, 0.1],
            "threshold": 1967,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare against the current year.
    values : Iterable[float
        The values to pick when the year is above or below the `threshold`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return float(timestep.year)

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        CurrentYearThresholdParameter
            The loaded class.
        """
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, threshold, values=values, predicate=predicate, **data)
CurrentYearThresholdParameter.register()


cdef class CurrentOrdinalDayThresholdParameter(AbstractThresholdParameter):
    """
    This parameter returns one of two values depending on the current proleptic Gregorian ordinal
    of the timestep, where January 1 of year 1 has ordinal 1:

        current_ordinal predicate_sign threshold

    where the predicate_sign is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign.
    When true, the parameter returns the second value of the two values, when false it returns the first
    number.

    Examples
    -------
    In the example below the parameter "Threshold" returns `0.1` when the date is above
    1/1/1976 (ordinal is 721354), `0.6` otherwise.

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import CurrentOrdinalDayThresholdParameter

    model = Model()
    CurrentOrdinalDayThresholdParameter(
        model=model,
        predicate=">",
        values=[0.6, 0.1],
        threshold=721354,
        name="Threshold"
    )
    ```

    JSON
    ======
    ```json
    {
        "Threshold": {
            "type": "CurrentOrdinalDayThresholdParameter",
            "predicate": "GT",
            "values": [0.6, 0.1],
            "threshold": 721354,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    threshold : float | Parameter
        Threshold to compare against the current ordinal.
    values : Iterable[float
        The values to pick when the year is above or below the `threshold`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    cpdef double _value_to_compare(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        return float(timestep.datetime.toordinal())

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        CurrentOrdinalDayThresholdParameter
            The loaded class.
        """
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, threshold, values=values, predicate=predicate, **data)
CurrentOrdinalDayThresholdParameter.register()

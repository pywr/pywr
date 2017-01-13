import numpy as np
cimport numpy as np
from .parameters import parameter_registry, ConstantParameter
from ._parameters import load_parameter, load_parameter_values, Parameter, IndexParameter


cdef class BaseControlCurveParameter(Parameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, AbstractStorage storage_node, control_curves):
        """

        Parameters
        ----------
        storage_node : `Storage`
            An optional `Storage` node that can be used to query the current percentage volume.
        control_curves : iterable of Parameter objects or single Parameter
            The Parameter objects to use as a control curve(s).
        """
        super(BaseControlCurveParameter, self).__init__()
        self.control_curves = control_curves
        if storage_node is None:
            raise ValueError("storage_node is required")
        self._storage_node = storage_node

    property control_curves:
        def __get__(self):
            return self._control_curves
        def __set__(self, control_curves):
            # Accept a single Parameter and convert to a list internally
            if isinstance(control_curves, Parameter):
                control_curves = [control_curves]

            # remove existing control curves (if any)
            if self._control_curves is not None:
                for control_curve in self._control_curves:
                    control_curve.parents.remove(self)

            _new_control_curves = []
            for control_curve in control_curves:
                # Accept numeric inputs and convert to `ConstantParameter`
                if isinstance(control_curve, (float, int)):
                    control_curve = ConstantParameter(control_curve)

                control_curve.parents.add(self)
                _new_control_curves.append(control_curve)
            self._control_curves = list(_new_control_curves)

    property storage_node:
        def __get__(self):
            return self._storage_node
        def __set__(self, value):
            self._storage_node = value

    @classmethod
    def _load_control_curves(cls, model, data):
        """ Private class method to load control curve data from dict. """

        control_curves = []
        if 'control_curve' in data:
            control_curves.append(load_parameter(model, data['control_curve']))
        elif 'control_curves' in data:
            for pdata in data['control_curves']:
                control_curves.append(load_parameter(model, pdata))
        return control_curves

    @classmethod
    def _load_storage_node(cls, model, data):
        """ Private class method to load storage node from dict. """
        node = model._get_node_from_ref(model, data["storage_node"])
        return node


BaseControlCurveParameter.register()


cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    """A control curve Parameter that interpolates between three or more values

    Parameters
    ----------
    storage_node : `Storage`
        The storage node to compare the control curve(s) to.
    control_curves : list of `Parameter`
        A list of parameters representing the control curve(s). These are
        often MonthlyProfileParameters or DailyProfileParameters, but may be
        any Parameter that returns values between 0.0 and 1.0.
    values : list of float
        A list of values to return corresponding to the control curves. The
        length of the list could be 1 + len(control_curves).

    For a single control curve if the storage volume is above the curve the
    first value is returned, otherwise the second value is returned. In the
    case that there are multiple curves, if the storage volume falls between
    two curves the return value is linearly interpolated between the values
    for the two curves.

    Example
    -------
    In the example below the cost of a storage node is related to it's volume.
    If the volume is above 50% the cost is 0. If the volume is between 50% and
    30% a weighted average is taken to return a cost between 0.0 and -2.0. For
    example, at 35% the cost is -1.5. Below 30% the cost is -10.0.

    >>> storage_node = Storage(model, "reservoir")
    >>> ccs = [ConstantParameter(0.5), ConstantParameter(0.3)]
    >>> values = [0.0, -2.0, -10.0]
    >>> cost = ControlCurveInterpolatedParameter(storage_node, ccs, values)
    >>> storage_node.cost = cost
    """
    def __init__(self, storage_node, control_curves, values):
        super(ControlCurveInterpolatedParameter, self).__init__(storage_node, control_curves)
        # Expected number of values is number of control curves plus one.
        nvalues = len(self.control_curves) + 2
        if len(values) != nvalues:
            raise ValueError('Length of values should be two more than the number of '
                             'control curves ({}).'.format(nvalues))
        self.values = values

    property values:
        def __get__(self):
            return np.array(self._values)
        def __set__(self, values):
            self._values = np.array(values)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = scenario_index._global_id
        cdef int j
        cdef Parameter cc_param
        cdef double cc, cc_prev
        cdef Storage node = self._storage_node
        # return the interpolated value for the current level.
        cdef double current_pc = node._current_pc[i]
        cdef double weight

        if current_pc > 1.0:
            return self._values[0]

        if current_pc < 0.0:
            return self._values[-1]

        # Assumes control_curves is sorted highest to lowest
        # First level 100%
        cc_prev = 1.0
        for j, cc_param in enumerate(self._control_curves):
            cc = cc_param.value(ts, scenario_index)
            # If level above control curve then return this level's value
            if current_pc >= cc:
                try:
                    weight = (current_pc - cc) / (cc_prev - cc)
                except ZeroDivisionError:
                    # Last two control curves identical; return the next value
                    return self._values[j+1]
                return self._values[j]*weight + self._values[j+1]*(1.0 - weight)
            # Update previous value for next iteration
            cc_prev = cc

        # Current storage is above none of the control curves
        # Therefore interpolate between last control curve and bottom
        cc = 0.0
        try:
            weight = (current_pc - cc) / (cc_prev - cc)
        except ZeroDivisionError:
            # cc_prev == cc  i.e. last control curve is close to 0%
            return self._values[-2]
        return self._values[-2]*weight + self._values[-1]*(1.0 - weight)

    @classmethod
    def load(cls, model, data):
        control_curves = super(ControlCurveInterpolatedParameter, cls)._load_control_curves(model, data)
        storage_node = super(ControlCurveInterpolatedParameter, cls)._load_storage_node(model, data)
        values = load_parameter_values(model, data)
        parameter = cls(storage_node, control_curves, values)
        return parameter
ControlCurveInterpolatedParameter.register()


cdef class ControlCurveIndexParameter(IndexParameter):
    """Multiple control curve holder which returns an index not a value

    Parameters
    ----------
    storage_node : `Storage`
    control_curves : iterable of `Parameter` instances or floats
    """
    def __init__(self, storage_node, control_curves, **kwargs):
        super(ControlCurveIndexParameter, self).__init__(**kwargs)
        self.storage_node = storage_node
        self.control_curves = control_curves

    property control_curves:
        def __get__(self):
            return self._control_curves
        def __set__(self, control_curves):
            # Accept a single Parameter and convert to a list internally
            if isinstance(control_curves, Parameter):
                control_curves = [control_curves]

            # remove existing control curves (if any)
            if self._control_curves is not None:
                for control_curve in self._control_curves:
                    control_curve.parents.remove(self)

            _new_control_curves = []
            for control_curve in control_curves:
                # Accept numeric inputs and convert to `ConstantParameter`
                if isinstance(control_curve, (float, int)):
                    control_curve = ConstantParameter(control_curve)

                control_curve.parents.add(self)
                _new_control_curves.append(control_curve)
            self._control_curves = list(_new_control_curves)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the index of the first control curve the storage is above

        The index is zero-based. For example, if only one control curve is
        supplied then the index is either 0 (above) or 1 (below). For two
        curves the index is either 0 (above both), 1 (in between), or 2 (below
        both), and so on.
        """
        cdef double current_percentage
        cdef double target_percentage
        cdef int index, j
        cdef Parameter control_curve
        current_percentage = self.storage_node._current_pc[scenario_index._global_id]
        index = len(self.control_curves)
        for j, control_curve in enumerate(self._control_curves):
            target_percentage = control_curve.value(timestep, scenario_index)
            if current_percentage >= target_percentage:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        storage_node = model._get_node_from_ref(model, data["storage_node"])
        control_curves = [load_parameter(model, data) for data in data["control_curves"]]
        return cls(storage_node, control_curves)
ControlCurveIndexParameter.register()


cdef class ControlCurveParameter(BaseControlCurveParameter):
    """ A generic multi-levelled control curve Parameter.

     This parameter can be used to return different values when a `Storage` node's current
      volumes is at different percentage of `max_volume` relative to predefined control curves.
      Control curves must be defined in the range [0, 1] corresponding to 0% and 100% volume.

     By default this parameter returns an integer sequence from zero if the first control curve
      is passed, and incrementing by one for each control curve (or "level") the `Storage` node
      is below.

    Parameters
    ----------
    storage_node : `Storage`
        An optional `Storage` node that can be used to query the current percentage volume.
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like or `None`, optional
        The values to return if the `Storage` object is above the correspond control curve.
        I.e. the first value is returned if the current volume is above the first control curve,
        and second value if above the second control curve, and so on. The length of `values`
        must be one more than than the length of `control_curves`.
    parameters : iterable `Parameter` objects or `None`, optional
        If `values` is `None` then `parameters` can specify a `Parameter` object to use at level
        of the control curves. In the same way as `values` the first `Parameter` is used if
        `Storage` is above the first control curve, and second `Parameter` if above the
        second control curve, and so on.
    variable_indices : iterable of ints, optional
        A list of indices that correspond to items in `values` which are to be considered variables
         when `self.is_variable` is True. This mechanism allows a subset of `values` to be variable.
    lower_bounds, upper_bounds : array_like, optional
        Bounds of the variables. The length must correspond to the length of `variable_indices`, i.e.
         there are bounds for each index to be considered as a variable.

    Notes
    -----
    If `values` and `parameters` are both `None`, the default, then `values` defaults to
     a range of integers, starting at zero, one more than length of `control_curves`.

    See also
    --------
    `BaseControlCurveParameter`

    """
    def __init__(self, storage_node, control_curves, values=None, parameters=None,
                 variable_indices=None, upper_bounds=None, lower_bounds=None):
        super(ControlCurveParameter, self).__init__(storage_node, control_curves)
        # Expected number of values is number of control curves plus one.
        self.size = nvalues = len(self.control_curves) + 1
        self.values = None
        self.parameters = None
        if values is not None:
            if len(values) != nvalues:
                raise ValueError('Length of values should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.values = values
        elif parameters is not None:
            if len(parameters) != nvalues:
                raise ValueError('Length of parameters should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.parameters = list(parameters)
        else:
            # No values or parameters given, default to sequence of integers
            self.values = np.arange(nvalues)

        # Default values
        self._upper_bounds = None
        self._lower_bounds = None
        self.variable_indices = variable_indices

        if variable_indices is not None:
            self.size = len(variable_indices)
        # Bounds for use as a variable (i.e. when self.is_variable = True)
        if upper_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Upper bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(upper_bounds) != self.size:
                raise ValueError('Length of upper_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.size))
            self._upper_bounds = np.array(upper_bounds)

        if lower_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Lower bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(lower_bounds) != self.size:
                raise ValueError('Length of lower_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.size))
            self._lower_bounds = np.array(lower_bounds)

    @classmethod
    def load(cls, model, data):
        control_curves = super(ControlCurveParameter, cls)._load_control_curves(model, data)
        storage_node = super(ControlCurveParameter, cls)._load_storage_node(model, data)

        parameters = None
        values = None
        if 'values' in data:
            values = load_parameter_values(model, data)
        elif 'parameters' in data:
            # Load parameters
            parameters_data = data['parameters']
            parameters = []
            for pdata in parameters_data:
                parameters.append(load_parameter(model, pdata))

        return cls(storage_node, control_curves, values=values, parameters=parameters)

    property values:
        def __get__(self):
            return np.array(self._values)
        def __set__(self, values):
            if values is not None:
                self._values = np.asarray(values, dtype=np.float64)
            else:
                self._values = None

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int j
        cdef Parameter cc_param, param
        cdef AbstractStorage node = self.node if self.storage_node is None else self.storage_node
        cdef double current_pc
        current_pc = node._current_pc[scenario_index._global_id]

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.value(ts, scenario_index)
            # If level above control curve then return this level's value
            if current_pc >= cc:
                if self.parameters is not None:
                    param = self.parameters[j]
                    return param.value(ts, scenario_index)
                else:
                    return self._values[j]

        if self.parameters is not None:
            param = self.parameters[-1]
            return param.value(ts, scenario_index)
        else:
            return self._values[-1]

    cpdef update(self, double[:] values):
        for i, v in zip(self.variable_indices, values):
            self.values[i] = v

    cpdef double[:] lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] upper_bounds(self):
        return self._upper_bounds

ControlCurveParameter.register()
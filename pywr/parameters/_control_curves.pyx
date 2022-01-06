import cython
import numpy as np
cimport numpy as np
from .parameters import parameter_registry, ConstantParameter, parameter_property
from ._parameters import load_parameter, load_parameter_values, Parameter, IndexParameter
import warnings


@cython.cdivision(True)
cpdef double _interpolate(double current_position, double lower_bound, double upper_bound, double lower_value, double upper_value):
    """Interpolation function used by PiecewiseLinearControlCurve"""
    cdef double factor
    cdef double value
    if current_position < lower_bound:
        value = lower_value
    elif current_position > upper_bound:
        value = upper_value
    elif upper_bound == lower_bound:
        value = lower_value
    else:
        factor = (current_position - lower_bound) / (upper_bound - lower_bound)
        value = lower_value + (upper_value - lower_value) * factor
    return value


cdef class BaseControlCurveParameter(Parameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, model, AbstractStorage storage_node, control_curves, **kwargs):
        """

        Parameters
        ----------
        storage_node : `Storage`
            An optional `Storage` node that can be used to query the current percentage volume.
        control_curves : iterable of Parameter objects or single Parameter
            The Parameter objects to use as a control curve(s).
        """
        super(BaseControlCurveParameter, self).__init__(model, **kwargs)
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
                    control_curve = ConstantParameter(self.model, control_curve)

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
            control_curves.append(load_parameter(model, data.pop('control_curve')))
        elif 'control_curves' in data:
            for pdata in data.pop('control_curves'):
                control_curves.append(load_parameter(model, pdata))
        return control_curves

    @classmethod
    def _load_storage_node(cls, model, data):
        """ Private class method to load storage node from dict. """
        node = model.nodes[data.pop("storage_node")]
        return node


BaseControlCurveParameter.register()


cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    """A control curve Parameter that interpolates between three or more values

    Return values are linearly interpolated between control curves, with the
    first and last value being 100% and 0% respectively.

    Parameters
    ----------
    storage_node : `Storage`
        The storage node to compare the control curve(s) to.
    control_curves : list of `Parameter` or floats
        A list of parameters representing the control curve(s). These are
        often MonthlyProfileParameters or DailyProfileParameters, but may be
        any Parameter that returns values between 0.0 and 1.0. If floats are
        passed they are converted to `ConstantParameter`.
    values : list of float
        A list of values to return corresponding to the control curves. The
        length of the list should be 2 + len(control_curves).
    parameters : iterable `Parameter` objects or `None`, optional
        If `values` is `None` then `parameters` can specify a `Parameter` object to use at each
        of the control curves. The number of parameters should be 2 + len(control_curves)

    Examples
    --------
    In the example below the cost of a storage node is related to it's volume.
    At 100% full the cost is 0. Between 100% and 50% the cost is linearly
    interpolated between 0 and -5. Between 50% and 30% the cost is interpolated
    between -5 and -10. Between 30% and 0% the cost is interpolated between -10
    and -20

    ::

        Volume:  100%             50%      30%       0%
                  |----------------|--------|--------|
          Cost:  0.0            -5.0     -10.0   -20.0


    >>> storage_node = Storage(model, "reservoir", max_volume=100, initial_volume=100)
    >>> ccs = [ConstantParameter(0.5), ConstantParameter(0.3)]
    >>> values = [0.0, -5.0, -10.0, -20.0]
    >>> cost = ControlCurveInterpolatedParameter(storage_node, ccs, values)
    >>> storage_node.cost = cost
    """
    def __init__(self, model, storage_node, control_curves, values=None, parameters=None, **kwargs):
        super(ControlCurveInterpolatedParameter, self).__init__(model, storage_node, control_curves, **kwargs)
        # Expected number of values is number of control curves plus two.
        nvalues = len(self.control_curves) + 2
        self.parameters = None

        if values is not None:
            if len(values) != nvalues:
                raise ValueError('Length of values should be two more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.values = np.asarray(values)

        elif parameters is not None:
            if len(parameters) != nvalues:
                raise ValueError('Length of parameters should be two more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.parameters = list(parameters)
            # Make sure these parameters depend on this parameter to ensure they are evaluated
            # in the correct order.
            for p in self.parameters:
                p.parents.add(self)
        else:
            raise ValueError('One of values or parameters keywords must be given.')

    property values:
        def __get__(self):
            return np.array(self._values)
        def __set__(self, values):
            self._values = np.array(values)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int j
        cdef Parameter cc_param, value_param
        cdef double cc, cc_prev
        # return the interpolated value for the current level.
        cdef double current_pc = self._storage_node.get_current_pc(scenario_index)
        cdef double weight
        cdef double[:] values  # y values to interpolate between in this time-step

        if self.parameters is not None:
            # If there are parameter use them to gather the interpolation values
            values = np.empty(len(self.parameters))
            for j, value_param in enumerate(self.parameters):
                values[j] = value_param.get_value(scenario_index)
        else:
            # Otherwise use the given array of floats.
            # This makes a reference rather than a copy.
            values = self._values

        # Assumes control_curves is sorted highest to lowest
        # First level 100%
        cc_prev = 1.0
        for j, cc_param in enumerate(self._control_curves):
            cc = cc_param.get_value(scenario_index)
            # If level above control curve then return this level's value
            if current_pc >= cc:
                try:
                    weight = (current_pc - cc) / (cc_prev - cc)
                except ZeroDivisionError:
                    # Last two control curves identical; return the next value
                    return values[j+1]
                return values[j]*weight + values[j+1]*(1.0 - weight)
            # Update previous value for next iteration
            cc_prev = cc

        # Current storage is above none of the control curves
        # Therefore interpolate between last control curve and bottom
        cc = 0.0
        try:
            weight = (current_pc - cc) / (cc_prev - cc)
        except ZeroDivisionError:
            # cc_prev == cc  i.e. last control curve is close to 0%
            return values[-2]
        return values[-2]*weight + values[-1]*(1.0 - weight)

    @classmethod
    def load(cls, model, data):
        control_curves = super(ControlCurveInterpolatedParameter, cls)._load_control_curves(model, data)
        storage_node = super(ControlCurveInterpolatedParameter, cls)._load_storage_node(model, data)
        if "parameters" in data:
            parameters = [load_parameter(model, p) for p in data.pop("parameters")]
            values = None
        else:
            values = load_parameter_values(model, data)
            parameters = None
        return cls(model, storage_node, control_curves, values=values, parameters=parameters)

ControlCurveInterpolatedParameter.register()


cdef class ControlCurvePiecewiseInterpolatedParameter(BaseControlCurveParameter):
    """A control curve Parameter that interpolates between two or more pairs of values.

    Return values are linearly interpolated between a pair of values depending on the current
    storage. The first pair is used between maximum and the first control curve, the next pair
    between the first control curve and second control curve, and so on until the last pair is
    used between the last control curve and the minimum value. The first value in each pair is the
    value at the upper position, and the second the value at the lower position.

    Parameters
    ----------
    storage_node : `Storage`
        The storage node to compare the control curve(s) to.
    control_curves : list of `Parameter` or floats
        A list of parameters representing the control curve(s). These are
        often MonthlyProfileParameters or DailyProfileParameters, but may be
        any Parameter that returns values between 0.0 and 1.0. If floats are
        passed they are converted to `ConstantParameter`.
    values : 2D array or list of lists
        A list of value pairs to interpolate between. The length of the list should be 1 + len(control_curves).
    minimum : float
        The storage considered the bottom of the lower curve, 0-1 (default=0).
    maximum : float
        The storage considered the top of the upper curve, 0-1 (default=1).

    """
    def __init__(self, model, storage_node, control_curves, values, minimum=0.0, maximum=1.0, **kwargs):
        super(ControlCurvePiecewiseInterpolatedParameter, self).__init__(model, storage_node, control_curves, **kwargs)
        self.values = np.array(values, dtype=np.float64)
        self.minimum = minimum
        self.maximum = maximum

    property values:
        def __get__(self):
            return np.array(self._values)
        def __set__(self, values):
            # Expected number of values is number of control curves plus one.
            nvalues = len(self.control_curves) + 1
            if len(values) != nvalues:
                raise ValueError('Length of values should be two more than the number of '
                                 'control curves ({}).'.format(nvalues))
            elif len(values[0]) != 2:
                raise ValueError('The second dimension of values should be of length 2.')
            self._values = np.array(values)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int j, ncc
        cdef Parameter cc_param
        cdef double cc, cc_prev, val
        # return the interpolated value for the current level.
        cdef double current_pc = self._storage_node.get_current_pc(scenario_index)

        cc_prev = self.maximum
        for j, cc_param in enumerate(self._control_curves):
            cc = cc_param.get_value(scenario_index)
            # If level above control curve then return this level's value
            if current_pc >= cc:
                return _interpolate(current_pc, cc, cc_prev, self._values[j, 1], self._values[j, 0])

            # Update previous value for next iteration
            cc_prev = cc

        # Current storage is above none of the control curves
        # Therefore interpolate between last control curve and minimum
        ncc = len(self._control_curves)
        val = _interpolate(current_pc, self.minimum, cc_prev, self._values[ncc, 1], self._values[ncc, 0])
        return val

    @classmethod
    def load(cls, model, data):
        control_curves = super(ControlCurvePiecewiseInterpolatedParameter, cls)._load_control_curves(model, data)
        storage_node = super(ControlCurvePiecewiseInterpolatedParameter, cls)._load_storage_node(model, data)
        values = load_parameter_values(model, data)
        return cls(model, storage_node, control_curves, values=values, **data)

ControlCurvePiecewiseInterpolatedParameter.register()


cdef class ControlCurveIndexParameter(IndexParameter):
    """Multiple control curve holder which returns an index not a value

    Parameters
    ----------
    storage_node : `Storage`
    control_curves : iterable of `Parameter` instances or floats
    """
    def __init__(self, model, storage_node, control_curves, **kwargs):
        super(ControlCurveIndexParameter, self).__init__(model, **kwargs)
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
                    control_curve = ConstantParameter(self.model, control_curve)

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
        current_percentage = self.storage_node.get_current_pc(scenario_index)
        index = len(self.control_curves)
        for j, control_curve in enumerate(self.control_curves):
            target_percentage = control_curve.get_value(scenario_index)
            if current_percentage >= target_percentage:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        storage_node = model.nodes[data.pop("storage_node")]
        control_curves = [load_parameter(model, d) for d in data.pop("control_curves")]
        return cls(model, storage_node, control_curves, **data)
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
    BaseControlCurveParameter
    """
    def __init__(self, model, storage_node, control_curves, values=None, parameters=None,
                 variable_indices=None, upper_bounds=None, lower_bounds=None, **kwargs):
        super(ControlCurveParameter, self).__init__(model, storage_node, control_curves, **kwargs)
        # Expected number of values is number of control curves plus one.
        nvalues = len(self.control_curves) + 1
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
            # Make sure these parameters depend on this parameter to ensure they are evaluated
            # in the correct order.
            for p in self.parameters:
                p.parents.add(self)
        else:
            # No values or parameters given, default to sequence of integers
            self.values = np.arange(nvalues)

        # Default values
        self._upper_bounds = None
        self._lower_bounds = None

        if variable_indices is not None:
            self.variable_indices = variable_indices
            self.double_size = len(variable_indices)
        else:
            self.double_size = 0
        # Bounds for use as a variable (i.e. when self.is_variable = True)
        if upper_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Upper bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(upper_bounds) != self.double_size:
                raise ValueError('Length of upper_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.double_size))
            self._upper_bounds = np.array(upper_bounds)

        if lower_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Lower bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(lower_bounds) != self.double_size:
                raise ValueError('Length of lower_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.double_size))
            self._lower_bounds = np.array(lower_bounds)

    property values:
        def __get__(self):
            return np.asarray(self._values)
        def __set__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

    property variable_indices:
        def __get__(self):
            return np.array(self._variable_indices)
        def __set__(self, values):
            self._variable_indices = np.array(values, dtype=np.int32)

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

        return cls(model, storage_node, control_curves, values=values, parameters=parameters)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int j
        cdef AbstractStorage node
        cdef double cc
        cdef Parameter param, cc_param
        node = self.node if self.storage_node is None else self.storage_node
        cdef double current_pc = node.get_current_pc(scenario_index)

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.get_value(scenario_index)
            # If level above control curve then return this level's value
            if current_pc >= cc:
                if self.parameters is not None:
                    param = self.parameters[j]
                    return param.get_value(scenario_index)
                else:
                    return self._values[j]

        if self.parameters is not None:
            param = self.parameters[-1]
            return param.get_value(scenario_index)
        else:
            return self._values[-1]

    cpdef set_double_variables(self, double[:] values):
        cdef int i
        cdef double v

        if len(values) != len(self.variable_indices):
            raise ValueError('Number of values must be the same as the number of variable_indices.')

        if self.double_size != 0:
            for i, v in zip(self.variable_indices, values):
                self._values[i] = v

    cpdef double[:] get_double_variables(self):
        cdef int i, j
        cdef double v
        cdef double[:] arry = np.empty((len(self.variable_indices), ))
        for i, j in enumerate(self.variable_indices):
            arry[i] = self._values[j]
        return arry

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds

ControlCurveParameter.register()

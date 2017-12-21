import numpy as np
cimport numpy as np
from .parameters import parameter_registry, ConstantParameter, parameter_property
from ._parameters import load_parameter, load_parameter_values, Parameter, IndexParameter

# http://stackoverflow.com/a/20031818/1300519
cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x)

cdef class PiecewiseLinearControlCurve(Parameter):
    """Piecewise function composed of two linear curves
    
    Parameters
    ----------
    model : Model
    storage_node : Storage
    control_curve : Parameter
    values : [(float, float), (float, float)]
        Iterable of 2-tuples, representing the lower and upper value of the
        linear interpolation below and above the control curve, respectively.
    minimum : float
        The storage considered the bottom of the lower curve, 0-1 (default=0).
    maximum : float
        The storage considered the top of the upper curve, 0-1 (default=1).
    """
    def __init__(self, model, storage_node, control_curve, values, minimum=0.0, maximum=1.0, *args, **kwargs):
        super(PiecewiseLinearControlCurve, self).__init__(model, *args, **kwargs)
        self._control_curve = None
        self.storage_node = storage_node
        self.control_curve = control_curve
        self.below_lower, self.below_upper = values[0]
        self.above_lower, self.above_upper = values[1]
        self.minimum = minimum
        self.maximum = maximum

    property control_curve:
        def __get__(self):
            return self._control_curve
        def __set__(self, parameter):
            if self._control_curve:
                self.children.remove(self._control_curve)
            self.children.add(parameter)
            self._control_curve = parameter

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef double value
        cdef double control_curve = self._control_curve.get_value(scenario_index)
        cdef double current_pc = self.storage_node._current_pc[scenario_index.global_id]
        if current_pc > control_curve:
            value = _interpolate(current_pc, control_curve, self.maximum, self.above_lower, self.above_upper)
        else:
            value = _interpolate(current_pc, self.minimum, control_curve, self.below_lower, self.below_upper)
        return value

    cpdef reset(self):
        super(PiecewiseLinearControlCurve, self).setup()
        assert self.maximum > self.minimum
        assert np.isfinite(self.below_lower)
        assert np.isfinite(self.below_upper)
        assert np.isfinite(self.above_lower)
        assert np.isfinite(self.above_upper)

    @classmethod
    def load(cls, model, data):
        storage_node = model._get_node_from_ref(model, data["storage_node"])
        control_curve = load_parameter(model, data["control_curve"])
        values = data["values"]
        kwargs = {}
        if "minimum" in data.keys():
            kwargs["minimum"] = data["minimum"]
        if "maximum" in data.keys():
            kwargs["maximum"] = data["maximum"]
        parameter = cls(model, storage_node, control_curve, values, **kwargs)
        return parameter

PiecewiseLinearControlCurve.register()

cpdef _interpolate(double current_position, double lower_bound, double upper_bound, double lower_value, double upper_value):
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
    control_curves : list of `Parameter` or floats
        A list of parameters representing the control curve(s). These are
        often MonthlyProfileParameters or DailyProfileParameters, but may be
        any Parameter that returns values between 0.0 and 1.0. If floats are
        passed they are converted to `ConstantParameter`.
    values : list of float
        A list of values to return corresponding to the control curves. The
        length of the list should be 2 + len(control_curves).

    Return values are linearly interpolated between control curves, with the
    first and last value being 100% and 0% respectively.

    Example
    -------
    In the example below the cost of a storage node is related to it's volume.
    At 100% full the cost is 0. Between 100% and 50% the cost is linearly
    interpolated between 0 and -5. Between 50% and 30% the cost is interpolated
    between -5 and -10. Between 30% and 0% the cost is interpolated between -10
    and -20.

    Volume:  100%            50%      30%       0%
             |...............|........|..........|
      Cost:  0.0            -5.0     -10.0   -20.0

    >>> storage_node = Storage(model, "reservoir", max_volume=100, initial_volume=100)
    >>> ccs = [ConstantParameter(0.5), ConstantParameter(0.3)]
    >>> values = [0.0, -5.0, -10.0, -20.0]
    >>> cost = ControlCurveInterpolatedParameter(storage_node, ccs, values)
    >>> storage_node.cost = cost
    """
    def __init__(self, model, storage_node, control_curves, values):
        super(ControlCurveInterpolatedParameter, self).__init__(model, storage_node, control_curves)
        # Expected number of values is number of control curves plus two.
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
        cdef int i = scenario_index.global_id
        cdef int j
        cdef Parameter cc_param
        cdef double cc, cc_prev
        cdef Storage node = self._storage_node
        # return the interpolated value for the current level.
        cdef double current_pc = node._current_pc[i]
        cdef double weight

        if current_pc > 1.0 or npy_isnan(current_pc):
            return self._values[0]

        if current_pc < 0.0:
            return self._values[-1]

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
        parameter = cls(model, storage_node, control_curves, values)
        return parameter

ControlCurveInterpolatedParameter.register()

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
        current_percentage = self.storage_node._current_pc[scenario_index.global_id]
        index = len(self.control_curves)
        for j, control_curve in enumerate(self.control_curves):
            target_percentage = control_curve.get_value(scenario_index)
            if current_percentage >= target_percentage:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        storage_node = model._get_node_from_ref(model, data["storage_node"])
        control_curves = [load_parameter(model, data) for data in data["control_curves"]]
        return cls(model, storage_node, control_curves)
ControlCurveIndexParameter.register()

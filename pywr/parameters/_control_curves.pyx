import numpy as np
cimport numpy as np
from .parameters import parameter_registry, ConstantParameter
from ._parameters import load_parameter, load_parameter_values

from ._parameters cimport Parameter, IndexParameter
from ._parameters import Parameter, IndexParameter


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


parameter_registry.add(BaseControlCurveParameter)


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

parameter_registry.add(ControlCurveInterpolatedParameter)

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

        # update dependency tree
        for control_curve in control_curves:
            control_curve.parents.add(self)

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
        current_percentage = self.storage_node.current_pc[scenario_index.global_id]
        index = len(self.control_curves)
        for j, control_curve in enumerate(self.control_curves):
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
parameter_registry.add(ControlCurveIndexParameter)

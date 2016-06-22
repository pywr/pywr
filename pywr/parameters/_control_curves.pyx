import numpy as np
cimport numpy as np
from .parameters import parameter_registry, ConstantParameter
from ._parameters import load_parameter, load_parameter_values


cdef class BaseControlCurveParameter(Parameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, Storage storage_node, control_curves):
        """

        Parameters
        ----------
        control_curves : iterable of Parameter objects or single Parameter
            The Parameter objects to use as a control curve(s).
        storage_node : `Storage` or `None`, optional
            An optional `Storage` node that can be used to query the current percentage volume. If
            not specified it is assumed that this object is attached to a `Storage` node and therefore
            `self.node` is used.
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
        return model.node[data['storage_node']]


parameter_registry.add(BaseControlCurveParameter)


cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    """ A control curve Parameter that interpolates between three values.
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
        values = load_parameter_values(data["values"])
        parameter = cls(control_curves, values)
        return parameter

parameter_registry.add(ControlCurveInterpolatedParameter)

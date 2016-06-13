import numpy as np
cimport numpy as np
from .parameters import parameter_registry

cdef class BaseControlCurveParameter(Parameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, control_curve, Storage storage_node=None):
        """

        Parameters
        ----------
        control_curve : Parameter
            The Parameter object to use as a control_curve. It should not be shared with other
            Nodes and Parameters because this object becomes control_curve.parent
        """
        super(BaseControlCurveParameter, self).__init__()
        if control_curve.parent is not None:
            raise RuntimeError('control_curve already has a parent.')
        control_curve.parent = self
        self._control_curve = control_curve
        self._storage_node = storage_node

    cpdef setup(self, model):
        self.control_curve.setup(model)
        super(BaseControlCurveParameter, self).setup(model)

    cpdef reset(self):
        self.control_curve.reset()
        super(BaseControlCurveParameter, self).reset()

    cpdef before(self, Timestep ts):
        self.control_curve.before(ts)
        super(BaseControlCurveParameter, self).before(ts)

    cpdef after(self, Timestep ts):
        self.control_curve.after(ts)
        super(BaseControlCurveParameter, self).after(ts)

    property control_curve:
        def __get__(self):
            return self._control_curve
        def __set__(self, value):
            self._control_curve = value

    property storage_node:
        def __get__(self):
            return self._storage_node
        def __set__(self, value):
            self._storage_node = value

parameter_registry.add(BaseControlCurveParameter)


cdef class ControlCurveInterpolatedParameter(BaseControlCurveParameter):
    """ A control curve Parameter that interpolates between three values.
    """
    def __init__(self, control_curve, values):
        super(ControlCurveInterpolatedParameter, self).__init__(control_curve)
        values = np.array(values)
        if len(values) != 3:
            raise ValueError("Three values must be given to define the interpolation knots.")
        self.lower_value, self.curve_value, self.upper_value = values

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = scenario_index._global_id
        cdef double control_curve = self._control_curve.value(ts, scenario_index)
        cdef Storage node = self.node if self._storage_node is None else self._storage_node
        # return the interpolated value for the current level.
        cdef double current_pc = node._current_pc[i]
        cdef double weight
        if current_pc < 0.0:
            return self.lower_value
        elif current_pc < control_curve:
            weight = (control_curve - current_pc) / control_curve
            return self.lower_value*weight + self.curve_value*(1.0 - weight)
        elif control_curve == 1.0:
            return self.curve_value
        elif current_pc <= 1.0:
            weight = (1.0 - current_pc) / (1.0 - control_curve)
            return self.curve_value*weight + self.upper_value*(1.0 - weight)
        else:
            return self.upper_value
parameter_registry.add(ControlCurveInterpolatedParameter)

import numpy as np
cimport numpy as np

cdef class BaseParameterControlCurve(Parameter):
    """ Base class for all Parameters that rely on a the attached Node containing a control_curve Parameter

    """
    def __init__(self, control_curve):
        """

        Parameters
        ----------
        control_curve : Parameter
            The Parameter object to use as a control_curve. It should not be shared with other
            Nodes and Parameters because this object becomes control_curve.parent
        """
        if control_curve.parent is not None:
            raise RuntimeError('control_curve already has a parent.')
        control_curve.parent = self
        self._control_curve = control_curve

    cpdef setup(self, model):
        self.control_curve.setup(model)
        super(BaseParameterControlCurve, self).setup(model)

    cpdef reset(self):
        self.control_curve.reset()
        super(BaseParameterControlCurve, self).reset()

    cpdef before(self, Timestep ts):
        self.control_curve.before(ts)
        super(BaseParameterControlCurve, self).before(ts)

    cpdef after(self, Timestep ts):
        self.control_curve.after(ts)
        super(BaseParameterControlCurve, self).after(ts)

    property control_curve:
        def __get__(self):
            return self._control_curve
        def __set__(self, value):
            self._control_curve = value


cdef class ParameterControlCurveInterpolated(BaseParameterControlCurve):
    """ A control curve Parameter that interpolates between three values.
    """
    def __init__(self, control_curve, values):
        super(ParameterControlCurveInterpolated, self).__init__(control_curve)
        values = np.array(values)
        if len(values) != 3:
            raise ValueError("Three values must be given to define the interpolation knots.")
        self.lower_value, self.curve_value, self.upper_value = values

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        cdef int i = self.node.model.scenarios.ravel_indices(scenario_indices)
        cdef double control_curve = self._control_curve.value(ts, scenario_indices)

        # return the interpolated value for the current level.
        cdef double current_pc = self.node._current_pc[i]
        cdef double weight
        if current_pc < 0.0:
            raise ValueError("Storage out of lower bounds.")
        elif current_pc < control_curve:
            weight = (control_curve - current_pc) / control_curve
            return self.lower_value*weight + self.curve_value*(1.0 - weight)
        elif control_curve == 1.0:
            return self.curve_value
        elif current_pc <= 1.0:
            weight = (1.0 - current_pc) / (1.0 - control_curve)
            return self.curve_value*weight + self.upper_value*(1.0 - weight)
        else:
            raise ValueError("Storage out of upper bounds.")


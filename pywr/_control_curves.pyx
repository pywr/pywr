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
        self.values = values
        # x values of interp, the parameter will updated the middle entry during execution
        self._interp_values = np.array([0.0, 0.5, 1.0])

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        cdef int i = self.node.model.scenarios.ravel_indices(scenario_indices)
        self._interp_values[1] = self._control_curve.value(ts, scenario_indices)
        # return the interpolated value for the current level.
        return np.interp(self.node.current_pc[i], self._interp_values, self.values)

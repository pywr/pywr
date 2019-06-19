""" Parameters for """
from ._parameters import load_parameter, ConstantParameter


cpdef double inverse_hydropower_calculation(double power, double water_elevation, double turbine_elevation, double efficiency,
                                    double flow_unit_conversion=1.0, double energy_unit_conversion=1e-6,
                                    double density=1000.0):
    """
    Calculate the flow required to produce power using the hydropower equation.
    
   
    Parameters
    ----------
    power: double
        Hydropower requirement in rate in units of energy per day.
    water_elevation : double
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : double
        An efficiency scaling factor for the power output of the turbine.
    flow_unit_conversion : double (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of $m^3/day$
    energy_unit_conversion : double (default=1e-6)
        A factor used to transform the units of power. Defaults to 1e-6 to assuming input of $MJ$/day. 
    density : double (default=1000)
        Density of water in $kgm^{-3}$.
        
    Returns
    -------
    flow : double 
        Required flow rate of water through the turbine. Converted using `flow_unit_conversion` to 
        units of $m^3£ per day (not per second).
    
    Notes
    -----
    The inverse hydropower calculation uses the following equation.
    
    .. math:: q = \frac{P}{\rho * g * \deltaH}
    
    The energy rate in should be converted to units of energy per day.    
    
    """
    cdef double head
    cdef double flow

    head = water_elevation - turbine_elevation
    if head < 0.0:
        head = 0.0

    # Power
    try:
        flow = power / (energy_unit_conversion * density * 9.81 * head * efficiency * flow_unit_conversion)
    except ZeroDivisionError:
        flow = float('inf')

    return flow


cdef class HydropowerTargetParameter(Parameter):
    """ A parameter that returns flow from a hydropower generation target.

    This parameter calculates the flow required to generate a particular hydropower production target. It
    is intended to be used on a node representing a turbine where a particular production target is required
    each time-step.

    Parameters
    ----------

    target : Parameter instance
        Hydropower production target. Units should be in units of energy per day.
    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    max_flow : Parameter instance (default=None)
        Upper bounds on the calculated flow. If set the flow returned by this parameter is at most the value
        of the max_flow parameter.
    min_flow : Parameter instance (default=None)
        Lower bounds on the calculated flow. If set the flow returned by this parameter is at least the value
        of the min_flow parameter.
    min_head : double (default=0.0)
        Minimum head for flow to occur. If actual head is less than this value zero flow is returned.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/day`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MJ`.

    Notes
    -----
    The inverse hydropower calculation uses the following equation.

    .. math:: q = \\frac{P}{\\rho * g * \\delta H}

    The energy rate in should be converted to units of energy per day. The returned flow rate in should is
    converted from units of :math:`m^3` per day to those used by the model using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.

    See Also
    --------
    pywr.recorders.TotalHydroEnergyRecorder
    pywr.recorders.HydropowerRecorder

    """
    def __init__(self, model, target, water_elevation_parameter=None, max_flow=None, min_flow=None,
                 turbine_elevation=0.0, efficiency=1.0, density=1000, min_head=0.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerTargetParameter, self).__init__(model, **kwargs)

        self.target = target
        self.water_elevation_parameter = water_elevation_parameter
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.min_head = min_head
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    property target:
        def __get__(self):
            return self._target
        def __set__(self, parameter):
            if self._target:
                self.children.remove(self._target)
            self.children.add(parameter)
            self._target = parameter

    property max_flow:
        def __get__(self):
            return self._max_flow
        def __set__(self, parameter):
            if self._max_flow:
                self.children.remove(self._max_flow)
            self.children.add(parameter)
            self._max_flow = parameter

    property min_flow:
        def __get__(self):
            return self._min_flow
        def __set__(self, parameter):
            if self._min_flow:
                self.children.remove(self._min_flow)
            self.children.add(parameter)
            self._min_flow = parameter

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i
        cdef double q, head, power

        power = self._target.get_value(scenario_index)

        if self._water_elevation_parameter is not None:
            head = self._water_elevation_parameter.get_value(scenario_index)
            if self.turbine_elevation is not None:
                head -= self.turbine_elevation
        elif self.turbine_elevation is not None:
            head = self.turbine_elevation
        else:
            raise ValueError('One or both of storage_node or level must be set.')

        # -ve head is not valid
        head = max(head, 0.0)

        # Apply minimum head threshold.
        if head < self.min_head:
            return 0.0

        # Get the flow from the current node
        q = inverse_hydropower_calculation(power, head, 0.0, self.efficiency, density=self.density,
                                           flow_unit_conversion=self.flow_unit_conversion,
                                           energy_unit_conversion=self.energy_unit_conversion)

        # Bound the flow if required
        if self._max_flow is not None:
            q = min(self._max_flow.get_value(scenario_index), q)
        if self._min_flow is not None:
            q = max(self._min_flow.get_value(scenario_index), q)

        assert q >= 0.0

        return q

    @classmethod
    def load(cls, model, data):

        target = load_parameter(model, data.pop("target"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        if "max_flow" in data:
            max_flow = load_parameter(model, data.pop("max_flow"))
        else:
            max_flow = None

        if "min_flow" in data:
            min_flow = load_parameter(model, data.pop("min_flow"))
        else:
            min_flow = None

        return cls(model, target, water_elevation_parameter=water_elevation_parameter,
                   max_flow=max_flow, min_flow=min_flow, **data)
HydropowerTargetParameter.register()

from ._recorders import NumpyArrayNodeRecorder


cpdef double hydropower_calculation(double flow, double water_elevation, double turbine_elevation, double efficiency,
                                    double flow_unit_conversion=1.0, double energy_unit_conversion=1e-6,
                                    double density=1000.0):
    """
    Calculate the total power produced using the hydropower equation.
    
   
    Parameters
    ----------
    flow : double 
        Flow rate of water through the turbine. Should be converted using `flow_unit_conversion` to 
        units of $m^3£ per day (not per second).
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
        A factor used to transform the units of power. Defaults to 1e-6 to return $MJ$/day. 
    density : double (default=1000)
        Density of water in $kgm^{-3}$.
        
    Returns
    -------
    power : double
        Hydropower production rate in units of energy per day.
    
    Notes
    -----
    The hydropower calculation uses the following equation.
    
    .. math:: P = \rho * g * \deltaH * q
    
    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.    
    
    """
    cdef double head
    cdef double power
    cdef double q

    head = water_elevation - turbine_elevation
    if head < 0.0:
        head = 0.0

    # Convert flow to correct units (typically to m3/day)
    q = flow * flow_unit_conversion
    # Power
    power = density * q * 9.81 * head * efficiency

    return power * energy_unit_conversion


cdef class HydropowerRecorder(NumpyArrayNodeRecorder):
    """ Calculates the power production using the hydropower equation

    This recorder saves an array of the hydrpower production in each timestep. It can be converted to a dataframe
    after a model run has completed. It does not calculate total energy production.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
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
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    TotalHydroEnergyRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
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

    cpdef after(self):
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

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
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._data[ts.index, scenario_index.global_id] = power

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model.nodes[data.pop("node")]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
HydropowerRecorder.register()


cdef class TotalHydroEnergyRecorder(BaseConstantNodeRecorder):
    """ Calculates the total energy production using the hydropower equation from a model run.

    This recorder saves the total energy production in each scenario during a model run. It does not save a timeseries
    or power, but rather total energy.

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
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
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    HydropowerRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(TotalHydroEnergyRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
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

    cpdef after(self):
        """ Calculate the  """
        cdef int i
        cdef double q, head, power
        cdef Timestep ts = self.model.timestepper.current
        cdef double days = ts.days
        cdef ScenarioIndex scenario_index
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

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
            # Get the flow from the current node
            q = self._node._flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._values[scenario_index.global_id] += power * days

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter
        node = model.nodes[data.pop("node")]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
TotalHydroEnergyRecorder.register()

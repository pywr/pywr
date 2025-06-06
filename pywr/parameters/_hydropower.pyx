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
    """A parameter that returns flow from a hydropower generation target.

    This parameter calculates the flow required to generate a given hydropower production target `P`. It
    is intended to be used on a node representing a turbine where a particular production target
    is required at each time-step. The parameter uses the following (hydropower) equation to calculate
    the flow `q` required to produce `P`:

    q = P / ( C<sub>E</sub> * ρ * g * H * δ * C<sub>F</sub>)

    where:

    - `q` is the flow needed to achieve `P`.
    - `P` is the desired hydropower production target.
    - C<sub>E</sub> is a coefficient to convert the energy unit.
    - `ρ` is the water density.
    - `g` is the gravitational acceleration (9.81 m s<sup>-2</sup>).
    - `H` is the turbine head. If `water_elevation` is given, then the head is the difference between `water_elevation`
        and `turbine_elevation`. If `water_elevation` is not provided, then the head is simply `turbine_elevation`.
    - `δ` is the turbine efficiency.
    - C<sub>E</sub> is a coefficient to convert the flow unit. Use the `flow_unit_conversion` parameter to convert `q`
        from units of m<sup>3</sup> day<sup>-1</sup> to those used by the model.
        
    Attributes
    ----------
    model : Model
        The model instance.
    target : Parameter
        Hydropower production target. Units should be in units of energy per day.
    water_elevation_parameter : Optional[Parameter]
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    max_flow : Optional[Parameter]
        Upper bounds on the calculated flow. If set the flow returned by this parameter is at most the value
        of the max_flow parameter.
    min_flow : Optional[Parameter]
        Lower bounds on the calculated flow. If set the flow returned by this parameter is at least the value
        of the min_flow parameter.
    min_head : float
        Minimum head for flow to occur. If actual head is less than this value zero flow is returned.
    turbine_elevation : float
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float
        The efficiency of the turbine.
    density : float
        The density of water.
    flow_unit_conversion : float
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of `m^3/day`.
    energy_unit_conversion : float
        A factor used to transform the units of total energy. 
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    See Also
    --------
    [pywr.recorders.TotalHydroEnergyRecorder][]
    [pywr.recorders.HydropowerRecorder][]
    """
    def __init__(self, model, target, water_elevation_parameter=None, max_flow=None, min_flow=None,
                 turbine_elevation=0.0, efficiency=1.0, density=1000, min_head=0.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        target : Parameter
            Hydropower production target. Units should be in units of energy per day.
        water_elevation_parameter : Optional[Parameter], default=None
            Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
            the working head of the turbine.
        max_flow : Optional[Parameter], default=None
            Upper bounds on the calculated flow. If set the flow returned by this parameter is at most the value
            of the max_flow parameter.
        min_flow : Optional[Parameter], default=None
            Lower bounds on the calculated flow. If set the flow returned by this parameter is at least the value
            of the min_flow parameter.
        turbine_elevation : Optional[float], default=0
            Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
            the working head of the turbine.
        efficiency : Optional[float], default=1
            The efficiency of the turbine.
        density : Optional[float], default=1000
            The density of water.
        min_head : Optional[float], default=0
            Minimum head for flow to occur. If actual head is less than this value zero flow is returned.
        flow_unit_conversion : Optional[float], default=1
            A factor used to transform the units of flow to be compatible with the equation here. This
            should convert flow to units of `m^3/day`
        energy_unit_conversion : Optional[float], default=1e-6
            A factor used to transform the units of total energy. Defaults to 1e-6 to return `MJ`.

        Other Parameters
        ----------------
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.
        """
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
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
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
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        HydropowerTargetParameter
            The loaded class.
        """

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

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
    """This recorder calculates the power production using the hydropower equation:

    P = q *  C<sub>F</sub> * ρ * g * H * δ  * C<sub>E</sub>

    where:

    - `P` is the hydropower production.
    - `q` is the turbine flow.
    - C<sub>F</sub> is a coefficient to convert the flow unit. Use the `flow_unit_conversion` parameter to convert `q`
        from units of m<sup>3</sup> day<sup>-1</sup> to those used by the model.
    - C<sub>E</sub> is a coefficient to convert the energy unit.
    - `ρ` is the water density.
    - `g` is the gravitational acceleration (9.81 m s<sup>-2</sup>).
    - `H` is the turbine head. If `water_elevation` is given, then the head is the difference between `water_elevation`
        and `turbine_elevation`. If `water_elevation` is not provided, then the head is simply `turbine_elevation`.
    - `δ` is the turbine efficiency.

    This recorder saves an array of the hydropower production in each timestep. It can be converted to a dataframe
    after a model run has completed.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.recorders import HydropowerRecorder

    model = Model()
    node = Link(model=model, name="Release")
    HydropowerRecorder(
        model=model,
        node=node,
        efficiency=0.98,
        turbine_elevation=10.3,
        name="Power"
    )
    ```

    JSON
    ======
    ```json
    {
        "Power": {
            "type": "HydropowerRecorder",
            "node": "Release",
            "efficiency": 0.98,
            "turbine_elevation": 10.3,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    water_elevation_parameter : Optional[Parameter]
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
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
    temporal_agg_func : str | Callable
        An aggregation function used to aggregate over time when computing a value per scenario in the
        `value()` method.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.

    See Also
    --------
    [pywr.recordersTotalHydroEnergyRecorder][]
    [pywr.parameters.HydropowerTargetParameter][]
    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.
        water_elevation_parameter : Optional[Parameter], default=None
            Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
            the working head of the turbine.
        turbine_elevation : Optional[float], default=0
            Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
            the working head of the turbine.
        efficiency : Optional[float], default=1
            The efficiency of the turbine.
        density : Optional[float], default=1000
            The density of water.
        flow_unit_conversion : Optional[float], default=1
            A factor used to transform the units of flow to be compatible with the equation here. This
            should convert flow to units of `m^3/day`
        energy_unit_conversion : Optional[float], default=1e-6
            A factor used to transform the units of total energy. Defaults to 1e-6 to return `MJ`.

        Other parameters
        ----------------
        temporal_agg_func : str | Callable, default="mean"
            An aggregation function used to aggregate over time when computing a value per scenario in the
            `value()` method. This can be used to return, for example, the median flow for a scenario.
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        super(HydropowerRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        """The water elevation parameter.

        **Setter:** set the parameter instance.
        """
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    cpdef after(self):
        """Apply the hydropower production equation to the timestep."""
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
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        HydropowerRecorder
            The loaded class.
        """
        from pywr.parameters import load_parameter
        node = model.nodes[data.pop("node")]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
HydropowerRecorder.register()


cdef class TotalHydroEnergyRecorder(BaseConstantNodeRecorder):
    """This recorder calculates the total energy production using the hydropower equation:

    E = q *  C<sub>F</sub> * ρ * g * H * δ  * C<sub>E</sub> * days

    where:

    - `E` is the energy.
    - `q` is the turbine flow.
    - C<sub>F</sub> is a coefficient to convert the flow unit. Use the `flow_unit_conversion` parameter to convert `q`
        from units of m<sup>3</sup> day<sup>-1</sup> to those used by the model.
    - C<sub>E</sub> is a coefficient to convert the energy unit.
    - `ρ` is the water density.
    - `g` is the gravitational acceleration (9.81 m s<sup>-2</sup>).
    - `H` is the turbine head. If `water_elevation` is given, then the head is the difference between `water_elevation`
        and `turbine_elevation`. If `water_elevation` is not provided, then the head is simply `turbine_elevation`.
    - `δ` is the turbine efficiency.
    - `days` is the number of days from the previous timestep.

    The recorder saves the used energy at each time step and scenario and at the end of the runs returns the total
    energy used in each scenario.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.recorders import TotalHydroEnergyRecorder

    model = Model()
    node = Link(model=model, name="Release")
    TotalHydroEnergyRecorder(
        model=model,
        node=node,
        efficiency=0.98,
        turbine_elevation=10.3,
        name="Energy"
    )
    ```

    JSON
    ======
    ```json
    {
        "Energy": {
            "type": "TotalHydroEnergyRecorder",
            "node": "Release",
            "efficiency": 0.98,
            "turbine_elevation": 10.3,
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    water_elevation_parameter : Optional[Parameter]
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
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
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.

    See Also
    --------
    [pywr.recorders.HydropowerRecorder][]
    [pywr.parameters.HydropowerTargetParameter][]

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        """
        Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            Node instance to record.
        water_elevation_parameter : Optional[Parameter], default=None
            Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
            the working head of the turbine.
        turbine_elevation : Optional[float], default=0
            Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
            the working head of the turbine.
        efficiency : Optional[float], default=1
            The efficiency of the turbine.
        density : Optional[float], default=1000
            The density of water.
        flow_unit_conversion : Optional[float], default=1
            A factor used to transform the units of flow to be compatible with the equation here. This
            should convert flow to units of `m^3/day`
        energy_unit_conversion : Optional[float], default=1e-6
            A factor used to transform the units of total energy. Defaults to 1e-6 to return `MJ`.

        Other parameters
        ----------------
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        super(TotalHydroEnergyRecorder, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    property water_elevation_parameter:
        """The water elevation parameter.

        **Setter:** set the parameter instance.
        """
        def __get__(self):
            return self._water_elevation_parameter
        def __set__(self, parameter):
            if self._water_elevation_parameter:
                self.children.remove(self._water_elevation_parameter)
            self.children.add(parameter)
            self._water_elevation_parameter = parameter

    cpdef after(self):
        """Apply the hydropower production equation to the timestep."""
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
        """Load the recorder from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        TotalHydroEnergyRecorder
            The loaded class.
        """
        from pywr.parameters import load_parameter
        node = model.nodes[data.pop("node")]
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
TotalHydroEnergyRecorder.register()

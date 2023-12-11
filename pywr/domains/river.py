import pandas as pd 
import numpy as np

from pywr.nodes import (
    Domain,
    Input,
    Output,
    Link,
    Storage,
    PiecewiseLink,
    NodeMeta,
    MultiSplitLink,
    AggregatedNode
)

from pywr.parameters import (
    pop_kwarg_parameter,
    ConstantParameter,
    Parameter,
    load_parameter,
    MonthlyProfileParameter,
    InterpolatedVolumeParameter,
    AggregatedParameter,
    ScenarioWrapperParameter
)

from pywr.recorders import (
    NumpyArrayNodeRecorder
)

from pywr.parameters.control_curves import (
    ControlCurveParameter,
    ControlCurveInterpolatedParameter
)

DEFAULT_RIVER_DOMAIN = Domain(name="river", color="#33CCFF")

import logging
LOG = logging.getLogger(__name__)

class RiverDomainMixin(object):
    def __init__(self, *args, **kwargs):
        # if 'domain' not in kwargs:
        #     kwargs['domain'] = DEFAULT_RIVER_DOMAIN
        if "color" not in kwargs:
            self.color = "#6ECFF6"  # blue
        super(RiverDomainMixin, self).__init__(*args, **kwargs)


class Catchment(RiverDomainMixin, Input):
    """A hydrological catchment, supplying water to the river network"""

    __parameter_attributes__ = ("cost", "flow")

    def __init__(self, *args, **kwargs):
        """Initialise a new Catchment node.

        A Catchment is an input node with a fixed inflow. I.e. min_flow and
        max_flow are the same. The value is specified as a flow keyword, and
        overrides any min_flow or max_flow keyword arguments.

        Parameters
        ----------
        flow : float or function
            The amount of water supplied by the catchment each timestep
        """
        self.color = "#82CA9D"  # green
        # Grab flow from kwargs
        flow = kwargs.pop("flow", 0.0)
        # Min/max flow set in super inits
        super(Catchment, self).__init__(*args, **kwargs)
        self.flow = flow

    def get_flow(self, timestep):
        """flow is ensured that both min_flow and max_flow are the same."""
        return self.get_min_flow(timestep)

    def __setattr__(self, name, value):
        if name == "flow":
            self.min_flow = value
            self.max_flow = value
            return
        super(Catchment, self).__setattr__(name, value)


class Reservoir(RiverDomainMixin, Storage):
    """A reservoir node with control curve.

    The Reservoir is a subclass of Storage with additional functionality to provide a
    simple control curve. The Reservoir has above_curve_cost when it is above its curve
    and the user defined cost when it is below. Typically the costs are negative
    to represent a benefit of filling the reservoir when it is below its curve.

    A reservoir can also be used to simplify evaporation and rainfall by creating
    these nodes internally when the following properties are set:
    TODO: explain these and how they relate to each other. Please be as explicit as possible
    TODO: what format should each of these take? Perhaps provide a simple example
    - bathymetry
        - volume
        - level
        - area
        - weather_cost
        - evaporation_cost
        - rainfall_cost
    """

    def __init__(self, model, *args, **kwargs):
        """

        Keywords:
            control_curve - A Parameter object that can return the control curve position,
                as a percentage of fill, for the given timestep.
        """
        control_curve = pop_kwarg_parameter(kwargs, "control_curve", None)
        above_curve_cost = kwargs.pop("above_curve_cost", None)
        cost = kwargs.pop("cost", 0.0)
        if above_curve_cost is not None:
            if control_curve is None:
                # Make a default control curve at 100% capacity
                control_curve = ConstantParameter(model, 1.0)
            elif not isinstance(control_curve, Parameter):
                # Assume parameter is some kind of constant and coerce to ConstantParameter
                control_curve = ConstantParameter(model, control_curve)

            if not isinstance(cost, Parameter):
                # In the case where an above_curve_cost is given and cost is not a Parameter
                # a default cost Parameter is created.
                kwargs["cost"] = ControlCurveParameter(
                    model, self, control_curve, [above_curve_cost, cost]
                )
            else:
                raise ValueError(
                    "If an above_curve_cost is given cost must not be a Parameter."
                )
        else:
            # reinstate the given cost parameter to pass to the parent constructors
            kwargs["cost"] = cost

        bathymetry = kwargs.pop('bathymetry', None)
        volume = kwargs.pop('volume', None)
        level = kwargs.pop('level', None)
        area = kwargs.pop('area', None)
        self.weather_cost = kwargs.pop('weather_cost', -999)
        self.evaporation_cost = kwargs.pop('evaporation_cost', -999)
        self.rainfall_cost = kwargs.pop('rainfall_cost', -999)
        const = kwargs.pop('const', 1e6 * 1e-3 * 1e-6)

        # Pywr Storage does not expect a 'weather' kwargs, so move this to instance
        self.weather = kwargs.pop("weather", None)


        super(Reservoir, self).__init__(model, *args, **kwargs)


        self.const = ConstantParameter(model, const)

        self.rainfall_node = None
        self.rainfall_recorder = None
        self.evaporation_node = None
        self.evaporation_recorder = None

    @classmethod
    def pre_load(cls, model, data):

        bathymetry = data.pop("bathymetry", None)
        name = data.pop("name")
        node = cls(name=name, model=model, **data)

        if bathymetry is not None:
            if isinstance(bathymetry, str):
                bathymetry = load_parameter(model, bathymetry)
                volumes = bathymetry.dataframe['volume'].astype(np.float64)
                levels = bathymetry.dataframe['level'].astype(np.float64)
                areas = bathymetry.dataframe['area'].astype(np.float64)
            else:
                bathymetry = pd.DataFrame.from_dict(bathymetry)
                volumes = bathymetry['volume'].astype(np.float64)
                levels = bathymetry['level'].astype(np.float64)
                areas = bathymetry['area'].astype(np.float64)

        if volumes is not None and levels is not None:
            node.level = InterpolatedVolumeParameter(model, node, volumes, levels)

        if volumes is not None and areas is not None:
            node.area = InterpolatedVolumeParameter(model, node, volumes, areas)
        if node.weather is not None:
            node._make_weather_nodes(model, node.weather, node.weather_cost)
        setattr(node, "_Loadable__parameters_to_load", {})
        return node


    def _make_weather_nodes(self, model, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        weather = pd.DataFrame.from_dict(weather)

        rainfall = weather['rainfall'].astype(np.float64)
        evaporation = weather['evaporation'].astype(np.float64)

        self._make_evaporation_node(model, evaporation, cost)
        self._make_rainfall_node(model, rainfall, cost)


    def _make_evaporation_node(self, model, evaporation, cost):

        if not isinstance(self.area, Parameter):
            LOG.warning('Evaporation nodes be created only if an area Parameter is given.')
            return

        if evaporation is None:
            try:
                evaporation_param = load_parameter(model, f'__{self.name}__:evaporation')
            except KeyError:
                LOG.warning(f"Please specify evaporation or weather on node {self.name}")
                return
        elif isinstance(evaporation, pd.DataFrame) or isinstance(evaporation, pd.Series):
            evaporation = evaporation.astype(np.float64)
            evaporation_param = MonthlyProfileParameter(model, evaporation)
        else:
            evaporation_param = evaporation

        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, self.const, self.area],
                                                     agg_func='product')

        evaporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evaporation_node.max_flow = evaporation_flow_param
        evaporation_node.cost = cost

        self.connect(evaporation_node)
        self.evaporation_node = evaporation_node

        self.evaporation_recorder = NumpyArrayNodeRecorder(model, evaporation_node,
                                                           name=f'__{evaporation_node.name}__:evaporation')

    def _make_rainfall_node(self, model, rainfall, cost):

        if not isinstance(self.area, Parameter):
            LOG.warning('Weather nodes can be created only if an area Parameter is given.')
            return

        if rainfall is None:
            try:
                rainfall_param = load_parameter(model, f'__{self.name}__:rainfall')
            except KeyError:
                LOG.warning(f"Please specify rainfall or weather on node {self.name}")
                return
        elif isinstance(rainfall, pd.DataFrame) or isinstance(rainfall, pd.Series):
            rainfall = rainfall.astype(np.float64)
            rainfall_param = MonthlyProfileParameter(model, rainfall)
        else:
            rainfall_param = rainfall

        # Create the flow parameters multiplying area by rate of rainfall/evap
        rainfall_flow_param = AggregatedParameter(model, [rainfall_param, self.const, self.area],
                                                  agg_func='product')

        # Create the nodes to provide the flows
        rainfall_node = Input(model, '{}.rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param
        rainfall_node.cost = cost

        rainfall_node.connect(self)
        self.rainfall_node = rainfall_node

        self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node,
                                                        name=f'__{rainfall_node.name}__:rainfall')

class River(RiverDomainMixin, Link):
    """A node in the river network

    This node may have multiple upstream nodes (i.e. a confluence) but only
    one downstream node.
    """

    def __init__(self, *args, **kwargs):
        super(River, self).__init__(*args, **kwargs)


class RiverSplit(MultiSplitLink):
    """A split in the river network

    RiverSplit is a specialised version of `pywr.nodes.MultiSplitLink` with a more convenient init method.
    It is intended for a simple case of where fixed ratio of flow is required to be distributed
    to multiple downstream routes.

    Parameters
    ----------
    factors : iterable of floats
        The factors to force on the additional splits. Number of extra_slot is assumed to be one less
        than the length of factors (as per `pywr.nodes.MultiSplitLink` documentation).
    slot_names : iterable
        The identifiers to refer to the slots when connect from this Node. Length must be one more than
         the number of extra slots required.

    See also
    --------
    pywr.nodes.MultiSplitLink

    """

    def __init__(self, model, *args, nsteps=1, **kwargs):
        factors = kwargs.pop("factors")
        extra_slots = len(factors) - 1

        # These are the defaults to pass to the parent class that makes this
        # class a convenience.
        # create keyword arguments for PiecewiseLink
        costs = kwargs.pop("costs", [0.0])
        max_flows = kwargs.pop("max_flows", [None])

        super(RiverSplit, self).__init__(
            model,
            nsteps,
            *args,
            extra_slots=extra_slots,
            factors=factors,
            costs=costs,
            max_flows=max_flows,
            **kwargs,
        )


class RiverSplitWithGauge(RiverSplit):
    """A split in the river network with a minimum residual flow.

    As per `RiverSplit` but by default creates another route in the underlying object
    to model an MRF. This route is such that the MRF is not part of forced ratios. The
    intent of this object is to model the case where a proportion of flow can be
    abstracted above the MRF (e.g. 90% of flow above MRF).

    ::

                 /  -->-- X0 {max_flow: mrf,  cost: mrf_cost} -->-- \\
        A -->-- Xo  -->-- X1 {max_flow: None, cost: cost}     -->-- Xi -->-- C
                 \\  -->-- X2 {max_flow: None, cost: 0.0}      -->-- /
                           |
                           Bo -->-- Bi --> D

        Ag {nodes: [X1, X2], factors: factors}

    Parameters
    ----------
    mrf : float
        The minimum residual flow (MRF) at the gauge
    mrf_cost : float
        The cost of the route via the MRF
    cost : float
        The cost of the other (unconstrained) route
    factors : iterable of floats
        The factors to force on the additional splits. Number of extra_slot is assumed to be one less
        than the length of factors (as per `MultiSplitLink` documentation).
    slot_names : iterable
        The identifiers to refer to the slots when connect from this Node. Length must be one more than
         the number of extra slots required.
    """

    __parameter_attributes__ = ("cost", "mrf_cost", "mrf")

    def __init__(self, model, *args, mrf=0.0, cost=0.0, mrf_cost=0.0, **kwargs):
        super(RiverSplitWithGauge, self).__init__(
            model,
            *args,
            nsteps=2,
            max_flows=[mrf, None],
            costs=[mrf_cost, cost],
            **kwargs,
        )

    def mrf():
        def fget(self):
            return self.sublinks[0].max_flow

        def fset(self, value):
            self.sublinks[0].max_flow = value

        return locals()

    mrf = property(**mrf())

    def mrf_cost():
        def fget(self):
            return self.sublinks[0].cost

        def fset(self, value):
            self.sublinks[0].cost = value

        return locals()

    mrf_cost = property(**mrf_cost())

    def cost():
        def fget(self):
            return self.sublinks[1].cost

        def fset(self, value):
            self.sublinks[1].cost = value

        return locals()

    cost = property(**cost())


class Discharge(Catchment):
    """An inline discharge to the river network

    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """

    pass


class RiverGauge(RiverDomainMixin, PiecewiseLink):
    """A river gauging station, with a minimum residual flow (MRF)"""

    __parameter_attributes__ = ("cost", "mrf_cost", "mrf")

    def __init__(self, *args, **kwargs):
        """Initialise a new RiverGauge instance

        Parameters
        ----------
        mrf : float
            The minimum residual flow (MRF) at the gauge
        mrf_cost : float
            The cost of the route via the MRF
        cost : float
            The cost of the other (unconstrained) route
        """
        # create keyword arguments for PiecewiseLink
        cost = kwargs.pop("cost", 0.0)
        kwargs["costs"] = [kwargs.pop("mrf_cost", 0.0), cost]
        kwargs["max_flows"] = [kwargs.pop("mrf", 0.0), None]
        super(RiverGauge, self).__init__(nsteps=2, *args, **kwargs)

    def mrf():
        def fget(self):
            return self.sublinks[0].max_flow

        def fset(self, value):
            self.sublinks[0].max_flow = value

        return locals()

    mrf = property(**mrf())

    def mrf_cost():
        def fget(self):
            return self.sublinks[0].cost

        def fset(self, value):
            self.sublinks[0].cost = value

        return locals()

    mrf_cost = property(**mrf_cost())

    def cost():
        def fget(self):
            return self.sublinks[1].cost

        def fset(self, value):
            self.sublinks[1].cost = value

        return locals()

    cost = property(**cost())

class ProportionalInput(Input, metaclass=NodeMeta):
    """
    TODO: Jose explain this
    """
    min_proportion = 1e-6

    def __init__(self, model, name, node, proportion, **kwargs):
        super().__init__(model, name, **kwargs)

        self.node = model.pre_load_node(node)

        # Create the flow factors for the other node and self
        if proportion < self.__class__.min_proportion:
            self.max_flow = 0.0
        else:
            factors = [1, proportion]
            # Create the aggregated node to apply the factors.
            self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [self.node, self])
            self.aggregated_node.factors = factors



class LinearStorageReleaseControl(Link, metaclass=NodeMeta):
    """
        A specialised node that provides a default max_flow based on a release rule.
        TODO: Jose explain this in more detail. WHat does it do, so someone reading the docs can understand it.
    """

    def __init__(self, model, name, storage_node, release_values, scenario=None, **kwargs):

        release_values = pd.DataFrame.from_dict(release_values)
        storage_node = model.pre_load_node(storage_node)

        if scenario is None:
            # Only one control curve should be defined. Get it explicitly
            control_curves = release_values['volume'].iloc[1:-1].astype(np.float64)
            values = release_values['value'].astype(np.float64)
            max_flow_param = ControlCurveInterpolatedParameter(model, storage_node, control_curves, values)
        else:
            # There should be multiple control curves defined.
            if release_values.shape[1] % 2 != 0:
                raise ValueError("An even number of columns (i.e. pairs) is required for the release rules "
                                 "when using a scenario.")

            ncurves = release_values.shape[1] // 2
            if ncurves != scenario.size:
                raise ValueError(f"The number of curves ({ncurves}) should equal the size of the "
                                 f"scenario ({scenario.size}).")

            curves = []
            for i in range(ncurves):
                volume = release_values.iloc[1:-1, i*2]
                values = release_values.iloc[:, i*2+1]
                control_curve = ControlCurveInterpolatedParameter(model, storage_node, volume, values)
                curves.append(control_curve)

            max_flow_param = ScenarioWrapperParameter(model, scenario, curves)

        self.max_flow = max_flow_param
        self.scenario = scenario
        super().__init__(model, name, max_flow=max_flow_param, **kwargs)

    @classmethod
    def pre_load(cls, model, data):
        name = data.pop("name")
        cost = data.pop("cost", 0.0)
        min_flow = data.pop("min_flow", None)

        node = cls(name=name, model=model, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0

        node.cost = cost
        node.min_flow = min_flow

        """
            The Pywr Loadable base class contains a reference to
            `self.__parameters_to_load.items()` which will fail unless
            a pre-mangled name which matches the expected value from
            inside the Loadable class is added here.

            See pywr/nodes.py:80 Loadable.finalise_load()
        """
        setattr(node, "_Loadable__parameters_to_load", {})
        return node

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
    DataFrameParameter,
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
    these nodes internally when the evaporation, rainfall, and area properties are set.

    Parameters
    ----------
    model : Model
        Model instance to which this storage node is attached.
    name : str
        The name of the storage node.
    min_volume : float (optional)
        The minimum volume of the storage. Defaults to 0.0.
    max_volume : float, Parameter (optional)
        The maximum volume of the storage. Defaults to 0.0.
    initial_volume, initial_volume_pc : float (optional)
        Specify initial volume in either absolute or proportional terms. Both are required if `max_volume`
        is a parameter because the parameter will not be evaluated at the first time-step. If both are given
        and `max_volume` is not a Parameter, then the absolute value is ignored.
    evaporation :   DataFrame, Parameter (optional)
        Normally a DataFrame with a index and a single column of 12 evaporation rates, representing each month in a year.
    evaporation_cost : float (optional)
        The cost of evaporation. Defaults to -999.
    unit_conversion : float (optional)
        The unit conversion factor for evaporation. Defaults to 1e6 * 1e-3 * 1e-6. This assumes area is Km2, level is m and evaporation is mm/day.
    rainfall : DataFrame, Parameter (optional)
        Normally a DataFrame with a index and a single column of 12 rainfall rates, representing each month in a year.
    area, level : float, Parameter (optional)
        Optional float or Parameter defining the area and level of the storage node. These values are
        accessible through the `get_area` and `get_level` methods respectively.
    """

    def __init__(self, model, *args, **kwargs):
        """

        Keywords:
            control_curve - A Parameter object that can return the control curve position,
                as a percentage of fill, for the given timestep.
        """

        __parameter_attributes__ = ("min_volume", "max_volume", "level", "area")

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

        # self.level = pop_kwarg_parameter(kwargs, "level", None)
        # self.area = pop_kwarg_parameter(kwargs, "area", None)

        self.evaporation_cost = kwargs.pop('evaporation_cost', -999)
        self.unit_conversion = kwargs.pop('unit_conversion', 1e6 * 1e-3 * 1e-6) #This assume area is Km2, level is m and evaporation is mm/day

        self.evaporation = kwargs.pop("evaporation", None)
        self.rainfall = kwargs.pop("rainfall", None)

        name = kwargs.pop('name')
        super().__init__(model, name, **kwargs)

        self.rainfall_node = None
        self.rainfall_recorder = None
        self.evaporation_node = None
        self.evaporation_recorder = None


    def finalise_load(self):
        super(Reservoir, self).finalise_load()

        #in some cases, this hasn't been converted to a constant parameter, such as in the unit tests, so
        #check for that here.
        if not isinstance(self.unit_conversion, Parameter):
            self.unit_conversion = ConstantParameter(self.model, self.unit_conversion)

        if self.evaporation is not None:
            self._make_evaporation_node(self.evaporation, self.evaporation_cost)

        if self.rainfall is not None:
            self._make_rainfall_node(self.rainfall)

    def _make_evaporation_node(self, evaporation, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Evaporation nodes can only be created if an area Parameter is given.')

        if isinstance(evaporation, Parameter):
            evaporation_param = evaporation
        elif isinstance(evaporation, str):
            evaporation_param = load_parameter(self.model, evaporation)
        elif isinstance(evaporation, (int, float)):
            evaporation_param = ConstantParameter(self.model, evaporation)
        else:
            evp = pd.DataFrame.from_dict(evaporation)
            evaporation_param = DataFrameParameter(self.model, evp)

        evaporation_flow_param = AggregatedParameter(self.model, [evaporation_param, self.unit_conversion, self.area],
                                                     agg_func='product')

        evaporation_node = Output(self.model, '{}.evaporation'.format(self.name), parent=self)
        evaporation_node.max_flow = evaporation_flow_param
        evaporation_node.cost = cost

        self.connect(evaporation_node)
        self.evaporation_node = evaporation_node

        self.evaporation_recorder = NumpyArrayNodeRecorder(self.model, evaporation_node,
                                                           name=f'__{evaporation_node.name}__:evaporation')

    def _make_rainfall_node(self, rainfall):
        if isinstance(rainfall, Parameter):
            rainfall_param = rainfall  
        elif isinstance(rainfall, str):
            rainfall_param = load_parameter(self.model, rainfall)
        elif isinstance(rainfall, (int, float)):
            rainfall_param = ConstantParameter(self.model, rainfall)
        else:
            #it's not a paramter or parameter reference, to try float and dataframe
            rain = pd.DataFrame.from_dict(rainfall)
            rainfall_param = DataFrameParameter(self.model, rain)

        # Create the flow parameters multiplying area by rate of rainfall/evap

        rainfall_flow_param = AggregatedParameter(self.model, [rainfall_param, self.unit_conversion, self.area],
                                                  agg_func='product')

        # Create the nodes to provide the flows
        rainfall_node = Catchment(self.model, '{}_rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param


        rainfall_node.connect(self)
        self.rainfall_node = rainfall_node
        self.rainfall_recorder = NumpyArrayNodeRecorder(self.model, rainfall_node,
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
    This node is an input node that has a max_flow that is a proportion of another node.
    An example of this is a return flow from an irrigation node. 
    The return flow is the proportion of the irrigation node flow.
    
    Parameters
    ----------
    model : Model
        The model object
    name : str
        The name of the node
    node : Node
        The node to which the input is proportional
    proportion : float
        The proportion of the node's flow to use as the max_flow
    
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

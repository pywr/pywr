from pywr.nodes import Domain, Input, Link, Storage, PiecewiseLink, MultiSplitLink
from pywr.parameters import (
    pop_kwarg_parameter,
    ConstantParameter,
    Parameter,
    load_parameter,
)
from pywr.parameters.control_curves import ControlCurveParameter

DEFAULT_RIVER_DOMAIN = Domain(name="river", color="#33CCFF")


class RiverDomainMixin(object):
    """Class to identify a river domain."""

    def __init__(self, *args, **kwargs):
        # if 'domain' not in kwargs:
        #     kwargs['domain'] = DEFAULT_RIVER_DOMAIN
        if "color" not in kwargs:
            self.color = "#6ECFF6"  # blue
        super(RiverDomainMixin, self).__init__(*args, **kwargs)


class Catchment(RiverDomainMixin, Input):
    """A hydrological catchment, supplying water to the river network.

    A Catchment is an `Input` node with a fixed inflow. I.e. `min_flow` and
    `max_flow` are the same. The value is specified as a flow keyword, and
    overrides any `min_flow` or `max_flow` keyword arguments.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    Catchment(model=model, flow=1.0, name="Inflow")
    ```

    JSON
    ======
    ```json
    {
        "name": "Inflow",
        "type": "Catchment",
        "flow": 1.0,
    }
    ```
    """

    __parameter_attributes__ = ("cost", "flow")

    def __init__(self, *args, **kwargs):
        """Initialise a new Catchment node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        flow : Optional[float | Parameter], default=0
            The amount of water supplied by the catchment each timestep.
        """
        self.color = "#82CA9D"  # green
        # Grab flow from kwargs
        flow = kwargs.pop("flow", 0.0)
        # Min/max flow set in super inits
        super(Catchment, self).__init__(*args, **kwargs)
        self.flow = flow

    def get_flow(self, scenario_index):
        """Get the flow.

        Parameters
        ---------
        scenario_index : ScenarioIndex
            The scenario index to get the min flow of.

        Returns
        -------
        float
            The flow.
        """
        return self.get_min_flow(scenario_index)

    def __setattr__(self, name, value):
        if name == "flow":
            self.min_flow = value
            self.max_flow = value
            return
        super(Catchment, self).__setattr__(name, value)


class Reservoir(RiverDomainMixin, Storage):
    """A [pywr.nodes.Storage][] with one control curve.

    The Reservoir is a subclass of [pywr.nodes.Storage][] with additional functionality to provide a
    simple control curve to control the cost. The Reservoir has above_curve_cost when it is above its curve
    and the user defined cost when it is below. Typically, the costs are negative
    to represent a benefit of filling the reservoir when it is below its curve.

    Examples
    --------
    Python
    ==================
    ```python
    model = Model()
    Reservoir(
        model=model,
        control_curve=0.4,
        above_curve_cost=-0.1,
        cost=-100,
        name="Reservoir"
    )
    ```

    JSON
    ==================
    ```json
    {
        "name": "Reservoir",
        "type": "Reservoir",
        "control_curve": 0.4,
        "above_curve_cost": -0.1,
        "cost": -100
    }
    ```
    """

    def __init__(self, model, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.


        Other Parameters
        ----------------
        name : string
            A unique name for the node.
        control_curve : Optional[float | Parameter], default=None
            This can be a number (to assume a flat control curve) or a Parameter that returns the control curve position,
            as relative volume of fill for the given timestep. When `None`, the control curve
            defaults to 1 (full reservoir).
        above_curve_cost : Optional[float], default=None
            The cost when the reservoir is above the control curve. When `None`, the reservoir cost
            defaults to `cost` and the control curve is ignored.
        cost: Optional[float], default=0.0
            The cost when the reservoir is below the control curve.
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
        super(Reservoir, self).__init__(model, *args, **kwargs)


class River(RiverDomainMixin, Link):
    """A node in the river network.

    This node may have multiple upstream nodes (i.e. a confluence), but only
    one downstream node.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    River(model=model, min_flow=1.0, name="Thames")
    ```

    JSON
    ======
    ```json
    {
        "name": "Thames",
        "type": "River",
        "min_flow": 1.0,
    }
    ```

    Notes
    -----
    This node is an extension of the [pywr.core.Link][] node.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Default to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        super(River, self).__init__(*args, **kwargs)


class RiverSplit(MultiSplitLink):
    """A split in the river network.

    RiverSplit is a specialised version of [pywr.nodes.MultiSplitLink][] with a
    more convenient `init` method.
    It is intended for a simple case of where fixed ratio of flow is required to be distributed
    to multiple downstream routes.

    See also
    --------
    [pywr.nodes.MultiSplitLink][]

    """

    def __init__(self, model, *args, nsteps=1, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        nsteps : int
            The number of steps to split.

        Other Parameters
        ----------------
        name : str
            The unique name of the node.
        factors : Iterable[float]
            The factors to force on the additional splits. The number of extra_slot is assumed to be one less
            than the length of factors (as per [pywr.nodes.MultiSplitLink][] documentation).
        slot_names : Iterable
            The identifiers to refer to the slots when connect from this Node. Length must be one more than
             the number of extra slots required.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.

        """
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

    As per [pywr.domains.river.RiverSplit][] but by default creates another route in the underlying object
    to model an MRF. This route is such that the MRF is not part of forced ratios. The
    intent of this object is to model the case where a proportion of flow can be
    abstracted above the MRF (e.g. 90% of flow above MRF).

    ```mermaid
    graph LR
        A --> Xo
        Xo --> D{X0 <br/>max_flow: mrf, cost: mrf_cost}
        D --> Xi
        Xo --> E{X1 <br/> max_flow: None, cost: cost}
        E --> Xi
        Xo --> F{X2 <br/>max_flow: None, cost: 0.0}
        F --> Xi
        Xi --> C
        F --> Bo
        Bo --> Bi
        Bi --> H

        L{Ag <br/>nodes: X1, X2, factors: factors}
    ```

    Attributes
    ----------
    mrf : float
        The minimum residual flow (MRF) at the gauge.

        **Setter:** set the minimum residual flow.
    mrf_cost : float
        The cost of the route via the MRF.

        **Setter:** set the minimum residual flow cost.
    mrf_cost : float
         The cost of the other (unconstrained) route.

        **Setter:** set the other route cost.
    """

    __parameter_attributes__ = ("cost", "mrf_cost", "mrf")

    def __init__(self, model, *args, mrf=0.0, cost=0.0, mrf_cost=0.0, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.

        Other Parameters
        ----------------
        name : str
            The unique name of the node.
        mrf : Optional[float], default = 0
            The minimum residual flow (MRF) at the gauge.
        mrf_cost : Optional[float], default = 0
            The cost of the route via the MRF.
        cost : Optional[float], default = 0
            The cost of the other (unconstrained) route.
        factors : Iterable[float]
            The factors to force on the additional splits. Number of extra_slot is assumed to be one less
            than the length of factors (as per `MultiSplitLink` documentation).
        slot_names : Iterable
            The identifiers to refer to the slots when connect from this Node. Length must be one more than
             the number of extra slots required.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        """
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
    """An inline discharge to the river network.

    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """

    pass


class RiverGauge(RiverDomainMixin, PiecewiseLink):
    """A river gauging station, with a minimum residual flow (MRF). This
    extends a [pywr.nodes.PiecewiseLink][] node.

    Attributes
    ----------
    mrf : float
        The minimum residual flow (MRF) at the gauge.

        **Setter:** set the minimum residual flow.
    mrf_cost : float
        The cost of the route via the MRF.

        **Setter:** set the minimum residual flow cost.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    RiverGauge(model=model, mrf=1.0, mrf_cost=-100, name="Compensation")
    ```

    JSON
    ======
    ```json
    {
        "name": "Compensation",
        "type": "RiverGauge",
        "mrf": 1.0,
        "mrf_cost": -100
    }
    ```
    """

    __parameter_attributes__ = ("cost", "mrf_cost", "mrf")

    def __init__(self, *args, **kwargs):
        """Initialise a new RiverGauge instance.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        mrf : Optional[float], default=0
            The minimum residual flow (MRF) at the gauge.
        mrf_cost : Optional[float], default=0
            The cost of the route via the MRF.
        cost : Optional[float], default=0
            The cost of the other (unconstrained) route.
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

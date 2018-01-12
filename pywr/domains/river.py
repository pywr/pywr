
from pywr.nodes import Node, Domain, Input, Output, Link, Storage, PiecewiseLink, MultiSplitLink
from pywr.parameters import pop_kwarg_parameter, ConstantParameter, Parameter, load_parameter
from pywr.parameters.control_curves import ControlCurveParameter

DEFAULT_RIVER_DOMAIN = Domain(name='river', color='#33CCFF')

class RiverDomainMixin(object):
    def __init__(self, *args, **kwargs):
        # if 'domain' not in kwargs:
        #     kwargs['domain'] = DEFAULT_RIVER_DOMAIN
        if 'color' not in kwargs:
            self.color = '#6ECFF6' # blue
        super(RiverDomainMixin, self).__init__(*args, **kwargs)


class Catchment(RiverDomainMixin, Input):
    """A hydrological catchment, supplying water to the river network"""
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
        self.color = '#82CA9D' # green
        # Grab flow from kwargs
        flow = kwargs.pop('flow', 0.0)
        # Min/max flow set in super inits
        super(Catchment, self).__init__(*args, **kwargs)
        self.flow = flow

    def get_flow(self, timestep):
        """ flow is ensured that both min_flow and max_flow are the same. """
        return self.get_min_flow(timestep)

    def __setattr__(self, name, value):
        if name == 'flow':
            self.min_flow = value
            self.max_flow = value
            return
        super(Catchment, self).__setattr__(name, value)

    @classmethod
    def load(cls, data, model):
        flow = data.pop('flow', 0.0)
        if flow is not None:
            flow = load_parameter(model, flow)
        node = super(Catchment, cls).load(data, model)
        node.flow = flow
        return node


class Reservoir(RiverDomainMixin, Storage):
    """A reservoir node with control curve.

    The Reservoir is a subclass of Storage with additional functionality to provide a
    simple control curve. The Reservoir has above_curve_cost when it is above its curve
    and the user defined cost when it is below. Typically the costs are negative
    to represent a benefit of filling the reservoir when it is below its curve.
    """
    def __init__(self, model, *args, **kwargs):
        """

        Keywords:
            control_curve - A Parameter object that can return the control curve position,
                as a percentage of fill, for the given timestep.
        """
        control_curve = pop_kwarg_parameter(kwargs, 'control_curve', None)
        above_curve_cost = kwargs.pop('above_curve_cost', None)
        cost = kwargs.pop('cost', 0.0)
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
                kwargs['cost'] = ControlCurveParameter(model, self, control_curve, [above_curve_cost, cost])
            else:
                raise ValueError('If an above_curve_cost is given cost must not be a Parameter.')
        else:
            # reinstate the given cost parameter to pass to the parent constructors
            kwargs['cost'] = cost
        super(Reservoir, self).__init__(model, *args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        def _make_iterable(val):
            try:
                len(val)
            except TypeError:
                return [val]
            return val

        factors = _make_iterable(kwargs.pop('factors'))
        extra_slots = len(factors) - 1

        # These are the defaults to pass to the parent class that makes this
        # class a convenience.
        # create keyword arguments for PiecewiseLink
        kwargs['cost'] = _make_iterable(kwargs.pop('cost', 0.0))
        kwargs['max_flow'] = _make_iterable(kwargs.pop('max_flow', None))
        kwargs['extra_slots'] = extra_slots
        kwargs['factors'] = factors

        super(RiverSplit, self).__init__(*args, **kwargs)


class RiverSplitWithGauge(RiverSplit):
    """A split in the river network with a minimum residual flow

    As per `RiverSplit` but by default creates another route in the underlying object
     to model a MRF. This route is such that the MRF is not part of forced ratios. The
     intent of this object is to model the case where a proportion of flow can be
     abstracted above the MRF (e.g. 90% of flow above MRF).

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
    def __init__(self, model, name, mrf=0.0, cost=0.0, mrf_cost=0.0, **kwargs):
        kwargs['cost'] = [mrf_cost, cost]
        kwargs['max_flow'] = [mrf, None]
        super(RiverSplitWithGauge, self).__init__(model, name, **kwargs)

    @classmethod
    def load(cls, data, model):
        cost = load_parameter(model, data.pop('cost', 0.0))
        mrf_cost = load_parameter(model, data.pop('mrf_cost', 0.0))
        mrf = load_parameter(model, data.pop('mrf', 0.0))
        name = data.pop("name")
        data.pop("type", None)
        parameter = cls(model, name, mrf=mrf, cost=cost, mrf_cost=mrf_cost, **data)
        return parameter

class Discharge(Catchment):
    """An inline discharge to the river network

    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """
    pass

class RiverGauge(RiverDomainMixin, PiecewiseLink):
    """A river gauging station, with a minimum residual flow (MRF)
    """
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
        cost = kwargs.pop('cost', 0.0)
        kwargs['cost'] = [kwargs.pop('mrf_cost', 0.0), cost]
        kwargs['max_flow'] = [kwargs.pop('mrf', 0.0), None]
        super(RiverGauge, self).__init__(*args, **kwargs)

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

    @classmethod
    def load(cls, data, model):
        mrf = load_parameter(model, data.pop("mrf"))
        mrf_cost = load_parameter(model, data.pop("mrf_cost"))
        cost = load_parameter(model, data.pop("cost", 0.0))
        del(data["type"])
        node = cls(model, mrf=mrf, mrf_cost=mrf_cost, cost=cost, **data)
        return node

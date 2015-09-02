
from ..core import Node, Domain, Input, Output, Link, Storage, PiecewiseLink, ParameterFunction, pop_kwarg_parameter

DEFAULT_RIVER_DOMAIN = Domain(name='river', color='#33CCFF')

class RiverDomainMixin(object):
    def __init__(self, *args, **kwargs):
        if 'domain' not in kwargs:
            kwargs['domain'] = DEFAULT_RIVER_DOMAIN
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


    def check(self):
        super(Catchment, self).check()
        successors = self.model.graph.successors(self)
        if not len(successors) == 1:
            raise ValueError('{} has invalid number of successors ({})'.format(self, len(successors)))

    def get_flow(self, timestep):
        """ flow is ensured that both min_flow and max_flow are the same. """
        return self.get_min_flow(timestep)

    def __setattr__(self, name, value):
        if name == 'flow':
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
    """
    def __init__(self, *args, **kwargs):
        """

        Keywords:
            control_curve - A Parameter object that can return the control curve position,
                as a percentage of fill, for the given timestep.
            above_curve_cost - The cost to apply when the reservoir is above its curve.
        """
        self.control_curve = kwargs.pop('control_curve', None)
        self.above_curve_cost = kwargs.pop('above_curve_cost', 0.0)
        super(Reservoir, self).__init__(*args, **kwargs)

    def get_cost(self, ts):
        # If no control curve given behaves like a normal Storage
        if self.control_curve is None:
            return super(Reservoir, self).get_cost(ts)

        control_curve = self.control_curve.value(ts)/100.0
        # If level above control curve then return above_curve_cost
        if self.current_pc >= control_curve:
            return self.above_curve_cost
        return super(Reservoir, self).get_cost(ts)


class River(RiverDomainMixin, Link):
    """A node in the river network

    This node may have multiple upstream nodes (i.e. a confluence) but only
    one downstream node.
    """
    def __init__(self, *args, **kwargs):
        super(River, self).__init__(*args, **kwargs)


class RiverSplit(River):
    """A split in the river network"""
    def __init__(self, *args, **kwargs):
        """Initialise a new RiverSplit instance

        Parameters
        ----------
        split : float or function
            The ratio to apportion the flow between the two downstream nodes as
            a ratio (0.0-1.0). If no value is given a default value of 0.5 is
            used.
        """
        River.__init__(self, *args, **kwargs)
        # These are the upstream slots
        self._slots = {1: None, 2: None}

        self.split = pop_kwarg_parameter(kwargs, 'split', 0.5)

    def iter_slots(self, slot_name=None, is_connector=True):
        # All sublinks are connected upstream and downstream
        if not is_connector:
            yield self._slots[slot_name]
        for link in self.sublinks:
            yield link

class Discharge(Catchment):
    """An inline discharge to the river network

    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """
    pass


class DemandDischarge(River):
    """River discharge for demands that aren't entirely consumptive
    """
    pass


class Terminator(RiverDomainMixin, Output):
    """A sink in the river network

    This node is required to close the network and is used by some of the
    routing algorithms. Every river must end in a Terminator.
    """
    def __init__(self, *args, **kwargs):
        super(Terminator, self).__init__(*args, **kwargs)


class RiverGauge(RiverDomainMixin, PiecewiseLink):
    """A river gauging station, with a minimum residual flow (MRF)
    """
    def __init__(self, *args, **kwargs):
        """Initialise a new RiverGauge instance

        Parameters
        ----------
        mrf : float
            The minimum residual flow (MRF) at the gauge
        """
        # create keyword arguments for PiecewiseLink
        cost = kwargs.pop('cost', 0.0)
        kwargs['cost'] = [kwargs.pop('mrf_cost', 0.0), cost]
        kwargs['max_flow'] = [kwargs.pop('mrf'), None]
        super(RiverGauge, self).__init__(*args, **kwargs)


class RiverAbstraction(RiverDomainMixin, Output):
    """An abstraction from the river network"""
    def __init__(self, *args, **kwargs):
        super(RiverAbstraction, self).__init__(*args, **kwargs)


class DemandCentre(RiverDomainMixin, Output):
    """A demand centre"""
    def __init__(self, *args, **kwargs):
        super(DemandCentre, self).__init__(*args, **kwargs)

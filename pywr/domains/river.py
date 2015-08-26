
from ..core import Node, Domain, Input, Output, Link, Storage, PiecewiseLink, ParameterFunction


class RiverDomainMixin(object):
    def __init__(self, *args, **kwargs):
        if 'domain' not in kwargs:
            kwargs['domain'] = Domain(name='river', color='#33CCFF')
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

        # Min/max flow set in super inits
        super(Catchment, self).__init__(*args, **kwargs)
        # Grab flow from kwargs
        self.flow = kwargs.pop('flow', 0.0)

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
    def __init__(self, *args, **kwargs):
        super(Reservoir, self).__init__(*args, **kwargs)


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
        self.slots = {1: None, 2: None}

        self.properties['split'] = self.pop_kwarg_parameter(kwargs, 'split', 0.5)

class Discharge(River):
    """An inline discharge to the river network

    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """
    def __init__(self, *args, **kwargs):
        River.__init__(self, *args, **kwargs)

        self.properties['flow'] = self.pop_kwarg_parameter(kwargs, 'flow', 0.0)


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
        kwargs['cost'] = [cost-kwargs.pop('benefit', 0.0), cost]
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

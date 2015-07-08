
from ..core import Node, Input, Output, Link, Storage, ParameterFunction


class RiverDomainMixin(object):
    def __init__(self, *args, **kwargs):
        if 'domain' not in kwargs:
            kwargs['domain'] = 'river'
        super(RiverDomainMixin, self).__init__(*args, **kwargs)


class Catchment(Input, RiverDomainMixin):
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
        super(Catchment, self).__init__(*args, **kwargs)
        self.color = '#82CA9D' # green

        self.properties['flow'] = self.pop_kwarg_parameter(kwargs, 'flow', 0.0)

        def func(parent, index):
            return self.properties['flow'].value(index)
        self.properties['min_flow'] = ParameterFunction(self, func)
        self.properties['max_flow'] = ParameterFunction(self, func)

    def check(self):
        super(Catchment, self).check()
        successors = self.model.graph.successors(self)
        if not len(successors) == 1:
            raise ValueError('{} has invalid number of successors ({})'.format(self, len(successors)))


class Reservoir(Storage, RiverDomainMixin):
    def __init__(self, *args, **kwargs):
        super(Reservoir, self).__init__(*args, **kwargs)


class River(Link, RiverDomainMixin):
    """A node in the river network

    This node may have multiple upstream nodes (i.e. a confluence) but only
    one downstream node.
    """
    def __init__(self, *args, **kwargs):
        super(River, self).__init__(*args, **kwargs)
        self.color = '#6ECFF6' # blue


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


class Terminator(Output, RiverDomainMixin):
    """A sink in the river network

    This node is required to close the network and is used by some of the
    routing algorithms. Every river must end in a Terminator.
    """
    def __init__(self, *args, **kwargs):
        super(Terminator, self).__init__(*args, **kwargs)


class RiverGauge(River):
    """A river gauging station, with a minimum residual flow (MRF)
    """
    def __init__(self, *args, **kwargs):
        """Initialise a new RiverGauge instance

        Parameters
        ----------
        mrf : float (optional)
            The minimum residual flow (MRF) at the gauge
        """
        River.__init__(self, *args, **kwargs)

        self.properties['mrf'] = self.pop_kwarg_parameter(kwargs, 'mrf', None)


class RiverAbstraction(Output, RiverDomainMixin):
    """An abstraction from the river network"""
    def __init__(self, *args, **kwargs):
        super(RiverAbstraction, self).__init__(*args, **kwargs)

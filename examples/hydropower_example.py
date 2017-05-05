from pywr.core import Model
from pywr.recorders import Recorder
from pywr.recorders._recorders import NodeRecorder

import numpy as np


class HydroPowerRecorder(NodeRecorder):
    """ Calculates the total energy produced using the hydropower equation
    
    Power (W) = density (kg/m3) * flow (m3/s) * g (m/s2) * head (m)
    
    Total energy (J) = Power (J/s) * time-step (s)
        
    Parameters
    ----------
    
    storage_node : Storage instance (default=None)
        The optional Storage instance from which to take the level in order to calculate head. If a Storage node is 
        given it must have a valid level parameter.
    level : float (default=None)
        The optional level to used to calculate head.
    efficiency : float (default=1.0)
        The efficiency of the turbine. 
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of $m^3/s$
        
    Notes
    -----
    
    Head is calucated from the given Storage node and/or level as follows:
    
     - If Storage is given, but level is not. Head is simply the level in the Storage node.
     - If Storage and level are both given. Head is difference between level in the Storage node and the level given here.
     - If only level is given then this is simply used as a constat head.
      

    """
    def __init__(self, model, node, storage_node=None, level=None, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, **kwargs):
        super(HydroPowerRecorder, self).__init__(model, node, **kwargs)

        self.storage_node = storage_node
        self.level = level
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion

        if storage_node is None and level is None:
            raise ValueError('One or both of storage_node or level must be given.')

    def setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def reset(self):
        self._values[...] = 0.0

    def values(self):
        return self._values

    def after(self):

        ts = self.model.timestepper.current
        # Timestep in seconds
        delta = self.model.timestepper.delta.days
        delta *= 24*3600

        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self.storage_node is not None:
                head = self.storage_node.get_level(ts, scenario_index)
                if self.level is not None:
                    head -= self.level
            elif self.level is not None:
                head = self.level
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)

            # Convert flow to correct units
            q = flow[scenario_index.global_id] * self.flow_unit_conversion

            # Power
            power = self.density * q * 9.81 * head
            print(q, power)
            # Accumulate total energy
            self._values[scenario_index.global_id] += power * delta

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        if "storage_node" in data:
            storage_node = model._get_node_from_ref(model, data.pop("storage_node"))
        else:
            storage_node = None

        return cls(model, node, storage_node=storage_node, **data)
HydroPowerRecorder.register()


if __name__ == '__main__':

    m = Model.load('hydropower_example.json')
    stats = m.run()
    print(stats)

    print(m.recorders["turbine1_energy"].values())


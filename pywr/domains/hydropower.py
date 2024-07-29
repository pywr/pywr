from pywr.nodes import (
    NodeMeta,
    Link
)

from pywr.parameters import (
    Parameter,
    load_parameter,
    ConstantParameter,
    HydropowerTargetParameter
)

from pywr.recorders import (
    HydropowerRecorder
)

class Turbine(Link, metaclass=NodeMeta):
    """ A hydropower turbine node.

    This node represents a hydropower turbine. It is used to model the generation of electricity from water flow.
    Internally, it uses a HydropowerTargetParameter to calculate the flow required to meet a given generation capacity.
    along with a HydropowerRecorder to record the generation and other relevant parameters.

    Parameters
    ----------
    model : Model
        Model instance to which this turbine node is attached.
    name : str
        Name of the node.
    efficiency : float (default=1.0)
        Turbine efficiency.
    density : float (default=1000.0)
        Water density.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/day`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of energy to be compatible with the equation here. This
        should convert energy to units of :math:`MW`
    storage_node : str (default=None)
        Name of the storage node to which this turbine is connected. If not None, the water elevation
        of the storage node is used to calculate the head of the turbine.
    generation_capacity : float, Parameter (default=0.0)
        The maximum generation capacity of the turbine. This is the maximum amount of energy that the
        turbine can generate. This can be a constant value or a parameter, in :math:`MW`.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    min_operating_elevation : double
        Minimum operating elevation of the turbine. This is used to calculate the minimum head of the turbine.
    min_flow : float, Parameter (default=0.0)
        The minimum flow required to operate the turbine. This can be a constant value or a parameter, in :math:`m^3/day`.
    """
    def __init__(self, model, name, **kwargs):
        hp_recorder_kwarg_names = ('efficiency', 'density', 'flow_unit_conversion', 'energy_unit_conversion')
        self.hp_kwargs = {}
        for kwd in hp_recorder_kwarg_names:
            try:
                self.hp_kwargs[kwd] = kwargs.pop(kwd)
            except KeyError:
                pass

        self.storage_node = kwargs.pop("storage_node", None)
        self.turbine_elevation = kwargs.pop('turbine_elevation', 0)
        self.generation_capacity = kwargs.pop('generation_capacity', 0)
        self.min_operating_elevation = kwargs.pop('min_operating_elevation', 0)
        self.min_head = self.min_operating_elevation - self.turbine_elevation

        super().__init__(model, name, **kwargs)

        if isinstance(self.generation_capacity, (float, int)):
            self.generation_capacity = ConstantParameter(model, self.generation_capacity)


    @classmethod
    def pre_load(cls, model, data):
        name = data.pop("name")
        cost = data.pop("cost", 0.0)
        min_flow = data.pop("min_flow", None)


        node = cls(name=name, model=model, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        if cost is None:
            cost = ConstantParameter(node.model, 0.0)
        if min_flow is None:
            min_flow = ConstantParameter(node.model, 0.0)

        try:
            float(min_flow)
            min_flow = ConstantParameter(node.model, min_flow)
        except:
            pass

        node.cost = cost
        node.min_flow = min_flow
        setattr(node, "_Loadable__parameters_to_load", {})

        return node
    
    def finalise_load(self):
        super().finalise_load()

        level_parameter = None
        if self.storage_node is not None:
            storage_node = self.model.nodes[self.storage_node]
            if hasattr(storage_node, 'level') and not isinstance(storage_node.level, Parameter):
                if storage_node._Loadable__parameters_to_load.get('level'):
                    level_parameter = self.model.parameters[storage_node._Loadable__parameters_to_load['level']]
                else:
                    try:
                        float(storage_node.level)
                        level_parameter = ConstantParameter(self.model, storage_node.level)
                    except:
                        raise Exception(f"An unknown value {storage_node.level} was set on the storage node level.")

        hp_target_flow = HydropowerTargetParameter(self.model, self.generation_capacity,
                                                   water_elevation_parameter=level_parameter,
                                                   min_head=self.min_head, min_flow=self.min_flow,
                                                   turbine_elevation=self.turbine_elevation,
                                                   **self.hp_kwargs)

        self.max_flow = hp_target_flow

        hp_recorder = HydropowerRecorder(self.model, self,
                                         name=f"__{self.name}__:hydropower recorder",
                                         water_elevation_parameter=level_parameter,
                                         turbine_elevation=self.turbine_elevation, **self.hp_kwargs)
        self.hydropower_recorder = hp_recorder
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
    """
    TODO: Explain what this does
    """
    def __init__(self, model, name, **kwargs):
        hp_recorder_kwarg_names = ('efficiency', 'density', 'flow_unit_conversion', 'energy_unit_conversion')
        hp_kwargs = {}
        for kwd in hp_recorder_kwarg_names:
            try:
                hp_kwargs[kwd] = kwargs.pop(kwd)
            except KeyError:
                pass

        level_parameter = None
        storage_node = kwargs.pop("storage_node", None)

        if storage_node is not None:
            storage_node = model.pre_load_node(storage_node)
            self.storage_node = storage_node
            if hasattr(storage_node, "level") and storage_node.level is not None:
                if not isinstance(storage_node.level, Parameter):
                    level_parameter = ConstantParameter(model, value=storage_node.level)
                else:
                    level_parameter = storage_node.level

        turbine_elevation = kwargs.pop('turbine_elevation', 0)
        generation_capacity = kwargs.pop('generation_capacity', 0)
        min_operating_elevation = kwargs.pop('min_operating_elevation', 0)
        min_head = min_operating_elevation - turbine_elevation

        super().__init__(model, name, **kwargs)

        if isinstance(generation_capacity, (float, int)):
            generation_capacity = ConstantParameter(model, generation_capacity)

        hp_target_flow = HydropowerTargetParameter(model, generation_capacity,
                                                   water_elevation_parameter=level_parameter,
                                                   min_head=min_head, min_flow=ConstantParameter(model, 0),
                                                   turbine_elevation=turbine_elevation,
                                                   **hp_kwargs)

        self.max_flow = hp_target_flow

        hp_recorder = HydropowerRecorder(model, self,
                                         name=f"__{name}__:hydropowerrecorder",
                                         water_elevation_parameter=level_parameter,
                                         turbine_elevation=turbine_elevation, **hp_kwargs)
        self.hydropower_recorder = hp_recorder

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
        setattr(node, "_Loadable__parameters_to_load", {})

        return node

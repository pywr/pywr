from ._parameters import Parameter, IndexParameter

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Model
    from ..core import ScenarioIndex, Timestep, Node


class OtherModelParameterValueParameter(Parameter):
    """This parameter gets the value from another model's parameter.

    Attributes
    ----------
    model : Model
        The current model instance.
    other_model : Model
        The other model instance.
    parameter : Parameter
        The parameter instance from `other_model`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(
        self, model: "Model", other_model: "Model", parameter: "Parameter", **kwargs
    ):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The current model instance.
        other_model : Model
            The other model instance.
        parameter : Parameter
            The parameter instance from `other_model`.

        Other parameters
        ----------
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.parameter = parameter

        self._other_model = None
        self._other_model_parameter = None

    def setup(self):
        """Initialise the internal references to the other model's instances."""
        super().setup()
        # Find the references to the other model and one of its parameters.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_parameter = self._other_model.parameters[self.parameter]

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.

        Returns
        -------
        float
            The parameter value.
        """
        return self._other_model_parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model: "Model", data: dict) -> "OtherModelParameterValueParameter":
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        OtherModelParameterValueParameter
            The loaded class.
        """
        return cls(model, **data)


OtherModelParameterValueParameter.register()


class OtherModelIndexParameterValueIndexParameter(IndexParameter):
    """This parameter gets the value from another model's parameter index.

    Attributes
    ----------
    model : Model
        The current model instance.
    other_model : Model
        The other model instance.
    parameter : ParameterIndex
        The parameter index instance from `other_model`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(
        self,
        model: "Model",
        other_model: "Model",
        parameter: "ParameterIndex",
        **kwargs,
    ):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The current model instance.
        other_model : Model
            The other model instance.
        parameter : ParameterIndex
            The parameter index instance from `other_model`.

        Other parameters
        ----------
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.parameter = parameter

        self._other_model = None
        self._other_model_parameter = None

    def setup(self):
        """Initialise the internal references to the other model's instances."""
        super().setup()
        # Find the references to the other model and one of its parameters.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_parameter = self._other_model.parameters[self.parameter]

    def index(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> int:
        """Get the parameter index for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.

        Returns
        -------
        int
            The parameter index.
        """
        return self._other_model_parameter.get_index(scenario_index)

    @classmethod
    def load(
        cls, model: "Model", data: dict
    ) -> "OtherModelIndexParameterValueIndexParameter":
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        OtherModelIndexParameterValueIndexParameter
            The loaded class.
        """
        return cls(model, **data)


OtherModelIndexParameterValueIndexParameter.register()


class OtherModelNodeFlowParameter(Parameter):
    """This parameter gets the flow from another model's node.

    Attributes
    ----------
    model : Model
        The current model instance.
    other_model : Model
        The other model instance.
    node : Node
        The node instance from `other_model`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(self, model: "Model", other_model: "Model", node: "Node", **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The current model instance.
        other_model : Model
            The other model instance.
        node : Node
            The node instance from `other_model`.

        Other parameters
        ----------
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """

        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.node = node

        self._other_model = None
        self._other_model_node = None

    def setup(self):
        """Initialise the internal references to the other model's instances."""
        super().setup()
        # Find the references to the other model and one of its nodes.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_node = self._other_model.nodes[self.node]

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.

        Returns
        -------
        float
            The parameter value.
        """
        return self._other_model_node.flow[scenario_index.global_id]

    @classmethod
    def load(cls, model: "Model", data: dict) -> "OtherModelNodeFlowParameter":
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        OtherModelNodeFlowParameter
            The loaded class.
        """
        return cls(model, **data)


OtherModelNodeFlowParameter.register()


class OtherModelNodeStorageParameter(Parameter):
    """This parameter gets the absolute storage from another model's node.

    Attributes
    ----------
    model : Model
        The current model instance.
    other_model : Model
        The other model instance.
    node : Storage
        The storage node instance from `other_model`.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(self, model: "Model", other_model: "Model", node: "Storage", **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The current model instance.
        other_model : Storage
            The other model instance.
        node : Node
            The storage node instance from `other_model`.

        Other parameters
        ----------
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        """
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.node = node

        self._other_model = None
        self._other_model_node = None

    def setup(self):
        """Initialise the internal references to the other model's instances."""
        super().setup()
        # Find the references to the other model and one of its nodes.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_node = self._other_model.nodes[self.node]

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.

        Returns
        -------
        float
            The parameter value.
        """
        return self._other_model_node.volume[scenario_index.global_id]

    @classmethod
    def load(cls, model: "Model", data: dict) -> "OtherModelNodeStorageParameter":
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        OtherModelNodeStorageParameter
            The loaded class.
        """
        return cls(model, **data)


OtherModelNodeStorageParameter.register()

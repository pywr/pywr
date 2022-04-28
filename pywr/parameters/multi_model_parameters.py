from ._parameters import Parameter, IndexParameter


class OtherModelParameterValueParameter(Parameter):
    """Parameter that obtains its value from another model's parameter."""

    def __init__(self, model, other_model, parameter, **kwargs):
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.parameter = parameter

        self._other_model = None
        self._other_model_parameter = None

    def setup(self):
        super().setup()
        # Find the references to the other model and one of its parameters.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_parameter = self._other_model.parameters[self.parameter]

    def value(self, ts, scenario_index):
        return self._other_model_parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


OtherModelParameterValueParameter.register()


class OtherModelIndexParameterValueIndexParameter(IndexParameter):
    """Parameter that obtains its value from another model's parameter."""

    def __init__(self, model, other_model, parameter, **kwargs):
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.parameter = parameter

        self._other_model = None
        self._other_model_parameter = None

    def setup(self):
        super().setup()
        # Find the references to the other model and one of its parameters.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_parameter = self._other_model.parameters[self.parameter]

    def index(self, ts, scenario_index):
        return self._other_model_parameter.get_index(scenario_index)

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


OtherModelIndexParameterValueIndexParameter.register()


class OtherModelNodeFlowParameter(Parameter):
    """Parameter that obtains its value from the flow through a node in another model."""

    def __init__(self, model, other_model, node, **kwargs):
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.node = node

        self._other_model = None
        self._other_model_node = None

    def setup(self):
        super().setup()
        # Find the references to the other model and one of its nodes.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_node = self._other_model.nodes[self.node]

    def value(self, ts, scenario_index):
        return self._other_model_node.flow[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


OtherModelNodeFlowParameter.register()


class OtherModelNodeStorageParameter(Parameter):
    """Parameter that obtains its value from the current storage of a node in another model."""

    def __init__(self, model, other_model, node, **kwargs):
        super().__init__(model, **kwargs)
        self.other_model = other_model
        self.node = node

        self._other_model = None
        self._other_model_node = None

    def setup(self):
        super().setup()
        # Find the references to the other model and one of its nodes.
        self._other_model = self.model.parent.models[self.other_model]
        self._other_model_node = self._other_model.nodes[self.node]

    def value(self, ts, scenario_index):
        return self._other_model_node.volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)


OtherModelNodeStorageParameter.register()

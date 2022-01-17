from ._parameters import Parameter


class OtherModelParameterValue(Parameter):
    """Parameter that obtains its value from another model's parameter."""

    def __init__(self, model, other_model, parameter, **kwargs):
        super(OtherModelParameterValue, self).__init__(model, **kwargs)
        self.other_model = other_model
        self.parameter = parameter

    def value(self, ts, scenario_index):
        return self.parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        other_model_name = data.pop("other_model")
        other_model = model.parent.models[other_model_name]
        parameter_name = data.pop("parameter")
        parameter = other_model.parameters[parameter_name]

        return cls(model, other_model, parameter, **data)


OtherModelParameterValue.register()

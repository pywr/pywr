from marshmallow import fields
from pywr.parameters import load_parameter


class ParameterField(fields.Field):
    """ Marshmallow field representing a Parameter. """

    def _serialize(self, value, attr, obj):
        raise NotImplementedError('Serializing Parameters is not yet supported.')

    def _deserialize(self, value, attr, data):
        model = self.context['model']
        return load_parameter(model, value)





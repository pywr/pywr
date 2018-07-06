from marshmallow import fields
from pywr.parameters import load_parameter_values

class ParameterField(fields.Field):
    """ Marshmallow field representing a Parameter. """

    def _serialize(self, value, attr, obj):
        raise NotImplementedError('Serializing Parameters is not yet supported.')

    def _deserialize(self, value, attr, data):
        return value

class ParameterValuesField(fields.Field):
    """ Marshmallow field representing a ParameterValues. """

    def _serialize(self, value, attr, obj):
        raise NotImplementedError('Serializing Parameters is not yet supported.')

    def _deserialize(self, value, attr, data):
        model = self.context['model']
        klass = self.context.get('klass', None)
        obj = self.context.get('obj', None)

        # Try to coerce initial volume to float.
        try:
            return float(value)
        except TypeError:
            return load_parameter_values(model, value)

class NodeField(fields.Field):
    """ Marshmallow field representing a node. """
    def _serialize(self, value, attr, obj):
        raise NotImplementedError('Serializing Nodes is not yet supported.')

    def _deserialize(self, value, attr, data):
        model = self.context['model']
        klass = self.context.get('klass', None)
        obj = self.context.get('obj', None)

        return model._get_node_from_ref(model, value)



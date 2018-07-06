import marshmallow
from .fields import ParameterField
from pywr.parameters import load_parameter


class NodeSchema(marshmallow.Schema):
    """ Default Schema for all Pywr nodes.

    This Schema is used for loading Nodes. The `make_node` method is
    decorated as `post_load`. It creates the Node instances. However,
    it avoids loading `Parameter` instances during this phases to avoid
    issues with circular references. Each node must complete a second
    load phase via the `load_parameters` method.
    """
    name = marshmallow.fields.Str()
    type = marshmallow.fields.Str()
    comment = marshmallow.fields.Str()
    position = marshmallow.fields.Dict()

    @marshmallow.validates_schema(pass_original=True)
    def check_unknown_fields(self, data, original_data):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_node(self, data):
        """ Create or append data to a node object. """
        model = self.context['model']
        klass = self.context['klass']

        param_data = {}
        non_param_data = {}

        # Separate the data values in to parameter and non-parameter
        for name, value in data.items():
            if isinstance(self.fields[name], ParameterField):
                param_data[name] = data[name]
            else:
                non_param_data[name] = data[name]

        if klass is not None:
            # Create a new instance with the non-parameter data
            return klass(model, **non_param_data)

    def load_parameters(self, data, obj):
        """ Load parameter data and set it on `obj`."""
        model = self.context['model']
        # Load and assign parameters to an existing node instance.
        for field_name, field in self.fields.items():
            if not isinstance(field, ParameterField):
                continue

            try:
                field_data = data[field.name]
            except KeyError:
                if field.required:
                    raise marshmallow.ValidationError('Missing field.')
                else:
                    continue
            param = load_parameter(model, field_data)
            setattr(obj, field.name, param)
        return obj








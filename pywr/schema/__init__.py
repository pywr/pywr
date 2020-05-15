import marshmallow
from .fields import ParameterReferenceField
import logging
logger = logging.getLogger(__name__)


class PywrSchema(marshmallow.Schema):
    """ Base class for all Pywr marshmallow schemas. """
    def handle_error(self, exc, data, **kwargs):
        """Log and raise our custom exception when (de)serialization fails."""
        klass = self.context['klass']

        msg = 'Error(s) occurred with the data for class {}: {}\n' \
              '  Validation error messages: \n'.format(klass, data)
        for k, msgs in exc.messages.items():
            msg += '    {}: {}\n'.format(k, msgs)

        logging.error(msg)
        raise marshmallow.exceptions.ValidationError(msg)


class NodeSchema(PywrSchema):
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
    def check_unknown_fields(self, data, original_data, **kwargs):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_node(self, data, **kwargs):
        """ Create or append data to a node object. """
        model = self.context['model']
        klass = self.context['klass']

        param_data = {}
        non_param_data = {}

        # Separate the data values in to parameter and non-parameter
        for name, value in data.items():
            if isinstance(self.fields[name], ParameterReferenceField):
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
            if not isinstance(field, ParameterReferenceField):
                continue

            try:
                field_data = data[field.name]
            except KeyError:
                if field.required:
                    raise marshmallow.ValidationError('Missing field.')
                else:
                    continue
            from pywr.parameters import load_parameter

            try:
                param = load_parameter(model, field_data)
            except Exception as e:
                raise Exception(f"Error loading parameter {field.name} on {obj.name}.\n"
                                f"Error Was: {e}")

            setattr(obj, field.name, param)
        return obj


class ComponentSchema(PywrSchema):
    name = marshmallow.fields.Str()
    comment = marshmallow.fields.Str()

    # TODO move this to a common base class for Nodes and Component schemas
    @marshmallow.validates_schema(pass_original=True)
    def check_unknown_fields(self, data, original_data, **kwargs):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_component(self, data, **kwargs):
        """ Create or append data to a node object. """
        model = self.context['model']
        klass = self.context['klass']
        return klass(model, **data)


class ParameterSchema(ComponentSchema):
    pass


class DataFrameSchema(ParameterSchema):
    __mutually_exclusive_fields__ = ('data', 'url', 'table')
    data = marshmallow.fields.Dict(required=False)
    url = marshmallow.fields.String(required=False)
    table = marshmallow.fields.String(required=False)
    pandas_kwargs = marshmallow.fields.Dict()
    #  TODO add validator for these fields. Could be str or int.
    index = marshmallow.fields.Raw()
    column = marshmallow.fields.Raw()
    key = marshmallow.fields.Raw()
    checksum = marshmallow.fields.Dict()

    @marshmallow.validates_schema()
    def validate_input_type(self, data, **kwargs):
        count = 0
        for input_field in self.__mutually_exclusive_fields__:
            if input_field in data:
                count += 1

        if count > 1:
            raise marshmallow.ValidationError('Only one of "data", "url" or "table" fields '
                                              'should be given.')
        elif count < 0:
            raise marshmallow.ValidationError('One of "data", "url" or "table" fields '
                                              'must be given.')

    @marshmallow.post_load
    def make_component(self, data, **kwargs):
        """ Create or append data to a node object. """
        from pywr.parameters import load_dataframe
        model = self.context['model']
        klass = self.context['klass']
        df = load_dataframe(model, data)
        return klass(model, dataframe=df, **data)


class ExternalDataSchema(DataFrameSchema):
    __mutually_exclusive_fields__ = ('data', 'url', 'table', 'values')
    values = marshmallow.fields.List(marshmallow.fields.Float())

    @marshmallow.post_load
    def make_component(self, data, **kwargs):
        """ Create or append data to a node object. """
        print(data)
        from pywr.parameters import load_parameter_values
        model = self.context['model']
        klass = self.context['klass']
        values = load_parameter_values(model, data)
        print(self, data)
        return klass(model, values=values, **data)

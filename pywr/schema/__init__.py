import marshmallow
from .fields import ParameterReferenceField
import logging
logger = logging.getLogger(__name__)


class PywrSchema(marshmallow.Schema):
    """ Base class for all Pywr marshmallow schemas. """
    def handle_error(self, exc, data):
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
            param = load_parameter(model, field_data)
            setattr(obj, field.name, param)
        return obj


# TODO organise these two classes better they have similar fields.
class ExternalDataSchemaMixin(marshmallow.Schema):
    __values_arg_name__ = 'values'
    values = marshmallow.fields.List(marshmallow.fields.Float())
    url = marshmallow.fields.String()
    table = marshmallow.fields.String()
    #  TODO add validator for these fields. Could be str or int.
    index_col = marshmallow.fields.Raw()
    index = marshmallow.fields.Raw()
    column = marshmallow.fields.Raw()
    key = marshmallow.fields.Raw()
    checksum = marshmallow.fields.Dict()
    parse_dates = marshmallow.fields.Boolean()
    dayfirst = marshmallow.fields.Boolean()
    sheetname = marshmallow.fields.String()


class DataFrameSchemaMixin(marshmallow.Schema):
    __values_arg_name__ = 'dataframe'
    url = marshmallow.fields.String()
    #  TODO add validator for these fields. Could be str or int.
    index_col = marshmallow.fields.Raw()
    index = marshmallow.fields.Raw()
    column = marshmallow.fields.Raw()
    key = marshmallow.fields.Raw()
    checksum = marshmallow.fields.Dict()
    parse_dates = marshmallow.fields.Boolean()
    dayfirst = marshmallow.fields.Boolean()
    sheetname = marshmallow.fields.String()


class ComponentSchema(PywrSchema):
    name = marshmallow.fields.Str()
    comment = marshmallow.fields.Str()

    # TODO move this to a common base class for Nodes and Component schemas
    @marshmallow.validates_schema(pass_original=True)
    def check_unknown_fields(self, data, original_data):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_component(self, data):
        """ Create or append data to a node object. """
        model = self.context['model']
        klass = self.context['klass']

        # TODO this if block could be similar it is copy/paste hell atm.
        if isinstance(self, ExternalDataSchemaMixin):
            from pywr.parameters import load_parameter_values

            # This seems a bit dodgy.
            external_data = {}
            non_external_data = {}
            for name, value in data.items():
                if name in ExternalDataSchemaMixin().fields:
                    external_data[name] = data[name]
                else:
                    non_external_data[name] = data[name]

            if len(external_data) > 0 and self.__values_arg_name__ not in non_external_data:
                values = load_parameter_values(model, external_data)
                non_external_data[self.__values_arg_name__] = values

            return klass(model, **non_external_data)
        elif isinstance(self, DataFrameSchemaMixin):
            from pywr.parameters import load_dataframe

            # This seems a bit dodgy.
            external_data = {}
            non_external_data = {}
            for name, value in data.items():
                if name in DataFrameSchemaMixin().fields:
                    external_data[name] = data[name]
                else:
                    non_external_data[name] = data[name]

            if len(external_data) > 0 and self.__values_arg_name__ not in non_external_data:
                values = load_dataframe(model, external_data)
                non_external_data[self.__values_arg_name__] = values

            return klass(model, **non_external_data)

        else:
            return klass(model, **data)

import marshmallow


class NodeSchema(marshmallow.Schema):
    """ Default schema for all Pywr nodes. """
    name = marshmallow.fields.Str()
    type = marshmallow.fields.Str()
    comment = marshmallow.fields.Str()

    @marshmallow.validates_schema(pass_original=True)
    def check_unknown_fields(self, data, original_data):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_node(self, data):
        """ Create the node object. """
        model = self.context['model']
        klass = self.context['klass']
        return klass(model, **data)

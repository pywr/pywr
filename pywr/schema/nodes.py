import marshmallow




class BaseNodeSchema(marshmallow.Schema):
    """ Default schema for all Pywr nodes. """
    name = marshmallow.fields.Str()
    type = marshmallow.fields.Str()
    comment = marshmallow.fields.Str()
    position = marshmallow.fields.Dict()

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register subclasses by name
        name = cls.__name__.lower()
        cls.subclasses[name] = cls

    @classmethod
    def get_schema(cls, node_type, model, node_cls_to_load=None):
        from ..nodes import NodeMeta, Node, Storage

        node_cls = NodeMeta.node_registry[node_type]
        if node_cls_to_load is None:
            node_cls_to_load = node_cls

        try:
            schema_cls = cls.subclasses[node_type+'schema']
        except KeyError:
            try:
                schema_cls = cls.subclasses[node_type]
            except KeyError:
                schema_cls = None

        if schema_cls is None:
            # No schema found. Try the parent's schema instead
            for parent_cls in node_cls.__bases__:
                if issubclass(parent_cls, (Node, Storage)):
                    return cls.get_schema(parent_cls.__name__.lower(), model,
                                          node_cls_to_load=node_cls_to_load)
            else:
                raise ValueError('No schema found for node class: "{}"'.format(node_cls))
        else:
            return schema_cls(context={'model': model, 'cls': node_cls_to_load})

    @marshmallow.validates_schema(pass_original=True)
    def check_unknown_fields(self, data, original_data):
        unknown = set(original_data) - set(self.fields)
        if unknown:
            raise marshmallow.ValidationError('Unknown field', unknown)

    @marshmallow.post_load
    def make_node(self, data):
        # Defers loading of nodes to the load classmethod on each node class.
        klass = self.context['cls']
        model = self.context['model']
        return klass.load(data, model)




class NodeSchema(BaseNodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flow = marshmallow.fields.Raw(allow_none=True)
    min_flow = marshmallow.fields.Raw(allow_none=True)
    cost = marshmallow.fields.Raw()


class StorageSchema(BaseNodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_volume = marshmallow.fields.Raw(required=False)
    min_volume = marshmallow.fields.Raw(required=False)
    cost = marshmallow.fields.Raw(required=False)
    initial_volume = marshmallow.fields.Raw(required=False)
    initial_volume_pc = marshmallow.fields.Number(required=False)
    level = marshmallow.fields.Raw(required=False)
    area = marshmallow.fields.Raw(required=False)
    inputs = marshmallow.fields.Integer(required=False, default=1)
    outputs = marshmallow.fields.Integer(required=False, default=1)


class VirtualStorageSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    nodes = marshmallow.fields.List(marshmallow.fields.Str, required=True)
    max_volume = marshmallow.fields.Raw(required=False)
    min_volume = marshmallow.fields.Raw(required=False)
    cost = marshmallow.fields.Raw(required=False)
    initial_volume = marshmallow.fields.Raw(required=False)
    factors = marshmallow.fields.List(marshmallow.fields.Number, required=True)


class AnnualVirtualStorageSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    nodes = marshmallow.fields.List(marshmallow.fields.Str, required=True)
    max_volume = marshmallow.fields.Raw(required=False)
    min_volume = marshmallow.fields.Raw(required=False)
    cost = marshmallow.fields.Raw(required=False)
    initial_volume = marshmallow.fields.Raw(required=False)
    factors = marshmallow.fields.List(marshmallow.fields.Number, required=True)
    reset_day = marshmallow.fields.Integer()
    reset_month = marshmallow.fields.Integer()


class PiecewiseLinkSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flows = marshmallow.fields.List(marshmallow.fields.Raw)
    max_flow = marshmallow.fields.List(marshmallow.fields.Raw(allow_none=True))
    costs = marshmallow.fields.List(marshmallow.fields.Raw)
    cost = marshmallow.fields.List(marshmallow.fields.Number)


class AggregatedStorageSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    storage_nodes = marshmallow.fields.List(marshmallow.fields.Str())


class AggregatedNodeSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flow = marshmallow.fields.Raw(required=False)
    min_flow = marshmallow.fields.Raw(required=False)
    factors = marshmallow.fields.List(marshmallow.fields.Number, required=False)
    flow_weights = marshmallow.fields.List(marshmallow.fields.Number, required=False)
    nodes = marshmallow.fields.List(marshmallow.fields.Str())


class CatchmentSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    flow = marshmallow.fields.Raw(allow_none=True)
    cost = marshmallow.fields.Raw()


class RiverSplitSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flow = marshmallow.fields.List(marshmallow.fields.Number)
    cost = marshmallow.fields.List(marshmallow.fields.Number)
    factors = marshmallow.fields.List(marshmallow.fields.Number)
    slot_names = marshmallow.fields.List(marshmallow.fields.Str)


class RiverSplitWithGaugeSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flow = marshmallow.fields.Raw(allow_none=True)
    mrf = marshmallow.fields.Raw()
    cost = marshmallow.fields.Raw()
    mrf_cost = marshmallow.fields.Raw()
    factors = marshmallow.fields.List(marshmallow.fields.Number)
    slot_names = marshmallow.fields.List(marshmallow.fields.Str)


class RiverGaugeSchema(NodeSchema):
    # The main attributes are not validated (i.e. `Raw`)
    # They could be many different things.
    max_flow = marshmallow.fields.Raw(allow_none=True)
    mrf = marshmallow.fields.Raw()
    cost = marshmallow.fields.Raw()
    mrf_cost = marshmallow.fields.Raw()
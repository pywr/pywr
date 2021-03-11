import logging


logger = logging.getLogger(__name__)


ROOT_NODE = "root"


class GraphInterface:
    def __init__(self, obj):
        self.obj = obj

    @property
    def graph(self):
        return self.obj.model.component_graph

    def add(self, item):
        if not isinstance(item, Component):
            return
        if self is self.obj.children:
            self.graph.add_edge(item, self.obj)
        else:
            self.graph.add_edge(self.obj, item)

    def remove(self, item):
        if not isinstance(item, Component):
            return
        if self is self.obj.children:
            self.graph.remove_edge(item, self.obj)
        else:
            self.graph.remove_edge(self.obj, item)

    def clear(self):
        for n in self._members:
            if self is self.obj.children:
                self.graph.remove_edge(n, self.obj)
            else:
                self.graph.remove_edge(self.obj, n)

    @property
    def _members(self):
        if self is self.obj.children:
            return [n for n in self.graph.predecessors(self.obj) if n != ROOT_NODE]
        else:
            return [x for x in self.graph.successors(self.obj)]

    def __len__(self):
        """Returns the number of nodes in the model"""
        return len(self._members)

    def __iter__(self):
        return iter(self._members)


cdef class Component:
    """ Components of a Model

    This is the base class for all the elements of a `pywr.Model`,
     except the the nodes, that require updates via the `setup`, `reset`,
     `before`, `after` and `finish` methods. This class handles
     registering the instances on the `Model.component_graph` and
     managing the parent/children interface.

    The parent/children interface, through the `pywr.Model.component_graph`
     is used to create a dependency tree such that the methods are
     called in the correct order. E.g. that a `before` method in
     one component that is a parent of another is called first.

    Parameters
    ==========
    name : str or None
        The name of the component.
    comment : str or None
        An optional comment for the component.
    tags : dict (default=None)
        An optional container of key-value pairs that the user can set to help group and identify components.

    See also
    --------
    pywr.Model

    """
    def __init__(self, model, name=None, comment=None, tags=None):
        self.model = model
        self.name = name
        self.comment = comment
        self.tags = tags
        model.component_graph.add_edge(ROOT_NODE, self)
        self.parents = GraphInterface(self)
        self.children = GraphInterface(self)

    property name:
        def __get__(self):
            return self._name

        def __set__(self, name):
            # check for name collision
            if name is not None and name in self.model.components.keys():
                raise ValueError('A component with the name "{}" already exists.'.format(name))
            # apply new name
            self._name = name

    cpdef setup(self):
        logger.debug('Setting up {}: "{}"'.format(self.__class__.__name__, self.name))

    cpdef reset(self):
        logger.debug('Resetting up {}: "{}"'.format(self.__class__.__name__, self.name))

    cpdef before(self):
        pass

    cpdef after(self):
        pass

    cpdef finish(self):
        pass

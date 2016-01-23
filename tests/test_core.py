#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pytest
from fixtures import simple_linear_model
from pywr._core import Timestep

from pywr.core import *
from pywr.domains.river import *
from pywr.parameters import Parameter, Timeseries


def test_names(solver):
    '''Test node names'''
    model = Model(solver=solver)

    node1 = Input(model, name='A')
    node2 = Output(model, name='B')
    assert(model.node['A'] is node1)
    assert(model.node['B'] is node2)

    nodes = sorted(model.nodes(), key=lambda node: node.name)
    assert(nodes == [node1, node2])

    # rename node
    node1.name = 'C'
    assert(model.node['C'] is node1)
    assert('A' not in model.node)

    # attempt name collision (via rename)
    with pytest.raises(ValueError):
        node2.name = 'C'

    # attempt name collision (via new)
    with pytest.raises(ValueError):
        node3 = Input(model, name='C')
    assert(len(model.node) == 2)

    # attempt to create a node without a name
    with pytest.raises(TypeError):
        node4 = Input(model)


def test_unexpected_kwarg(solver):
    model = Model(solver=solver)

    with pytest.raises(TypeError):
        node = Node(model, 'test_node', invalid=True)
    with pytest.raises(TypeError):
        inpt = Input(model, 'test_input', invalid=True)
    with pytest.raises(TypeError):
        storage = Storage(model, 'test_storage', invalid=True)
    # none of the nodes should have been added to the model as they all
    # raised exceptions during __init__
    # TODO FIXME: nodes are still added, even if __init__ raises exception
    # assert(not model.nodes())


# TODO Update this test. Blender is not implemented.
@pytest.mark.xfail
def test_slots_to(solver):
    model = Model(solver=solver)

    supply1 = Input(model, name='supply1')
    supply2 = Input(model, name='supply2')
    blender = Blender(model, name='blender', ratio=0.5)

    # attempt to connect to invalid slot
    with pytest.raises(ValueError):
        supply1.connect(blender, to_slot=3)
    assert((supply1, blender) not in model.edges())

    supply1.connect(blender, to_slot=1)
    assert((supply1, blender) in model.edges())

    # a node can connect to multiple slots
    supply1.connect(blender, to_slot=2)
    assert(blender.slots[1] is supply1)
    assert(blender.slots[2] is supply1)

    supply1.disconnect(blender)
    assert((supply1, blender) not in model.edges())
    # as well as removing the edge, the node should be removed from the slot
    assert(supply1 not in blender.slots.values())

    # override slot with another node
    supply1.connect(blender, to_slot=1)
    supply1.connect(blender, to_slot=2)
    supply2.connect(blender, to_slot=2)
    assert((supply1, blender) in model.edges())
    assert((supply2, blender) in model.edges())
    assert(blender.slots[1] is supply1)
    assert(blender.slots[2] is supply2)

# TODO Update this test. RiverSplit is deprecated.
@pytest.mark.xfail
def test_slots_from(solver):
    model = Model(solver=solver)

    riversplit = RiverSplit(model, name='split')
    river1 = River(model, name='river1')
    river2 = River(model, name='river2')

    riversplit.connect(river1, from_slot=1)
    assert((riversplit, river1) in model.edges())

    riversplit.connect(river2, from_slot=2)
    assert((riversplit, river2) in model.edges())

    riversplit.disconnect(river2)
    assert(riversplit.slots[2] is None)
    assert(riversplit.slots[1] is river1)

    riversplit.disconnect()
    assert(len(model.edges()) == 0)


def test_timeseries_csv(solver):
    model = Model(solver=solver)
    ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')
    timestep = Timestep(pandas.to_datetime('2015-01-31'), 0, 1)
    assert(ts.value(timestep) == 21.92)


def test_timeseries_excel(solver):
    model = Model(solver=solver)
    ts = Timeseries.read(model, name='ts', path='tests/timeseries1.xlsx', sheet='mydata', column='Data')
    timestep = Timestep(pandas.to_datetime('2015-01-31'), 0, 1)
    assert(ts.value(timestep) == 21.92)


def test_timeseries_name_collision(solver):
    model = Model(solver=solver)
    ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')
    with pytest.raises(ValueError):
        ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')


def test_dirty_model(solver):
    """Test that the LP is updated when the model structure is redefined"""
    # start dirty
    model = Model(solver=solver)
    assert(model.dirty)

    # add some nodes, still dirty
    supply1 = Input(model, 'supply1')
    demand1 = Output(model, 'demand1')
    supply1.connect(demand1)
    assert(model.dirty)

    # run the model, clean
    result = model.step()
    assert(not model.dirty)

    # add a new node, dirty
    supply2 = Input(model, 'supply2')

    # run the model, clean
    result = model.step()
    assert(not model.dirty)

    # add a new connection, dirty
    supply2.connect(demand1)
    assert(model.dirty)

    # run the model, clean
    result = model.step()
    assert(not model.dirty)

    # remove a connection, dirty
    supply2.disconnect()
    assert(model.dirty)


def test_shorthand_property(solver):
    # test shorthand assignment of constant properties
    model = Model(solver=solver)
    node = Node(model, 'node')
    for attr in ('min_flow', 'max_flow', 'cost', 'conversion_factor'):
        # should except int, float or Paramter
        setattr(node, attr, 123)
        if attr == 'conversion_factor':
            with pytest.raises(ValueError):
                setattr(node, attr, Parameter())
        else:
            setattr(node, attr, Parameter())

        with pytest.raises(TypeError):
            setattr(node, attr, '123')
            setattr(node, attr, None)


def test_reset_before_run(solver):
    # See issue #82. Previously this would raise:
    #    AttributeError: Memoryview is not initialized
    model = Model(solver=solver)
    node = Node(model, 'node')
    model.reset()


def test_check_isolated_nodes(simple_linear_model):
    """Test model storage checker"""
    # the simple model shouldn't have any isolated nodes
    model = simple_linear_model
    model.check()
    
    # add a node, but don't connect it to the network
    isolated_node = Input(model, 'isolated')
    with pytest.raises(ModelStructureError):
        model.check()

def test_check_isolated_nodes_storage(solver):
    """Test model structure checker with Storage
    
    The Storage node itself doesn't have any connections, but it's child
    nodes do need to be connected.
    """
    model = Model(solver=solver)
    
    # add a storage, but don't connect it's outflow to anything
    storage = Storage(model, 'storage', num_inputs=1, num_outputs=0)
    with pytest.raises(ModelStructureError):
        model.check()

    # add a demand node and connect it to the storage outflow
    demand = Output(model, 'demand')
    storage.connect(demand, from_slot=0)
    model.check()


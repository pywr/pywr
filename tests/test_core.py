#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pytest
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


def test_slots_connect_disconnect(solver):
    """Test connection and disconnection to storage node slots
    """
    model = Model(solver=solver)

    supply1 = Input(model, name='supply1')
    supply2 = Input(model, name='supply2')
    storage = Storage(model, name='storage', num_inputs=2, num_outputs=2)

    storage_inputs = [[x for x in storage.iter_slots(slot_name=n, is_connector=True)][0] for n in (0, 1)]
    storage_outputs = [[x for x in storage.iter_slots(slot_name=n, is_connector=False)][0] for n in (0, 1)]

    # attempt to connect to invalid slot
    # an error is raised, and no connection is made
    with pytest.raises(IndexError):
        supply1.connect(storage, to_slot=3)
    assert((supply1, storage_inputs[0]) not in model.edges())
    assert((supply1, storage_inputs[1]) not in model.edges())

    # connect node to storage slot 0
    supply1.connect(storage, to_slot=0)
    assert((supply1, storage_outputs[0]) in model.edges())
    assert((supply1, storage_inputs[1]) not in model.edges())

    # the same node can be connected to multiple slots
    # connect node to storage slot 1
    supply1.connect(storage, to_slot=1)
    assert((supply1, storage_outputs[0]) in model.edges())
    assert((supply1, storage_outputs[1]) in model.edges())

    # disconnect the node from a particular slot (recommended method)
    supply1.disconnect(storage, slot_name=0)
    assert((supply1, storage_outputs[0]) not in model.edges())
    assert((supply1, storage_outputs[1]) in model.edges())

    # disconnect the node from a particular slot (direct, not recommended)
    supply1.connect(storage, to_slot=0)
    supply1.disconnect(storage_outputs[0])
    assert((supply1, storage_outputs[0]) not in model.edges())
    assert((supply1, storage_outputs[1]) in model.edges())

    # specifying the storage in general removes the connection from all slots
    supply1.connect(storage, to_slot=0)
    supply1.disconnect(storage)
    assert((supply1, storage_outputs[0]) not in model.edges())
    assert((supply1, storage_outputs[1]) not in model.edges())

    # it's an error to attempt to disconnect if nodes aren't connected
    with pytest.raises(Exception):
        supply1.disconnect(storage)


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

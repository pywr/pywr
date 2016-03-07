#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from pywr._core import Timestep
from pywr.core import *
from pywr.domains.river import *
from pywr.parameters import Parameter, Timeseries

import pytest
from fixtures import *

TEST_FOLDER = os.path.dirname(__file__)

def test_names(solver):
    '''Test node names'''
    model = Model(solver=solver)

    node1 = Input(name='A')
    node2 = Output(name='B')
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
        node3 = Input(name='C')
    assert(len(model.node) == 2)  # node3 not added to graph

    # attempt to create a node without a name
    with pytest.raises(TypeError):
        node4 = Input()


def test_model_nodes(model):
    """Test Model.nodes API"""
    node = Input('test')

    # test node by index
    assert(model.nodes['test'] is node)

    with pytest.raises(KeyError):
        model.nodes['invalid']

    # test node iterator
    all_nodes = [node for node in model.nodes]
    assert(all_nodes == [node])

    # support for item deletion
    del(model.nodes['test'])
    all_nodes = [node for node in model.nodes]
    assert(all_nodes == [])


def test_unexpected_kwarg_node(solver):
    model = Model(solver=solver)

    with pytest.raises(TypeError):
        node = Node('test_node', invalid=True)
    with pytest.raises(TypeError):
        inpt = Input('test_input', invalid=True)
    with pytest.raises(TypeError):
        storage = Storage('test_storage', invalid=True)
    # none of the nodes should have been added to the model as they all
    # raised exceptions during __init__
    assert(not model.nodes())


def test_unexpected_kwarg_model(solver):
    with pytest.raises(TypeError):
        model = Model(solver=solver, thisisgoingtofail=True)
    model = Model(solver=solver)

def test_slots_connect_disconnect(solver):
    """Test connection and disconnection to storage node slots
    """
    model = Model(solver=solver)

    supply1 = Input(name='supply1')
    supply2 = Input(name='supply2')
    storage = Storage(name='storage', num_inputs=2, num_outputs=2)

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

    riversplit = RiverSplit(name='split')
    river1 = River(name='river1')
    river2 = River(name='river2')

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
    filename = os.path.join(TEST_FOLDER, 'timeseries1.csv')
    ts = Timeseries.read(model=model, name='ts1', path=filename, column='Data')
    timestep = Timestep(pandas.to_datetime('2015-01-31'), 0, 1)
    assert(ts.value(timestep) == 21.92)


def test_timeseries_excel(solver):
    model = Model(solver=solver)
    filename = os.path.join(TEST_FOLDER, 'timeseries1.xlsx')
    ts = Timeseries.read(model=model, name='ts', path=filename, sheet='mydata', column='Data')
    timestep = Timestep(pandas.to_datetime('2015-01-31'), 0, 1)
    assert(ts.value(timestep) == 21.92)


def test_timeseries_name_collision(solver):
    model = Model(solver=solver)
    filename = os.path.join(TEST_FOLDER, 'timeseries1.csv')
    ts = Timeseries.read(model=model, name='ts1', path=filename, column='Data')
    with pytest.raises(ValueError):
        ts = Timeseries.read(model=model, name='ts1', path=filename, column='Data')


def test_dirty_model(solver):
    """Test that the LP is updated when the model structure is redefined"""
    # start dirty
    model = Model(solver=solver)
    assert(model.dirty)

    # add some nodes, still dirty
    supply1 = Input('supply1')
    demand1 = Output('demand1')
    supply1.connect(demand1)
    assert(model.dirty)

    # run the clean
    result = model.step()
    assert(not model.dirty)

    # add a new node, dirty
    supply2 = Input('supply2')

    # run the clean
    result = model.step()
    assert(not model.dirty)

    # add a new connection, dirty
    supply2.connect(demand1)
    assert(model.dirty)

    # run the clean
    result = model.step()
    assert(not model.dirty)

    # remove a connection, dirty
    supply2.disconnect()
    assert(model.dirty)


def test_shorthand_property(solver):
    # test shorthand assignment of constant properties
    model = Model(solver=solver)
    node = Node('node')
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
    node = Node('node')
    model.reset()


def test_check_isolated_nodes(simple_linear_model):
    """Test model storage checker"""
    # the simple model shouldn't have any isolated nodes
    model = simple_linear_model
    model.check()
    
    # add a node, but don't connect it to the network
    isolated_node = Input('isolated')
    with pytest.raises(ModelStructureError):
        model.check()

def test_check_isolated_nodes_storage(solver):
    """Test model structure checker with Storage
    
    The Storage node itself doesn't have any connections, but it's child
    nodes do need to be connected.
    """
    model = Model(solver=solver)
    
    # add a storage, but don't connect it's outflow to anything
    storage = Storage('storage', num_inputs=1, num_outputs=0)
    with pytest.raises(ModelStructureError):
        model.check()

    # add a demand node and connect it to the storage outflow
    demand = Output('demand')
    storage.connect(demand, from_slot=0)
    model.check()


def test_default_model(solver):
    model = Model(solver=solver)
    
    assert(gcm() is model)
    
    supply1 = Input(name='supply1', max_flow=10)
    demand1 = Output(name='demand1', max_flow=15, cost=-20)
    supply1.connect(demand1)
    
    model.run()
    assert(supply1.flow == [10])


def test_no_model(solver):
    clm()  # clear last model
    with pytest.raises(RuntimeError):
        supply1 = Input(name='supply1', max_flow=10)


def test_explicit_model(solver):
    model = Model(solver=solver)
    # name as kwarg
    supply1 = Input(name='supply1', model=model)
    # name as arg
    demand1 = Output('demand1', model=model)


def test_two_models(solver):
    model1 = Model(solver=solver)
    model2 = Model(solver=solver)
    
    supply1 = Input(name='supply1', max_flow=10, model=model1)
    demand1 = Output(name='demand1', max_flow=15, cost=-20, model=model1)
    supply1.connect(demand1)

    supply2 = Input(name='supply2', max_flow=100, model=model2)
    demand2 = Output(name='demand2', max_flow=150, cost=-200, model=model2)
    supply2.connect(demand2)
    
    model1.run()
    model2.run()
    assert(supply1.flow == [10])
    assert(supply2.flow == [100])

    # attempt connection between models
    with pytest.raises(RuntimeError):
        supply1.connect(demand2)
    
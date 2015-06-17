#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import pytest

from pywr.core import *

def test_names():
    '''Test node names'''
    model = Model()
    
    node1 = Supply(model, name='A')
    node2 = Demand(model, name='B')
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
        node3 = Supply(model, name='C')
    assert(len(model.node) == 2)

    # attempt to create a node without a name
    with pytest.raises(TypeError):
        node4 = Supply(model)

def test_slots_to():
    model = Model()
    
    supply1 = Supply(model, name='supply1')
    supply2 = Supply(model, name='supply2')
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

def test_slots_from():
    model = Model()
    
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

def test_rivergauge_mrf():
    # test programmatic creation of a river gauge with an MRF
    model = Model()
    node = RiverGauge(model, name='gauge', mrf=42.0)
    assert(node.properties['mrf'].value(None) == 42.0)

def test_timeseries_csv():
    model = Model()
    ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')
    assert(ts.value(pandas.to_datetime('2015-01-31')) == 21.92)

def test_timeseries_excel():
    model = Model()
    ts = Timeseries.read(model, name='ts', path='tests/timeseries1.xlsx', sheet='mydata', column='Data')
    assert(ts.value(pandas.to_datetime('2015-01-31')) == 21.92)

def test_timeseries_name_collision():
    model = Model()
    ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')
    with pytest.raises(ValueError):
        ts = Timeseries.read(model, name='ts1', path='tests/timeseries1.csv', column='Data')

def test_dirty_model():
    """Test that the LP is updated when the model structure is redefined"""
    # start dirty
    model = Model()
    assert(model.dirty)

    # add some nodes, still dirty
    supply1 = Supply(model, 'supply1')
    demand1 = Demand(model, 'demand1')
    supply1.connect(demand1)
    assert(model.dirty)

    # run the model, clean
    result = model.step()
    assert(not model.dirty)

    # add a new node, dirty
    supply2 = Supply(model, 'supply2')

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

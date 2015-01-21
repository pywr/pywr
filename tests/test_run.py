#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime
import pytest

import pywr.core
import pywr.xmlutils

def test_run_simple1():
    '''Test the most basic model possible'''
    # parse the XML into a model
    data = file(os.path.join(os.path.dirname(__file__), 'simple1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)

    # run the model
    t0 = model.timestamp
    result = model.step()
    
    # check results
    assert(result == ('optimal', 10.0, 10.0))
    
    # check the timestamp incremented
    assert(model.timestamp - t0 == datetime.timedelta(1))

def test_run_reservoir1():
    '''Test a reservoir with no refill
    
    Without an additional supply the reservoir should empty and cause a failure.
    '''
    data = file(os.path.join(os.path.dirname(__file__), 'reservoir1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()

    for delivered in [10.0, 10.0, 10.0, 5.0, 0.0]:
        result = model.step()
        assert(result == ('optimal', 10.0, delivered))

def test_run_reservoir2():
    '''Test a reservoir fed by a river abstraction
    
    The river abstraction should refill the reservoir, but not quickly enough
    to keep up with the demand.
    '''
    data = file(os.path.join(os.path.dirname(__file__), 'reservoir2.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    for demand, supply in [(10.0, 10.0), (20.0, 14.0), (26.0, 14.0), (32.0, 14.0), (38.0, 11.0), (41.0, 8.0), (41.0, 8.0)]:
        result = model.step()
        assert(result == ('optimal', demand, supply))

def test_run_river1():
    '''Test a river abstraction with a simple catchment'''
    data = file(os.path.join(os.path.dirname(__file__), 'river1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result == ('optimal', 10.0, 5.0))

def test_run_river2():
    '''Test a river abstraction with two catchments, a confluence and a split'''
    data = file(os.path.join(os.path.dirname(__file__), 'river2.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result == ('optimal', 12.0, 9.25))

def test_run_cost1():
    data = file(os.path.join(os.path.dirname(__file__), 'cost1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    nodes = dict([(node.name, node) for node in model.nodes()])
    assert(nodes['supply1'].properties['cost'].value(None) == 1)
    assert(nodes['supply2'].properties['cost'].value(None) == 2)
    
    result = model.step()
    assert(result == ('optimal', 10.0, 10.0))
    
    # TODO: check that the supply has come entirely from supply1

def test_solver_cylp():
    '''Test specifying the solver in XML'''
    data = '''<pywr><solver name="cylp" /><nodes /><edges /><metadata /></pywr>'''
    model = pywr.xmlutils.parse_xml(data)
    assert(model.solver.name.lower() == 'cylp')

def test_solver_unrecognised():
    '''Test specifying an unrecognised solver XML'''
    data = '''<pywr><solver name="foobar" /><nodes /><edges /><metadata /></pywr>'''
    with pytest.raises(KeyError):
        model = pywr.xmlutils.parse_xml(data)

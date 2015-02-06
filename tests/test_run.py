#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime
import pytest

import pywr.core
import pywr.xmlutils
import pywr.licenses

def test_run_simple1():
    '''Test the most basic model possible'''
    # parse the XML into a model
    data = file(os.path.join(os.path.dirname(__file__), 'simple1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)

    # run the model
    t0 = model.timestamp
    result = model.step()
    
    # check results
    assert(result[0:3][0:3] == ('optimal', 10.0, 10.0))
    
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
        assert(result[0:3] == ('optimal', 10.0, delivered))

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
        assert(result[0:3] == ('optimal', demand, supply))

def test_run_river1():
    '''Test a river abstraction with a simple catchment'''
    data = file(os.path.join(os.path.dirname(__file__), 'river1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 5.0))

def test_run_river2():
    '''Test a river abstraction with two catchments, a confluence and a split'''
    data = file(os.path.join(os.path.dirname(__file__), 'river2.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result[0:3] == ('optimal', 12.0, 9.25))

def test_run_timeseries1():
    data = file(os.path.join(os.path.dirname(__file__), 'timeseries1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    # check first day initalised
    assert(model.timestamp == datetime.datetime(2015, 1, 1))
    
    # check timeseries has been loaded correctly
    assert(model.data['riverflow1'].value(datetime.datetime(2015, 1, 1)) == 23.92)
    assert(model.data['riverflow1'].value(datetime.datetime(2015, 1, 2)) == 22.14)
    
    # check results
    supplied = []
    for n in range(0, 5):
        result = model.step()
        supplied.append(result[2])
    assert(supplied == [23.0, 22.14, 22.57, 23.0, 23.0])

def test_run_cost1():
    data = file(os.path.join(os.path.dirname(__file__), 'cost1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    nodes = dict([(node.name, node) for node in model.nodes()])
    assert(nodes['supply1'].properties['cost'].value(None) == 1)
    assert(nodes['supply2'].properties['cost'].value(None) == 2) # more expensive
    
    result = model.step()
    # check entire demand was supplied by supply1
    assert(result[0:3] == ('optimal', 10.0, 10.0))
    assert(result[3].items() == [((nodes['supply1'], nodes['demand1']), 10.0)])
    
    # increase demand to more than supply1 can provide on it's own
    # and check that supply2 is used to pick up the slack
    nodes['demand1'].properties['demand'] = pywr.core.Parameter(20.0)
    result = model.step()
    assert(result[0:3] == ('optimal', 20.0, 20.0))
    assert(result[3][(nodes['supply1'], nodes['demand1'])] == 15.0)
    assert(result[3][(nodes['supply2'], nodes['demand1'])] == 5.0)
    
    # supply as much as possible, even if it isn't enough
    nodes['demand1'].properties['demand'] = pywr.core.Parameter(40.0)
    result = model.step()
    assert(result[0:3] == ('optimal', 40.0, 30.0))

def test_run_license():
    data = file(os.path.join(os.path.dirname(__file__), 'simple1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.timestamp = datetime.datetime(2015, 1, 1)
    
    # add licenses to supply node
    nodes = dict([(node.name, node) for node in model.nodes()])
    supply1 = nodes['supply1']
    daily_lic = pywr.licenses.DailyLicense(5)
    annual_lic = pywr.licenses.AnnualLicense(7)
    collection = pywr.licenses.LicenseCollection([daily_lic, annual_lic])
    supply1.licenses = collection
    
    model.check()
    
    # daily license is limit
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 5.0))
    
    # resource state is getting worse
    assert(annual_lic.resource_state(model.timestamp) < 1.0)
    
    # annual license is limit
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 2.0))
    
    # annual license is exhausted
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 0.0))
    assert(annual_lic.resource_state(model.timestamp) == 0.0)

def test_run_bottleneck():
    '''Test max flow constraint on intermediate nodes is upheld'''
    data = file(os.path.join(os.path.dirname(__file__), 'bottleneck.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    result = model.step()
    assert(result[0:3] == ('optimal', 20.0, 15.0))

def test_solver_glpk():
    '''Test specifying the solver in XML'''
    data = '''<pywr><solver name="glpk" /><nodes /><edges /><metadata /></pywr>'''
    model = pywr.xmlutils.parse_xml(data)
    assert(model.solver.name.lower() == 'glpk')

def test_solver_unrecognised():
    '''Test specifying an unrecognised solver XML'''
    data = '''<pywr><solver name="foobar" /><nodes /><edges /><metadata /></pywr>'''
    with pytest.raises(KeyError):
        model = pywr.xmlutils.parse_xml(data)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime
import pytest
import pandas

import pywr.core
import pywr.licenses

from helpers import load_model  

def test_run_simple1():
    '''Test the most basic model possible'''
    # parse the XML into a model
    model = load_model('simple1.xml')

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
    model = load_model('reservoir1.xml')

    for delivered in [10.0, 10.0, 10.0, 5.0, 0.0]:
        result = model.step()
        assert(result[0:3] == ('optimal', 10.0, delivered))

def test_run_reservoir2():
    '''Test a reservoir fed by a river abstraction
    
    The river abstraction should refill the reservoir, but not quickly enough
    to keep up with the demand.
    '''
    model = load_model('reservoir2.xml')
    
    for demand, supply in [(10.0, 10.0), (20.0, 14.0), (26.0, 14.0), (32.0, 14.0), (38.0, 11.0), (41.0, 8.0), (41.0, 8.0)]:
        result = model.step()
        assert(result[0:3] == ('optimal', demand, supply))

def test_run_river1():
    '''Test a river abstraction with a simple catchment'''
    model = load_model('river1.xml')
    
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 5.0))

def test_run_river2():
    '''Test a river abstraction with two catchments, a confluence and a split'''
    model = load_model('river2.xml')
    
    result = model.step()
    assert(result[0:3] == ('optimal', 12.0, 9.25))

def test_run_timeseries1():
    model = load_model('timeseries1.xml')
    
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
    model = load_model('cost1.xml')
    
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    demand1 = model.node['demand1']
    
    assert(supply1.properties['cost'].value(None) == 1)
    assert(supply2.properties['cost'].value(None) == 2) # more expensive
    
    result = model.step()
    # check entire demand was supplied by supply1
    assert(result[0:3] == ('optimal', 10.0, 10.0))
    assert(list(result[3].items()) == [((supply1, demand1), 10.0)])
    
    # increase demand to more than supply1 can provide on it's own
    # and check that supply2 is used to pick up the slack
    demand1.properties['demand'] = pywr.core.ParameterConstant(20.0)
    result = model.step()
    assert(result[0:3] == ('optimal', 20.0, 20.0))
    assert(result[3][(supply1, demand1)] == 15.0)
    assert(result[3][(supply2, demand1)] == 5.0)
    
    # supply as much as possible, even if it isn't enough
    demand1.properties['demand'] = pywr.core.ParameterConstant(40.0)
    result = model.step()
    assert(result[0:3] == ('optimal', 40.0, 30.0))

def test_run_license1():
    model = load_model('simple1.xml')
    
    model.timestamp = datetime.datetime(2015, 1, 1)
    
    # add licenses to supply node
    supply1 = model.node['supply1']
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

def test_run_license2():
    '''Test licenses loaded from XML'''
    model = load_model('license1.xml')
    
    model.timestamp = datetime.datetime(2015, 1, 1)
    
    supply1 = model.node['supply1']
    
    assert(len(supply1.licenses) == 2)
    
    # daily license limit
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 5.0))
    
    # annual license limit
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 2.0))

def test_run_license_group():
    '''Test license groups'''
    model = load_model('groups1.xml')
    
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    
    assert(len(model.group) == 2)
    
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 6.0))

def test_run_bottleneck():
    '''Test max flow constraint on intermediate nodes is upheld'''
    model = load_model('bottleneck.xml')
    result = model.step()
    assert(result[0:3] == ('optimal', 20.0, 15.0))

def test_run_mrf():
    '''Test minimum residual flow constraint'''
    model = load_model('river_mrf1.xml')
    
    # check mrf value was parsed from xml
    river_gauge = model.node['gauge1']
    assert(river_gauge.properties['mrf'].value(model.timestamp) == 10.5)
    
    # test model result
    data = {
        None: 12.0,
        100: 0.0,
        10.5: 8.5,
        11.75: 7.25,
        0.0: 12.0,
    }
    for mrf_value, expected_supply in data.items():
        river_gauge.properties['mrf'] = pywr.core.ParameterConstant(mrf_value)
        result = model.step()
        assert(result[0:3] == ('optimal', 12.0, expected_supply))

def test_run_discharge_upstream():
    '''Test river with inline discharge (upstream)
    
    In this instance the discharge is upstream of the abstraction, and so can
    be abstracted in the same way as the water from the catchment
    '''
    model = load_model('river_discharge1.xml')
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 8.0))

def test_run_discharge_downstream():
    '''Test river with inline discharge (downstream)
    
    In this instance the discharge is downstream of the abstraction, so the
    water shouldn't be available.
    '''
    model = load_model('river_discharge2.xml')
    result = model.step()
    assert(result[0:3] == ('optimal', 10.0, 5.0))

def test_run_blender1():
    '''Test blender constraint/component'''
    model = load_model('blender1.xml')

    blender = model.node['blender1']
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    supply3 = model.node['supply3']

    # check blender ratio
    assert(blender.properties['ratio'].value(model.timestamp) == 0.75)

    # check supplies have been connected correctly
    assert(len(blender.slots) == 2)
    assert(blender.slots[1] is supply1)
    assert(blender.slots[2] is supply2)

    # test model results
    result = model.step()
    assert(result[3][(supply1, blender)] == 7.5)
    assert(result[3][(supply2, blender)] == 2.5)

def test_run_blender2():
    '''Test blender constraint/component'''
    model = load_model('blender2.xml')

    blender = model.node['blender1']
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    
    # test model results
    result = model.step()
    assert(result[3][(supply1, blender)] == 3.0)
    assert(result[3][(supply2, blender)] == 7.0)

def test_run():
    model = load_model('simple1.xml')
    
    # run model from start to finish
    timesteps = model.run()
    assert(timesteps == 365)
    
    # try to run finished model
    timesteps = model.run()
    assert(timesteps is None)
    
    # reset model and run again
    model.reset()
    timesteps = model.run()
    assert(timesteps == 365)
    
    # run remaining timesteps
    model.timestamp = pandas.to_datetime('2015-12-01')
    timesteps = model.run()
    assert(timesteps == 31)

def test_run_until_failure():
    model = load_model('simple1.xml')
    
    # run until failure
    model.timestamp = pandas.to_datetime('2015-12-01')
    demand1 = model.node['demand1']
    def demand_func(node, timestamp):
        return timestamp.day
    demand1.properties['demand'] = pywr.core.ParameterFunction(demand1, demand_func)
    timesteps = model.run(until_failure=True)
    assert(timesteps == 16)

def test_run_until_date():
    model = load_model('simple1.xml')
    
    # run until date
    model.reset()
    timesteps = model.run(until_date=pandas.to_datetime('2015-01-20'))
    assert(timesteps == 20)

def test_solver_glpk():
    '''Test specifying the solver in XML'''
    data = '''<pywr><solver name="glpk" /><nodes /><edges /><metadata /></pywr>'''
    model = load_model(data=data)
    assert(model.solver.name.lower() == 'glpk')

def test_solver_unrecognised():
    '''Test specifying an unrecognised solver XML'''
    data = '''<pywr><solver name="foobar" /><nodes /><edges /><metadata /></pywr>'''
    with pytest.raises(KeyError):
        model = load_model(data=data)

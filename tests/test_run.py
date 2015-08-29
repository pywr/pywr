#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime
import pytest
import pandas

import pywr.core
import pywr.licenses
import pywr.domains.river

from helpers import load_model

def test_run_simple1(solver):
    '''Test the most basic model possible'''
    # parse the XML into a model
    model = load_model('simple1.xml', solver=solver)

    # run the model
    t0 = model.timestepper.current
    result = model.step()

    # check results
    demand1 = model.node['demand1']
    assert(demand1.flow == 10.0)

    # check the timestamp incremented
    assert(model.timestepper.current - t0 == datetime.timedelta(1))


def test_run_reservoir1(solver):
    '''Test a reservoir with no refill

    Without an additional supply the reservoir should empty and cause a failure.
    '''
    model = load_model('reservoir1.xml', solver=solver)
    demand1 = model.node['demand1']
    supply1 = model.node['supply1']
    for demand, stored in [(10.0, 25.0), (10.0, 15.0), (10.0, 5.0), (5.0, 0.0), (0.0, 0.0)]:
        result = model.step()
        assert(demand1.flow == demand)
        assert(supply1.volume == stored)


def test_run_reservoir2(solver):
    '''Test a reservoir fed by a river abstraction

    The river abstraction should refill the reservoir, but not quickly enough
    to keep up with the demand.
    '''
    model = load_model('reservoir2.xml', solver=solver)

    demand1 = model.node['demand1']
    supply1 = model.node['supply1']
    for demand, stored in [(15.0, 25.0), (15.0, 15.0), (15.0, 5.0), (10.0, 0.0), (5.0, 0.0)]:
        result = model.step()
        assert(demand1.flow == demand)
        assert(supply1.volume == stored)


def test_run_river1(solver):
    '''Test a river abstraction with a simple catchment'''
    model = load_model('river1.xml', solver=solver)

    result = model.step()
    demand1 = model.node['demand1']
    assert(demand1.flow == 5.0)


# Contains a RiverSplit which needs addressing
@pytest.mark.xfail
def test_run_river2(solver):
    '''Test a river abstraction with two catchments, a confluence and a split'''
    model = load_model('river2.xml', solver=solver)

    result = model.step()
    assert(result[0:3] == ('optimal', 12.0, 9.25))
    assert(model.failure)

def test_run_timeseries1(solver):
    model = load_model('timeseries1.xml', solver=solver)

    # check first day initalised
    assert(model.timestepper.start == datetime.datetime(2015, 1, 1))

    # check timeseries has been loaded correctly
    from pywr._core import Timestep
    assert(model.data['riverflow1'].value(Timestep(datetime.datetime(2015, 1, 1), 0, 0)) == 23.92)
    assert(model.data['riverflow1'].value(Timestep(datetime.datetime(2015, 1, 2), 1, 0)) == 22.14)

    # check results
    demand1 = model.node['demand1']
    supplied = []
    for n in range(0, 5):
        result = model.step()
        print(model.timestep.datetime)
        supplied.append(demand1.flow)
    assert(supplied == [23.0, 22.14, 22.57, 23.0, 23.0])

def test_run_cost1(solver):
    model = load_model('cost1.xml', solver=solver)

    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    demand1 = model.node['demand1']

    assert(supply1.get_cost(None) == 1)
    assert(supply2.get_cost(None) == 2)  # more expensive

    result = model.step()
    # check entire demand was supplied by supply1
    assert(supply1.flow == 10.0)
    assert(supply2.flow == 0.0)
    assert(demand1.flow == 10.0)

    # increase demand to more than supply1 can provide on it's own
    # and check that supply2 is used to pick up the slack
    demand1.max_flow = 20.0
    result = model.step()
    assert(supply1.flow == 15.0)
    assert(supply2.flow == 5.0)
    assert(demand1.flow == 20.0)

    # supply as much as possible, even if it isn't enough
    demand1.max_flow = 40.0
    result = model.step()
    assert(supply1.flow == 15.0)
    assert(supply2.flow == 15.0)
    assert(demand1.flow == 30.0)


def test_run_license1(solver):
    model = load_model('simple1.xml', solver=solver)

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
    d1 = model.node['demand1']
    assert(d1.flow == 5.0)

    # resource state is getting worse
    assert(annual_lic.resource_state(model.timestep) < 1.0)

    # annual license is limit
    result = model.step()
    d1 = model.node['demand1']
    assert(d1.flow == 2.0)

    # annual license is exhausted
    result = model.step()
    d1 = model.node['demand1']
    assert(d1.flow == 0.0)
    assert(annual_lic.resource_state(model.timestep) == 0.0)


def test_run_license2(solver):
    '''Test licenses loaded from XML'''
    model = load_model('license1.xml', solver=solver)

    model.timestamp = datetime.datetime(2015, 1, 1)

    supply1 = model.node['supply1']

    assert(len(supply1.licenses) == 2)

    # daily license limit
    result = model.step()
    d1 = model.node['demand1']
    assert(d1.flow == 5.0)

    # annual license limit
    result = model.step()
    assert(d1.flow == 2.0)


def test_run_license_group(solver):
    '''Test license groups'''
    model = load_model('groups1.xml', solver=solver)

    supply1 = model.node['supply1']
    supply2 = model.node['supply2']

    assert(len(model.group) == 2)

    result = model.step()
    d1 = model.node['demand1']
    assert(d1.flow == 6.0)


def test_run_bottleneck(solver):
    '''Test max flow constraint on intermediate nodes is upheld'''
    model = load_model('bottleneck.xml', solver=solver)
    result = model.step()
    d1 = model.node['demand1']
    d2 = model.node['demand2']
    assert(d1.flow+d2.flow == 15.0)


@pytest.mark.xfail
def test_run_mrf(solver):
    '''Test minimum residual flow constraint'''
    model = load_model('river_mrf1.xml', solver=solver)

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

def test_run_discharge_upstream(solver):
    '''Test river with inline discharge (upstream)

    In this instance the discharge is upstream of the abstraction, and so can
    be abstracted in the same way as the water from the catchment
    '''
    model = load_model('river_discharge1.xml', solver=solver)
    model.step()
    demand = model.node['demand1']
    term = model.node['term1']
    assert(demand.flow == 8.0)
    assert(term.flow == 0.0)

def test_run_discharge_downstream(solver):
    '''Test river with inline discharge (downstream)

    In this instance the discharge is downstream of the abstraction, so the
    water shouldn't be available.
    '''
    model = load_model('river_discharge2.xml', solver=solver)
    model.step()
    demand = model.node['demand1']
    term = model.node['term1']
    assert(demand.flow == 5.0)
    assert(term.flow == 3.0)


@pytest.mark.xfail
def test_run_blender1(solver):
    '''Test blender constraint/component'''
    model = load_model('blender1.xml', solver=solver)

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

@pytest.mark.xfail
def test_run_blender2(solver):
    '''Test blender constraint/component'''
    model = load_model('blender2.xml', solver=solver)

    blender = model.node['blender1']
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']

    # test model results
    result = model.step()
    assert(result[3][(supply1, blender)] == 3.0)
    assert(result[3][(supply2, blender)] == 7.0)

@pytest.mark.xfail
def test_run_demand_discharge(solver):
    """Test demand discharge node"""
    model = pywr.core.Model(solver=solver)
    catchment = pywr.core.Catchment(model, 'catchment', flow=10.0)
    abstraction1 = pywr.core.RiverAbstraction(model, 'abstraction1', max_flow=100)
    demand1 = pywr.core.Demand(model, 'demand1', demand=8.0)
    discharge = pywr.core.DemandDischarge(model, 'discharge')
    abstraction2 = pywr.core.RiverAbstraction(model, 'abstraction2', max_flow=100)
    demand2 = pywr.core.Demand(model, 'demand2', demand=5.0)
    term = pywr.core.Terminator(model, 'term')
    catchment.connect(abstraction1)
    abstraction1.connect(demand1)
    abstraction1.connect(discharge)
    demand1.connect(discharge)
    discharge.connect(abstraction2)
    abstraction2.connect(demand2)
    abstraction2.connect(term)

    # when consumption is 100% there is not enough water
    # 8 + 5 > 10
    demand1.properties['consumption'] = pywr.core.ParameterConstant(1.0)
    result = model.step()
    assert(model.failure)

    # when demand #1 consumes 90% of it's supply there still isn't enough
    demand1.properties['consumption'] = pywr.core.ParameterConstant(0.9)
    result = model.step()
    assert(model.failure)

    # when demand #1 only consumes 50% of it's supply there is enough for all
    demand1.properties['consumption'] = pywr.core.ParameterConstant(0.5)
    result = model.step()
    assert(not model.failure)

def test_reset(solver):
    """Test model reset"""
    model = load_model('license1.xml', solver=solver)
    supply1 = model.node['supply1']
    license_collection = supply1.licenses
    license = [lic for lic in license_collection._licenses if isinstance(lic, pywr.licenses.AnnualLicense)][0]
    assert(license.available(None) == 7.0)
    model.step()
    assert(license.available(None) == 2.0)
    model.reset()
    assert(license.available(None) == 7.0)


def test_run(solver):
    model = load_model('simple1.xml', solver=solver)

    # run model from start to finish
    timestep = model.run()
    assert(timestep.index == 364)

    # try to run finished model
    timestep = model.run()
    assert(timestep is None)

    # reset model and run again
    model.reset()
    timestep = model.run()
    assert(timestep.index == 364)

    # run remaining timesteps
    model.reset(start=pandas.to_datetime('2015-12-01'))
    timestep = model.run()
    assert(timestep.index == 364)


@pytest.mark.xfail
def test_run_until_failure(solver):
    model = load_model('simple1.xml', solver=solver)

    # run until failure
    model.timestamp = pandas.to_datetime('2015-12-01')
    demand1 = model.node['demand1']
    def demand_func(node, timestamp):
        return timestamp.datetime.day
    demand1.min_flow = pywr.core.ParameterFunction(demand1, demand_func)
    timesteps = model.run(until_failure=True)
    assert(model.failure)
    assert(timesteps == 16)


def test_run_until_date(solver):
    model = load_model('simple1.xml', solver=solver)

    # run until date
    model.reset()
    timestep = model.run(until_date=pandas.to_datetime('2015-01-20'))
    assert(timestep.index == 20)


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

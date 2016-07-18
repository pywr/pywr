#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime
import pytest
import pandas
from numpy.testing import assert_allclose

import pywr.core
from pywr.core import Model, Storage, Input, Output, Link
import pywr.solvers
import pywr.parameters.licenses
import pywr.domains.river

from helpers import load_model

import pywr.parameters


def test_run_simple1(solver):
    '''Test the most basic model possible'''
    # parse the JSON into a model
    model = load_model('simple1.json', solver=None)

    # run the model
    t0 = model.timestepper.current
    model.step()

    # check results
    demand1 = model.node['demand1']
    assert_allclose(demand1.flow, 10.0, atol=1e-7)
    # initially the timestepper returns the first time-step, so timestepper.current
    # does not change after the first 'step'.
    assert(model.timestepper.current.datetime - t0.datetime == datetime.timedelta(0))
    # check the timestamp incremented
    model.step()
    assert(model.timestepper.current.datetime - t0.datetime == datetime.timedelta(1))


def test_run_reservoir1(solver):
    '''Test a reservoir with no refill

    Without an additional supply the reservoir should empty and cause a failure.
    '''
    model = load_model('reservoir1.json', solver=solver)
    demand1 = model.node['demand1']
    supply1 = model.node['supply1']
    for demand, stored in [(10.0, 25.0), (10.0, 15.0), (10.0, 5.0), (5.0, 0.0), (0.0, 0.0)]:
        result = model.step()
        assert_allclose(demand1.flow, demand, atol=1e-7)
        assert_allclose(supply1.volume, stored, atol=1e-7)


def test_run_reservoir2(solver):
    '''Test a reservoir fed by a river abstraction

    The river abstraction should refill the reservoir, but not quickly enough
    to keep up with the demand.
    '''
    model = load_model('reservoir2.json', solver=solver)

    demand1 = model.node['demand1']
    supply1 = model.node['supply1']
    catchment = model.node['catchment1']
    assert(catchment.min_flow == 5)
    for demand, stored in [(15.0, 25.0), (15.0, 15.0), (15.0, 5.0), (10.0, 0.0), (5.0, 0.0)]:
        result = model.step()
        assert_allclose(demand1.flow[0], demand, atol=1e-7)
        assert_allclose(supply1.volume[0], stored, atol=1e-7)

def test_empty_storage_min_flow(solver):

    model = Model(solver=solver)
    storage = Storage(model, "storage", initial_volume=100, max_volume=100, num_inputs=1, num_outputs=0)
    otpt = Output(model, "output", min_flow=75)
    storage.connect(otpt)
    model.check()
    model.step()
    with pytest.raises(RuntimeError):
        model.step()

def test_run_river1(solver):
    '''Test a river abstraction with a simple catchment'''
    model = load_model('river1.json', solver=solver)

    result = model.step()
    demand1 = model.node['demand1']
    assert_allclose(demand1.flow, 5.0, atol=1e-7)


def test_run_river2(solver):
    '''Test a river abstraction with two catchments, a confluence and a split'''
    model = load_model('river2.json', solver=solver)

    model.step()

    demand1 = model.node['demand1']
    assert_allclose(demand1.flow, 7.25, atol=1e-7)
    demand2 = model.node['demand2']
    assert_allclose(demand2.flow, 2.0, atol=1e-7)


# Contains an out of range date for pandas.to_datetime
def test_run_timeseries1(solver):
    model = load_model('timeseries1.json', solver=solver)

    # check first day initalised
    assert(model.timestepper.start == datetime.datetime(2015, 1, 1))

    # check results
    demand1 = model.node['demand1']
    catchment1 = model.node['catchment1']
    for expected in (23.92, 22.14, 22.57, 24.97, 27.59):
        result = model.step()
        assert_allclose(catchment1.flow, expected, atol=1e-7)
        assert_allclose(demand1.flow, min(expected, 23.0), atol=1e-7)

def test_run_cost1(solver):
    model = load_model('cost1.json', solver=solver)

    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    demand1 = model.node['demand1']

    assert_allclose(supply1.get_cost(None, None), 1)
    assert_allclose(supply2.get_cost(None, None), 2)  # more expensive

    result = model.step()
    # check entire demand was supplied by supply1
    assert_allclose(supply1.flow, 10.0, atol=1e-7)
    assert_allclose(supply2.flow, 0.0, atol=1e-7)
    assert_allclose(demand1.flow, 10.0, atol=1e-7)

    # increase demand to more than supply1 can provide on it's own
    # and check that supply2 is used to pick up the slack
    demand1.max_flow = 20.0
    result = model.step()
    assert_allclose(supply1.flow, 15.0, atol=1e-7)
    assert_allclose(supply2.flow, 5.0, atol=1e-7)
    assert_allclose(demand1.flow, 20.0, atol=1e-7)

    # supply as much as possible, even if it isn't enough
    demand1.max_flow = 40.0
    result = model.step()
    assert_allclose(supply1.flow, 15.0, atol=1e-7)
    assert_allclose(supply2.flow, 15.0, atol=1e-7)
    assert_allclose(demand1.flow, 30.0, atol=1e-7)

# Licence XML needs addressing
@pytest.mark.xfail
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
    assert_allclose(d1.flow, 5.0, atol=1e-7)

    # resource state is getting worse
    assert(annual_lic.resource_state(model.timestep) < 1.0)

    # annual license is limit
    result = model.step()
    d1 = model.node['demand1']
    assert_allclose(d1.flow, 2.0, atol=1e-7)

    # annual license is exhausted
    result = model.step()
    d1 = model.node['demand1']
    assert_allclose(d1.flow, 0.0, atol=1e-7)
    assert_allclose(annual_lic.resource_state(model.timestep), 0.0, atol=1e-7)

# Licence XML needs addressing
@pytest.mark.xfail
def test_run_license2(solver):
    '''Test licenses loaded from XML'''
    model = load_model('license1.xml', solver=solver)

    model.timestamp = datetime.datetime(2015, 1, 1)

    supply1 = model.node['supply1']

    assert(len(supply1.licenses) == 2)

    # daily license limit
    result = model.step()
    d1 = model.node['demand1']
    assert_allclose(d1.flow, 5.0, atol=1e-7)

    # annual license limit
    result = model.step()
    assert_allclose(d1.flow, 2.0, atol=1e-7)


@pytest.mark.xfail
def test_run_license_group(solver):
    '''Test license groups'''
    model = load_model('groups1.xml', solver=solver)

    supply1 = model.node['supply1']
    supply2 = model.node['supply2']

    assert(len(model.group) == 2)

    result = model.step()
    d1 = model.node['demand1']
    assert_allclose(d1.flow, 6.0, atol=1e-7)


def test_run_bottleneck(solver):
    '''Test max flow constraint on intermediate nodes is upheld'''
    model = load_model('bottleneck.json', solver=solver)
    result = model.step()
    d1 = model.node['demand1']
    d2 = model.node['demand2']
    assert_allclose(d1.flow+d2.flow, 15.0, atol=1e-7)

def test_run_discharge_upstream(solver):
    '''Test river with inline discharge (upstream)

    In this instance the discharge is upstream of the abstraction, and so can
    be abstracted in the same way as the water from the catchment
    '''
    model = load_model('river_discharge1.json', solver=solver)
    model.step()
    demand = model.node['demand1']
    term = model.node['term1']
    assert_allclose(demand.flow, 8.0, atol=1e-7)
    assert_allclose(term.flow, 0.0, atol=1e-7)

def test_run_discharge_downstream(solver):
    '''Test river with inline discharge (downstream)

    In this instance the discharge is downstream of the abstraction, so the
    water shouldn't be available.
    '''
    model = load_model('river_discharge2.json', solver=solver)
    model.step()
    demand = model.node['demand1']
    term = model.node['term1']
    assert_allclose(demand.flow, 5.0, atol=1e-7)
    assert_allclose(term.flow, 3.0, atol=1e-7)


@pytest.mark.xfail
def test_run_blender1(solver):
    '''Test blender constraint/component'''
    model = load_model('blender1.xml', solver=solver)

    blender = model.node['blender1']
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']
    supply3 = model.node['supply3']

    # check blender ratio
    assert_allclose(blender.properties['ratio'].value(model.timestamp), 0.75, atol=1e-7)

    # check supplies have been connected correctly
    assert(len(blender.slots) == 2)
    assert(blender.slots[1] is supply1)
    assert(blender.slots[2] is supply2)

    # test model results
    result = model.step()
    assert_allclose(result[3][(supply1, blender)], 7.5, atol=1e-7)
    assert_allclose(result[3][(supply2, blender)], 2.5, atol=1e-7)

@pytest.mark.xfail
def test_run_blender2(solver):
    '''Test blender constraint/component'''
    model = load_model('blender2.xml', solver=solver)

    blender = model.node['blender1']
    supply1 = model.node['supply1']
    supply2 = model.node['supply2']

    # test model results
    result = model.step()
    assert_allclose(result[3][(supply1, blender)], 3.0, atol=1e-7)
    assert_allclose(result[3][(supply2, blender)], 7.0, atol=1e-7)

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
    demand1.properties['consumption'] = pywr.parameters.ParameterConstant(1.0)
    result = model.step()
    assert(model.failure)

    # when demand #1 consumes 90% of it's supply there still isn't enough
    demand1.properties['consumption'] = pywr.parameters.ParameterConstant(0.9)
    result = model.step()
    assert(model.failure)

    # when demand #1 only consumes 50% of it's supply there is enough for all
    demand1.properties['consumption'] = pywr.parameters.ParameterConstant(0.5)
    result = model.step()
    assert(not model.failure)

def test_new_storage(solver):
    """Test new-style storage node with multiple inputs"""
    model = pywr.core.Model(
        start=pandas.to_datetime('1888-01-01'),
        end=pandas.to_datetime('1888-01-01'),
        timestep=datetime.timedelta(1),
        solver=solver
    )

    supply1 = pywr.core.Input(model, 'supply1')

    splitter = pywr.core.Storage(model, 'splitter', num_outputs=1, num_inputs=2, max_volume=10, volume=5)

    demand1 = pywr.core.Output(model, 'demand1')
    demand2 = pywr.core.Output(model, 'demand2')

    supply1.connect(splitter)

    splitter.connect(demand1, from_slot=0)
    splitter.connect(demand2, from_slot=1)

    supply1.max_flow = 45.0
    demand1.max_flow = 20
    demand2.max_flow = 40

    demand1.cost = -150
    demand2.cost = -100

    model.run()

    assert_allclose(supply1.flow, [45], atol=1e-7)
    assert_allclose(splitter.volume, [0], atol=1e-7)  # New volume is zero
    assert_allclose(demand1.flow, [20], atol=1e-7)
    assert_allclose(demand2.flow, [30], atol=1e-7)


def test_storage_spill_compensation(solver):
    """Test storage spill and compensation flows

    The upstream catchment has min_flow == max_flow, so it "pushes" water into
    the reservoir. The reservoir is already at it's maximum volume, so the
    water must go *somewhere*. The compensation flow has the most negative cost,
    so that is satisfied first. Once that is full, the demand is supplied.
    Finally, any surplus is forced into the spill despite the cost.

    Catchment -> Reservoir -> Demand
                         |--> Spill        --|
                         |--> Compensation --|
                                             |--> Terminator
    """
    model = pywr.core.Model(solver=solver)

    catchment = pywr.core.Input(model, name="Input", min_flow=10.0, max_flow=10.0, cost=1)
    reservoir = pywr.core.Storage(model, name="Storage", max_volume=100, volume=100.0)
    spill = pywr.core.Link(model, name="Spill", cost=1.0)
    compensation = pywr.core.Link(model, name="Compensation", max_flow=3.0, cost=-999)
    terminator = pywr.core.Output(model, name="Terminator", cost=-1.0)
    demand = pywr.core.Output(model, name="Demand", max_flow=5.0, cost=-500)

    catchment.connect(reservoir)
    reservoir.connect(spill)
    reservoir.connect(compensation)
    reservoir.connect(demand)
    spill.connect(terminator)
    compensation.connect(terminator)

    model.check()
    model.run()
    assert_allclose(catchment.flow[0], 10.0, atol=1e-7)
    assert_allclose(demand.flow[0], 5.0, atol=1e-7)
    assert_allclose(compensation.flow[0], 3.0, atol=1e-7)
    assert_allclose(spill.flow[0], 2.0, atol=1e-7)
    assert_allclose(terminator.flow[0], (compensation.flow[0] + spill.flow[0]), atol=1e-7)


def test_reservoir_circle(solver):
    """
    Issue #140. A model with a circular route, from a reservoir Input back
    around to it's own Output.
    
                 Demand
                    ^
                    |
                Reservoir <- Pumping
                    |           ^ 
                    v           |
              Compensation      |
                    |           |
                    v           |
    Catchment -> River 1 -> River 2 ----> MRFA -> Waste
                                    |              ^
                                    |---> MRFB ----|
    """
    model = Model(solver=solver)

    catchment = Input(model, "catchment", max_flow=500, min_flow=500)
    
    reservoir = Storage(model, "reservoir", max_volume=10000, initial_volume=5000)
    
    demand = Output(model, "demand", max_flow=50, cost=-100)
    pumping_station = Link(model, "pumping station", max_flow=100, cost=-10)
    river1 = Link(model, "river1")
    river2 = Link(model, "river2")
    compensation = Link(model, "compensation", cost=600)
    mrfA = Link(model, "mrfA", cost=-500, max_flow=50)
    mrfB = Link(model, "mrfB")
    waste = Output(model, "waste")

    catchment.connect(river1)
    river1.connect(river2)
    river2.connect(mrfA)
    river2.connect(mrfB)
    mrfA.connect(waste)
    mrfB.connect(waste)
    river2.connect(pumping_station)
    pumping_station.connect(reservoir)
    reservoir.connect(compensation)
    compensation.connect(river1)
    reservoir.connect(demand)

    model.check()
    model.setup()
    
    # not limited by mrf, pump capacity is constraint
    model.step()
    assert_allclose(catchment.flow, 500)
    assert_allclose(waste.flow, 400)
    assert_allclose(compensation.flow, 0)
    assert_allclose(pumping_station.flow, 100)
    assert_allclose(demand.flow, 50)
    
    # limited by mrf
    catchment.min_flow = catchment.max_flow = 100
    model.step()
    assert_allclose(waste.flow, 50)
    assert_allclose(compensation.flow, 0)
    assert_allclose(pumping_station.flow, 50)
    assert_allclose(demand.flow, 50)
    
    # reservoir can support mrf, but doesn't need to
    compensation.cost = 200
    model.step()
    assert_allclose(waste.flow, 50)
    assert_allclose(compensation.flow, 0)
    assert_allclose(pumping_station.flow, 50)
    assert_allclose(demand.flow, 50)
    
    # reservoir supporting mrf
    catchment.min_flow = catchment.max_flow = 0
    model.step()
    assert_allclose(waste.flow, 50)
    assert_allclose(compensation.flow, 50)
    assert_allclose(pumping_station.flow, 0)
    assert_allclose(demand.flow, 50)


# Licence XML needs addressing
@pytest.mark.xfail
def test_reset(solver):
    """Test model reset"""
    model = load_model('license1.xml', solver=solver)
    supply1 = model.node['supply1']
    license_collection = supply1.licenses
    license = [lic for lic in license_collection._licenses if isinstance(lic, pywr.licenses.AnnualLicense)][0]
    assert_allclose(license.available(None), 7.0, atol=1e-7)
    model.step()
    assert_allclose(license.available(None), 2.0, atol=1e-7)
    model.reset()
    assert_allclose(license.available(None), 7.0, atol=1e-7)


def test_run(solver):
    model = load_model('simple1.json', solver=solver)

    # run model from start to finish
    timestep = model.run()
    assert(timestep.index == 364)

    # try to run finished model
    timestep = model.run(reset=False)
    assert(timestep is None)

    # reset model and run again
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
    demand1.min_flow = pywr.parameters.ParameterFunction(demand1, demand_func)
    timesteps = model.run(until_failure=True)
    assert(model.failure)
    assert(timesteps == 16)


def test_run_until_date(solver):
    model = load_model('simple1.json', solver=solver)

    # run until date
    timestep = model.run(until_date=pandas.to_datetime('2015-01-20'))
    assert(timestep.index == 20)


def test_select_solver():
    """Test specifying the solver in XML"""
    solver_names = [solver.name for solver in pywr.solvers.solver_registry]
    for solver_name in solver_names:
        data = '''{"metadata": {}, "nodes": {}, "edges": {}, "timestepper": {"start": "1990-01-01","end": "1999-12-31","timestep": 1}, "solver": {"name": "%s"}}''' % solver_name
        model = load_model(data=data)
        assert(model.solver.name.lower() == solver_name)


def test_solver_unrecognised():
    '''Test specifying an unrecognised solver XML'''
    solver_name = 'foobar'
    data = '''{"metadata": {}, "nodes": {}, "edges": {}, "timestepper": {"start": "1990-01-01","end": "1999-12-31","timestep": 1}, "solver": {"name": "%s"}}''' % solver_name
    with pytest.raises(KeyError):
        model = load_model(data=data)

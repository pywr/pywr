#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import pytest
import pandas
from numpy.testing import assert_allclose

import pywr.core
from pywr.model import Model, ModelStructureError, ModelResult
from pywr.nodes import Storage, Input, Output, Link
import pywr.solvers
import pywr.parameters.licenses
import pywr.domains.river

from helpers import load_model

import pywr.parameters


def test_run_simple1():
    '''Test the most basic model possible'''
    # parse the JSON into a model
    model = load_model('simple1.json')

    # run the model
    t0 = model.timestepper.start.to_pydatetime()
    model.step()

    # check results
    demand1 = model.nodes['demand1']
    assert_allclose(demand1.flow, 10.0, atol=1e-7)
    # initially the timestepper returns the first time-step, so timestepper.current
    # does not change after the first 'step'.
    assert(model.timestepper.current.datetime - t0 == datetime.timedelta(0))
    # check the timestamp incremented
    model.step()
    assert(model.timestepper.current.datetime - t0 == datetime.timedelta(1))

def test_model_results():
    '''Test model results object'''
    import pywr

    model = load_model('simple1.json')
    res = model.run()
    assert (isinstance(res, ModelResult))
    assert (res.timesteps == 365)
    assert (res.version == pywr.__version__)
    assert res.solver_stats['number_of_cols']
    assert res.solver_stats['number_of_rows']
    assert res.solver_name == model.solver.name
    print(res)
    print(res._repr_html_())


@pytest.mark.parametrize("json_file", ['reservoir1.json', 'reservoir1_pc.json'])
def test_run_reservoir1(json_file):
    '''Test a reservoir with no refill

    Without an additional supply the reservoir should empty and cause a failure.
    '''
    model = load_model(json_file)
    demand1 = model.nodes['demand1']
    supply1 = model.nodes['supply1']
    for demand, stored in [(10.0, 25.0), (10.0, 15.0), (10.0, 5.0), (5.0, 0.0), (0.0, 0.0)]:
        result = model.step()
        assert_allclose(demand1.flow, demand, atol=1e-7)
        assert_allclose(supply1.volume, stored, atol=1e-7)


def test_run_reservoir2():
    '''Test a reservoir fed by a river abstraction

    The river abstraction should refill the reservoir, but not quickly enough
    to keep up with the demand.
    '''
    model = load_model('reservoir2.json')

    demand1 = model.nodes['demand1']
    supply1 = model.nodes['supply1']
    catchment = model.nodes['catchment1']
    assert(catchment.min_flow == 5)
    for demand, stored in [(15.0, 25.0), (15.0, 15.0), (15.0, 5.0), (10.0, 0.0), (5.0, 0.0)]:
        result = model.step()
        assert_allclose(demand1.flow[0], demand, atol=1e-7)
        assert_allclose(supply1.volume[0], stored, atol=1e-7)

def test_empty_storage_min_flow():

    model = Model()
    storage = Storage(model, "storage", initial_volume=100, max_volume=100, num_inputs=1, num_outputs=0)
    otpt = Output(model, "output", min_flow=75)
    storage.connect(otpt)
    model.check()
    model.step()
    with pytest.raises(RuntimeError):
        model.step()

def test_run_river1():
    '''Test a river abstraction with a simple catchment'''
    model = load_model('river1.json')

    result = model.step()
    demand1 = model.nodes['demand1']
    assert_allclose(demand1.flow, 5.0, atol=1e-7)


def test_run_river2():
    '''Test a river abstraction with two catchments, a confluence and a split'''
    model = load_model('river2.json')

    model.step()

    demand1 = model.nodes['demand1']
    assert_allclose(demand1.flow, 7.25, atol=1e-7)
    demand2 = model.nodes['demand2']
    assert_allclose(demand2.flow, 2.0, atol=1e-7)


# Contains an out of range date for pandas.to_datetime
@pytest.mark.parametrize("json_file", ['timeseries1.json', 'timeseries1_xlsx.json'])
def test_run_timeseries1(json_file):
    model = load_model(json_file)

    # check first day initalised
    assert(model.timestepper.start == datetime.datetime(2015, 1, 1))

    # check results
    demand1 = model.nodes['demand1']
    catchment1 = model.nodes['catchment1']
    for expected in (23.92, 22.14, 22.57, 24.97, 27.59):
        result = model.step()
        assert_allclose(catchment1.flow, expected, atol=1e-7)
        assert_allclose(demand1.flow, min(expected, 23.0), atol=1e-7)


# Contains an out of range date for pandas.to_datetime
@pytest.mark.parametrize("json_file", ['timeseries1_weekly.json', 'timeseries1_weekly_hdf.json'])
def test_run_timeseries1_weekly(json_file):
    model = load_model(json_file)

    # check first day initalised
    assert(model.timestepper.start == datetime.datetime(2015, 1, 1))

    # check results
    demand1 = model.nodes['demand1']
    catchment1 = model.nodes['catchment1']
    for expected in (23.92, 25.67, 28.24, 25.28, 21.84):
        result = model.step()
        assert_allclose(catchment1.flow, expected, atol=1e-7)
        assert_allclose(demand1.flow, min(expected, 23.0), atol=1e-7)


def test_run_cost1():
    model = load_model('cost1.json')

    supply1 = model.nodes['supply1']
    supply2 = model.nodes['supply2']
    demand1 = model.nodes['demand1']

    assert_allclose(supply1.get_cost(None), 1)
    assert_allclose(supply2.get_cost(None), 2)  # more expensive

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


def test_run_bottleneck():
    '''Test max flow constraint on intermediate nodes is upheld'''
    model = load_model('bottleneck.json')
    result = model.step()
    d1 = model.nodes['demand1']
    d2 = model.nodes['demand2']
    assert_allclose(d1.flow+d2.flow, 15.0, atol=1e-7)

@pytest.mark.skipif(Model().solver.name == "glpk-edge", reason="Not valid for GLPK Edge based solver.")
def test_run_discharge_upstream():
    '''Test river with inline discharge (upstream)

    In this instance the discharge is upstream of the abstraction, and so can
    be abstracted in the same way as the water from the catchment
    '''
    model = load_model('river_discharge1.json')
    model.step()
    demand = model.nodes['demand1']
    term = model.nodes['term1']
    assert_allclose(demand.flow, 8.0, atol=1e-7)
    assert_allclose(term.flow, 0.0, atol=1e-7)

@pytest.mark.skipif(Model().solver.name == "glpk-edge", reason="Not valid for GLPK Edge based solver.")
def test_run_discharge_downstream():
    '''Test river with inline discharge (downstream)

    In this instance the discharge is downstream of the abstraction, so the
    water shouldn't be available.
    '''
    model = load_model('river_discharge2.json')
    model.step()
    demand = model.nodes['demand1']
    term = model.nodes['term1']
    assert_allclose(demand.flow, 5.0, atol=1e-7)
    assert_allclose(term.flow, 3.0, atol=1e-7)

def test_new_storage():
    """Test new-style storage node with multiple inputs"""
    model = pywr.core.Model(
        start=pandas.to_datetime('1888-01-01'),
        end=pandas.to_datetime('1888-01-01'),
        timestep=datetime.timedelta(1)
    )

    supply1 = pywr.core.Input(model, 'supply1')

    splitter = pywr.core.Storage(model, 'splitter', num_outputs=1, num_inputs=2, max_volume=10, initial_volume=5)

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


def test_virtual_storage():
    """ Test the VirtualStorage node """

    model = pywr.core.Model()

    inpt = Input(model, "Input", max_flow=20)
    lnk = Link(model, "Link")
    inpt.connect(lnk)
    otpt = Output(model, "Output", max_flow=10, cost=-10.0)
    lnk.connect(otpt)

    vs = pywr.core.VirtualStorage(model, "Licence", [lnk], initial_volume=10.0, max_volume=10.0)

    model.setup()

    assert_allclose(vs.volume, [10], atol=1e-7)

    model.step()

    assert_allclose(otpt.flow, [10], atol=1e-7)
    assert_allclose(vs.volume, [0], atol=1e-7)

    model.step()

    assert_allclose(otpt.flow, [0], atol=1e-7)
    assert_allclose(vs.volume, [0], atol=1e-7)


def test_virtual_storage_duplicate_route():
    """ Test the VirtualStorage node """

    model = pywr.core.Model()

    inpt = Input(model, "Input", max_flow=20)
    lnk = Link(model, "Link")
    inpt.connect(lnk)
    otpt = Output(model, "Output", max_flow=10, cost=-10.0)
    lnk.connect(otpt)

    vs = pywr.core.VirtualStorage(model, "Licence", [lnk, otpt], factors=[0.5, 1.0], initial_volume=10.0, max_volume=10.0)

    model.setup()

    assert_allclose(vs.volume, [10], atol=1e-7)

    model.step()

    assert_allclose(otpt.flow, [10/1.5], atol=1e-7)
    assert_allclose(vs.volume, [0], atol=1e-7)

    model.step()

    assert_allclose(otpt.flow, [0], atol=1e-7)
    assert_allclose(vs.volume, [0], atol=1e-7)


def test_annual_virtual_storage():
    model = load_model('virtual_storage1.json')
    model.run()
    node = model.nodes["supply1"]
    rec = node.recorders[0]
    assert_allclose(rec.data[0], 10) # licence is not a constraint
    assert_allclose(rec.data[19], 10)
    assert_allclose(rec.data[20], 5) # licence is constraint
    assert_allclose(rec.data[21], 0) # licence is exhausted
    assert_allclose(rec.data[365], 10) # licence is refreshed


def test_annual_virtual_storage_with_dynamic_cost():
    model = load_model('virtual_storage2.json')
    model.run()
    node = model.nodes["supply1"]
    rec = node.recorders[0]

    assert_allclose(rec.data[0], 10)  # licence is not a constraint
    assert_allclose(rec.data[1], 5)  # now used slightly too much; switch to the other source
    assert_allclose(rec.data[2], 10)  # continue back and forth.
    assert_allclose(rec.data[3], 5)


def test_storage_spill_compensation():
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
    model = pywr.core.Model()

    catchment = pywr.core.Input(model, name="Input", min_flow=10.0, max_flow=10.0, cost=1)
    reservoir = pywr.core.Storage(model, name="Storage", max_volume=100, initial_volume=100.0)
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


def test_reservoir_circle():
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
    model = Model()

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


def test_breaklink_node():
    model = load_model('breaklink.json')
    supply = model.nodes["A"]
    transfer = model.nodes["B"]
    demand = model.nodes["C"]
    model.check()
    model.run()
    assert_allclose(supply.flow, 20)
    assert_allclose(transfer.flow, 20)
    assert_allclose(demand.flow, 20)
    assert_allclose(transfer.storage.volume, 0)


@pytest.mark.xfail(reason="Circular dependency in the JSON definition. "
                          "See GitHub issue #380: https://github.com/pywr/pywr/issues/380")
def test_reservoir_surface_area():
    from pywr.parameters import InterpolatedVolumeParameter
    model = load_model('reservoir_evaporation.json')
    model.timestepper.start = "1920-01-01"
    model.timestepper.end = "1920-01-02"
    res = model.run()
    assert (hasattr(Storage, area))
    assert isinstance(model.nodes["reservoir1"].area, InterpolatedVolumeParameter)
    assert_allclose(model.nodes["evaporation"].flow, 2.46875)


def test_reservoir_surface_area_without_area_property():
    """ Temporary test while the above test is not working. """
    model = load_model('reservoir_evaporation_without_area_property.json')
    model.timestepper.start = "1920-01-01"
    model.timestepper.end = "1920-01-02"
    res = model.run()
    assert_allclose(model.nodes["evaporation"].flow, 2.46875)


def test_run_empty():
    # empty model should raise an exception if run
    model = Model()
    with pytest.raises(ModelStructureError):
        model.run()


def test_run():
    model = load_model('simple1.json')

    # run model from start to finish
    result = model.run()
    assert(result.timestep.index == 364)

    # reset model and run again
    result = model.run()
    assert(result.timestep.index == 364)

    # run remaining timesteps
    model.reset(start=pandas.to_datetime('2015-12-01'))
    result = model.run()
    assert(result.timestep.index == 364)


def test_run_monthly():
    model = load_model('simple1_monthly.json')

    result = model.run()
    assert result.timestep.index == 11

    result = model.run()
    assert result.timestep.index == 11


def test_select_solver():
    """Test specifying the solver in JSON"""
    solver_names = [solver.name for solver in pywr.solvers.solver_registry]
    for solver_name in solver_names:
        data = '''{"metadata": {"minimum_version": "0.1"}, "nodes": {}, "edges": {}, "timestepper": {"start": "1990-01-01","end": "1999-12-31","timestep": 1}, "solver": {"name": "%s"}}''' % solver_name
        model = load_model(data=data)
        assert(model.solver.name.lower() == solver_name)

def test_solver_unrecognised():
    '''Test specifying an unrecognised solver JSON'''
    solver_name = 'foobar'
    data = '''{"metadata": {"minimum_version": "0.1"}, "nodes": {}, "edges": {}, "timestepper": {"start": "1990-01-01","end": "1999-12-31","timestep": 1}, "solver": {"name": "%s"}}''' % solver_name
    with pytest.raises(KeyError):
        model = load_model(data=data)

@pytest.mark.skipif(Model().solver.name != "glpk", reason="only valid for glpk")
@pytest.mark.parametrize("use_presolve", ["true", "false"])
def test_select_glpk_presolve(use_presolve):
    """Test specifying the solver in JSON"""
    solver_names = ["glpk"]
    for solver_name in solver_names:
        data = '''{"metadata": {"minimum_version": "0.1"}, "nodes": {}, "edges": {}, "timestepper": {"start": "1990-01-01","end": "1999-12-31","timestep": 1}, "solver": {"name": "%s", "use_presolve": %s}}''' % (solver_name, use_presolve)
        model = load_model(data=data)
        assert(model.solver.name.lower() == solver_name)
        assert(model.solver._cy_solver.use_presolve == (use_presolve == "true"))

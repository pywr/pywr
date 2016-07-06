from pywr.core import Model, Input, Output, Link, Storage, AggregatedNode

import pytest
from numpy.testing import assert_allclose
import pandas
from pandas import Timestamp

@pytest.fixture
def model(solver):
    model = Model(solver=solver)
    model.timestepper.start = Timestamp("2016-01-01")
    model.timestepper.end = Timestamp("2016-01-02")
    return model

def test_aggregated_node_two_factors(model):
    """Nodes constrained by a fixed ratio between flows (2 nodes)"""
    A = Input(model, "A")
    B = Input(model, "B", max_flow=40.0)
    Z = Output(model, "Z", max_flow=100, cost=-10)
    
    agg = AggregatedNode(model, "agg", [A, B])
    agg.factors = [0.5, 0.5]
    assert_allclose(agg.factors, [0.5, 0.5])
    
    A.connect(Z)
    B.connect(Z)
    
    model.run()
    
    assert_allclose(agg.flow, 80.0)
    assert_allclose(A.flow, 40.0)
    assert_allclose(B.flow, 40.0)

def test_aggregated_node_three_factors(model):
    """Nodes constrained by a fixed ratio between flows (3 nodes)"""
    A = Input(model, "A")
    B = Input(model, "B", max_flow=10.0)
    C = Input(model, "C")
    Z = Output(model, "Z", max_flow=100, cost=-10)
    
    agg = AggregatedNode(model, "agg", [A, B, C])
    agg.factors = [0.5, 1.0, 2.0]
    assert_allclose(agg.factors, [0.5, 1.0, 2.0])
    
    A.connect(Z)
    B.connect(Z)
    C.connect(Z)
    
    model.run()
    
    assert_allclose(agg.flow, 35.0)
    assert_allclose(A.flow, 5.0)
    assert_allclose(B.flow, 10.0)
    assert_allclose(C.flow, 20.0)

def test_aggregated_node_two_factors_time_varying(model):
    """Nodes constrained by a time-varying ratio between flows (2 nodes)"""
    model.timestepper.end = Timestamp("2016-01-03")
    
    A = Input(model, "A")
    B = Input(model, "B", max_flow=40.0)
    Z = Output(model, "Z", max_flow=100, cost=-10)
    
    agg = AggregatedNode(model, "agg", [A, B])
    agg.factors = [0.5, 0.5]
    assert_allclose(agg.factors, [0.5, 0.5])
    
    A.connect(Z)
    B.connect(Z)
    
    model.setup()
    model.step()
    
    assert_allclose(agg.flow, 80.0)
    assert_allclose(A.flow, 40.0)
    assert_allclose(B.flow, 40.0)
    
    agg.factors = [1.0, 2.0]
    
    model.step()
    
    assert_allclose(agg.flow, 60.0)
    assert_allclose(A.flow, 20.0)
    assert_allclose(B.flow, 40.0)

def test_aggregated_node_max_flow(model):
    """Nodes constrained by the max_flow of their AggregatedNode"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=2)
    Z = Output(model, "Z", max_flow=100, cost=-10)
    
    A.connect(Z)
    B.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.max_flow = 30.0
    
    model.run()
    
    assert_allclose(agg.flow, 30.0)
    assert_allclose(A.flow, 20.0)
    assert_allclose(B.flow, 10.0)

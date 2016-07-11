from pywr.core import Model, Input, Output, Link, Storage, AggregatedNode, PiecewiseLink

import pytest
from numpy.testing import assert_allclose
import pandas
from pandas import Timestamp

if pytest.config.getoption("--solver") != "glpk":
    # AggregatedNode constraints only supported by GLPK presently
    pytestmark = pytest.mark.xfail()

from helpers import load_model

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

@pytest.mark.skip(reason="not implemented")
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

def test_aggregated_node_min_flow(model):
    """Nodes constrained by the min_flow of their AggregatedNode"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=100)
    Z = Output(model, "Z", max_flow=100, cost=0)

    A.connect(Z)
    B.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.min_flow = 15.0

    model.run()

    assert_allclose(agg.flow, 15.0)
    assert_allclose(A.flow, 15.0)
    assert_allclose(B.flow, 0.0)

def test_aggregated_node_max_flow_same_route(model):
    """Unusual case where the aggregated nodes are in the same route"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=2)
    C = Input(model, "C", max_flow=50.0, cost=0)
    Z = Output(model, "Z", max_flow=100, cost=-10)

    A.connect(B)
    B.connect(Z)
    C.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.max_flow = 30.0

    model.run()

    assert_allclose(agg.flow, 30.0)
    assert_allclose(A.flow + B.flow, 30.0)

def test_aggregated_constraint_json():
    model = load_model("aggregated1.json")

    agg = model.nodes["agg"]
    assert(agg.nodes == [model.nodes["A"], model.nodes["B"]])
    assert_allclose(agg.factors, [2.0, 4.0])
    assert_allclose(agg.max_flow, 30.0)
    assert_allclose(agg.min_flow, 5.0)

def test_piecewise_constraint(model):
    """Test using an aggregated node constraint in combination with a
    piecewise link in order to create a minimum flow constraint of the form
    y = mx + c, where y is the MRF, x is the upstream flow and m and c are
    constants.

    TODO: build a standard class which simplifies the construction of this setup

             / -->-- X0 -->-- \
    A -->-- Xo -->-- X1 -->-- Xi -->-- C
             \ -->-- X2 -->-- /
                     |
                     Bo -->-- Bi --> D
    """
    flow = 100
    A = Input(model, "A", min_flow=flow, max_flow=flow)
    X = PiecewiseLink(model, name="X", cost=[-500.0, 0, 0], max_flow=[40.0, None, None])
    C = Output(model, "C")

    A.connect(X)
    X.connect(C)

    # create a new input inside the piecewise link which only has access
    # to flow travelling via the last sublink (X2)
    Bo = Output(model, "Bo", domain=X.sub_domain)
    Bi = Input(model, "Bi")
    D = Output(model, "D", max_flow=50, cost=-100)
    Bo.connect(Bi)
    Bi.connect(D)
    X.sublinks[-1].connect(Bo)


    agg = AggregatedNode(model, "agg", X.sublinks[1:])
    agg.factors = [3.0, 1.0]

    model.step()
    assert_allclose(D.flow, (100 - 40) * 0.25)

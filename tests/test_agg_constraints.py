import pywr.solvers
from pywr.core import (
    Model,
    Input,
    Output,
    Link,
    Storage,
    AggregatedNode,
    PiecewiseLink,
    MultiSplitLink,
)
from pywr.parameters import ConstantParameter, DailyProfileParameter

import pytest
from numpy.testing import assert_allclose
import pandas
import numpy as np
from pandas import Timestamp

from helpers import load_model


@pytest.fixture
def model():
    model = Model()
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

    for f in agg.factors:
        assert isinstance(f, ConstantParameter)
        assert_allclose(f.get_double_variables(), 0.5)

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

    for f, v in zip(agg.factors, [0.5, 1.0, 2.0]):
        assert isinstance(f, ConstantParameter)
        assert_allclose(f.get_double_variables(), v)

    A.connect(Z)
    B.connect(Z)
    C.connect(Z)

    model.run()

    assert_allclose(agg.flow, 35.0)
    assert_allclose(A.flow, 5.0)
    assert_allclose(B.flow, 10.0)
    assert_allclose(C.flow, 20.0)


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


@pytest.mark.parametrize(
    "flow_weights,expected_agg_flow,expected_A_flow,expected_B_flow",
    [
        ([2.0, 1.0], 30.0, 15.0, 0.0),
        ([0.5, 2.0], 30.0, 20.0, 10.0),
    ],
)
def test_aggregated_node_max_flow_with_weights(
    model, flow_weights, expected_agg_flow, expected_A_flow, expected_B_flow
):
    """Nodes constrained by the weighted max_flow of their AggregatedNode"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=8)
    Z = Output(model, "Z", max_flow=100, cost=-10)

    A.connect(Z)
    B.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.flow_weights = flow_weights
    agg.max_flow = 30.0

    model.run()

    assert_allclose(agg.flow, expected_agg_flow)
    assert_allclose(A.flow, expected_A_flow)
    assert_allclose(B.flow, expected_B_flow)


@pytest.mark.skipif(
    Model().solver.name == "lpsolve", reason="Not supported in lpsolve."
)
def test_aggregated_node_max_flow_parameter(model):
    """Nodes constrained by the max_flow of their AggregatedNode using a Parameter"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=2)
    Z = Output(model, "Z", max_flow=100, cost=-10)

    A.connect(Z)
    B.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.max_flow = ConstantParameter(model, 30.0)

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


@pytest.mark.skipif(
    Model().solver.name == "lpsolve", reason="Not supported in lpsolve."
)
def test_aggregated_node_min_flow_parameter(model):
    """Nodes constrained by the min_flow of their AggregatedNode"""
    A = Input(model, "A", max_flow=20.0, cost=1)
    B = Input(model, "B", max_flow=20.0, cost=100)
    Z = Output(model, "Z", max_flow=100, cost=0)

    A.connect(Z)
    B.connect(Z)

    agg = AggregatedNode(model, "agg", [A, B])
    agg.min_flow = ConstantParameter(model, 15.0)

    model.run()

    assert_allclose(agg.flow, 15.0)
    assert_allclose(A.flow, 15.0)
    assert_allclose(B.flow, 0.0)


@pytest.mark.skipif(
    Model().solver.name == "glpk-edge", reason="Not valid for GLPK Edge based solver."
)
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
    assert agg.nodes == [model.nodes["A"], model.nodes["B"]]

    for f, v in zip(agg.factors, [2.0, 4.0]):
        assert isinstance(f, ConstantParameter)
        assert_allclose(f.get_double_variables(), v)

    assert_allclose(agg.max_flow, 30.0)
    assert_allclose(agg.min_flow, 5.0)

    model.run()


@pytest.mark.skipif(
    Model().solver.name != "glpk" or Model().solver.use_unsafe_api,
    reason="Only valid for GLPK route based solver using safe API.",
)
def test_aggregated_constraint_with_two_nodes_in_same_route_json():
    """Test the case where an aggregated node contains two nodes in the same route.

    In the route based solver this is not currently possible.
    """
    model = load_model("aggregated_with_two_nodes_same_route.json")

    agg = model.nodes["agg"]
    assert agg.nodes == [
        model.nodes["A"],
        model.nodes["B"],
        model.nodes["C"],
        model.nodes["X"],
    ]

    for f, v in zip(agg.factors, [1.0, 1.0, 1.0, 1.0]):
        assert isinstance(f, ConstantParameter)
        assert_allclose(f.get_double_variables(), v)

    assert_allclose(agg.max_flow, 30.0)
    assert_allclose(agg.min_flow, 5.0)

    with pytest.raises(pywr.solvers.GLPKInternalError):
        model.run()


@pytest.mark.parametrize("flow", (100.0, 200.0, 300.0))
def test_piecewise_constraint(model, flow):
    """Test using an aggregated node constraint in combination with a
    piecewise link in order to create a minimum flow constraint of the form
    y = mx + c, where y is the MRF, x is the upstream flow and m and c are
    constants.

    Flows are tested at 100, 200 and 300 to ensure the aggregated ratio works
    when there is too much to route entirely through to node 'D'.

    ::

                 / -->-- X0 -->-- \
        A -->-- Xo -->-- X1 -->-- Xi -->-- C
                 \\ -->-- X2 -->-- /
                         |
                         Bo -->-- Bi --> D
    """
    A = Input(model, "A", min_flow=flow, max_flow=flow)
    X = PiecewiseLink(
        model, name="X", nsteps=3, costs=[-500.0, 0, 0], max_flows=[40.0, None, None]
    )
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
    assert_allclose(D.flow, min((flow - 40) * 0.25, 50.0))


@pytest.mark.parametrize("flow", (100.0, 200.0, 300.0))
def test_multipiecewise_constraint(model, flow):
    """Test using an aggregated node in combination with a MultiSplitLink.

    This test is the same as the `test_piecewise_constraint` but using the MultiSplitLink API
     for brevity.
    """
    A = Input(model, "A", min_flow=flow, max_flow=flow)
    X = MultiSplitLink(
        model,
        name="X",
        nsteps=2,
        costs=[-500.0, 0],
        max_flows=[40.0, None],
        factors=[3, 1],
        extra_slots=1,
        slot_names=["river", "abstraction"],
    )
    C = Output(model, "C")
    D = Output(model, "D", max_flow=50, cost=-100)

    A.connect(X)
    X.connect(C, from_slot="river")
    X.connect(D, from_slot="abstraction")

    model.step()
    assert_allclose(D.flow, min((flow - 40) * 0.25, 50.0))


@pytest.mark.skipif(
    Model().solver.name not in ["glpk", "glpk-edge"],
    reason="Dynamic factors for agg nodes have only \
                                                                              been implemented for glpk solvers",
)
def test_dynamic_factors(model):

    model.timestepper.end = Timestamp("2016-01-03")

    A = Input(model, "A", max_flow=10.0)
    B = Input(model, "B", max_flow=10.0)
    C = Input(model, "C", max_flow=10.0)
    Z = Output(model, "Z", cost=-10)

    agg = AggregatedNode(model, "agg", [A, B, C])
    agg.max_flow = 10.0
    factor1 = DailyProfileParameter(
        model, np.append(np.array([0.8, 0.3]), np.ones(364))
    )
    factor2 = DailyProfileParameter(
        model, np.append(np.array([0.1, 0.3]), np.ones(364))
    )
    factor3 = DailyProfileParameter(
        model, np.append(np.array([0.1, 0.4]), np.ones(364))
    )

    agg.factors = [factor1, factor2, factor3]

    A.connect(Z)
    B.connect(Z)
    C.connect(Z)

    model.step()

    assert_allclose(A.flow, 8)
    assert_allclose(B.flow, 1)
    assert_allclose(C.flow, 1)

    model.step()

    assert_allclose(A.flow, 3)
    assert_allclose(B.flow, 3)
    assert_allclose(C.flow, 4)


@pytest.mark.skipif(
    Model().solver.name not in ["glpk", "glpk-edge"],
    reason="Dynamic factors for agg nodes have only \
                                                                              been implemented for glpk solvers",
)
def test_dynamic_factors_load(model):

    model.timestepper.end = Timestamp("2016-01-03")

    A = Input(model, "A", max_flow=10.0)
    B = Input(model, "B", max_flow=10.0)
    Z = Output(model, "Z", cost=-10, max_flow=10.0)

    A.connect(Z)
    B.connect(Z)

    DailyProfileParameter(
        model, np.append(np.array([3, 4]), np.ones(364)), name="factor1"
    )

    data = {"name": "agg", "factors": [1, "factor1"], "nodes": ["A", "B"]}

    node = AggregatedNode.pre_load(model, data)
    node.finalise_load()

    model.step()

    assert_allclose(A.flow, 2.5)
    assert_allclose(B.flow, 7.5)

    model.step()

    assert_allclose(A.flow, 2)
    assert_allclose(B.flow, 8)

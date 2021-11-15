import numpy as np
import pytest
from fixtures import three_storage_model

from pywr.core import Model, Input, Output, Link, Storage, AggregatedStorage
from pywr.parameters.control_curves import ControlCurveParameter


def test_aggregated_storage(three_storage_model):
    """Test `AggregatedStorage` correct sums multiple `Storage`."""
    m = three_storage_model

    agg_stg = m.nodes["Total Storage"]
    stgs = [m.nodes["Storage {}".format(num)] for num in range(3)]
    # Check initial volume is aggregated correctly
    np.testing.assert_allclose(
        agg_stg.initial_volume, sum(s.initial_volume for s in stgs)
    )

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))


def test_aggregated_storage_initial_pc(three_storage_model):
    """Test `AggregatedStorage` correct sums multiple `Storage` with proportional initial volumes."""
    m = three_storage_model

    agg_stg = m.nodes["Total Storage"]
    stgs = [m.nodes["Storage {}".format(num)] for num in range(3)]
    for stg in stgs:
        stg.initial_volume = None
        stg.initial_volume_pc = 0.9
    # Check initial volume is aggregated correctly
    np.testing.assert_allclose(
        agg_stg.initial_volume, sum(s.initial_volume_pc * s.max_volume for s in stgs)
    )

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))


def test_aggregated_storage_scenarios(three_storage_model):
    """Test `AggregatedScenario` correctly sums over scenarios."""
    from pywr.core import Scenario
    from pywr.parameters import ConstantScenarioParameter

    m = three_storage_model
    sc = Scenario(m, "A", size=5)

    agg_stg = m.nodes["Total Storage"]
    stgs = [m.nodes["Storage {}".format(num)] for num in range(3)]
    # Setup scenario
    inpt1 = m.nodes["Input 1"].max_flow = ConstantScenarioParameter(
        m, sc, range(sc.size)
    )

    m.setup()

    # Test initial volume
    np.testing.assert_allclose(agg_stg.volume, sum(s.initial_volume for s in stgs))
    current_pc = sum(s.initial_volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, sum(s.volume for s in stgs))
    current_pc = sum(s.volume for s in stgs) / (sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, sum(s.flow for s in stgs))


def test_aggregated_storage_connectivity(three_storage_model):
    """Test `AggregatedStorage` has the correct attributes."""
    m = three_storage_model
    agg_stg = m.nodes["Total Storage"]
    inpt = m.nodes["Input 1"]

    with pytest.raises(AttributeError):
        agg_stg.connect()

    with pytest.raises(TypeError):
        # If not `Connectable` the node won't have a iter_slots attribute
        inpt.connect(agg_stg)


def test_aggregated_storage_attributes(three_storage_model):
    """Test `AggregatedStorage` has the correct attributes."""
    m = three_storage_model
    agg_stg = m.nodes["Total Storage"]

    with pytest.raises(AttributeError):
        agg_stg.get_max_volume()

    with pytest.raises(AttributeError):
        agg_stg.get_min_volume()


def test_aggregated_storage_control_curve(three_storage_model):
    """Test using a control curve based on an aggregate storage, rather than
    a single storage.
    """
    model = three_storage_model

    # create a new supply node
    inpt = Input(model, "Input 3", cost=-1000)
    inpt.connect(model.nodes["Output 0"])
    inpt.connect(model.nodes["Output 1"])
    inpt.connect(model.nodes["Output 2"])

    # limit the flow of the new node using a control curve on the aggregate storage
    curves = [0.5]  # 50%
    values = [0, 5]
    inpt.max_flow = ControlCurveParameter(
        model, model.nodes["Total Storage"], curves, values
    )

    # initial storage is > 50% so flow == 0
    model.step()
    np.testing.assert_allclose(inpt.flow, 0.0)

    # set initial storage to < 50%
    storages = [node for node in model.nodes if isinstance(node, Storage)]
    for node, value in zip(storages, [0.6, 0.1, 0.1]):
        if isinstance(node, Storage):
            node.initial_volume = node.max_volume * value

    # now below the control curve, so flow is allowed
    model.reset()
    model.step()
    np.testing.assert_allclose(inpt.flow, 5.0)


def test_aggregated_node(three_storage_model):
    """Test `AggregatedNode` correct sums multiple `Node`."""
    m = three_storage_model

    agg_otpt = m.nodes["Total Output"]
    otpts = [m.nodes["Output {}".format(num)] for num in range(3)]

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, sum(o.flow for o in otpts))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, sum(o.flow for o in otpts))


def test_aggregated_node_scenarios(three_storage_model):
    """Test `AggregatedScenario` correctly sums over scenarios."""
    from pywr.core import Scenario
    from pywr.parameters import ConstantScenarioParameter

    m = three_storage_model
    sc = Scenario(m, "A", size=5)

    agg_otpt = m.nodes["Total Output"]
    otpts = [m.nodes["Output {}".format(num)] for num in range(3)]
    # Setup scenario
    inpt1 = m.nodes["Input 1"].max_flow = ConstantScenarioParameter(
        m, sc, range(sc.size)
    )

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, sum(o.flow for o in otpts))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, sum(o.flow for o in otpts))


def test_aggregated_node_connectivity(three_storage_model):
    """Test `AggregatedStorage` has the correct attributes."""
    m = three_storage_model
    agg_otpt = m.nodes["Total Output"]
    inpt = m.nodes["Input 1"]

    with pytest.raises(AttributeError):
        agg_otpt.connect()

    with pytest.raises(TypeError):
        # If not `Connectable` the node won't have a iter_slots attribute
        inpt.connect(agg_otpt)


def test_aggregated_node_attributes(three_storage_model):
    """Test `AggregatedStorage` has the correct attributes."""
    m = three_storage_model
    agg_otpt = m.nodes["Total Output"]

    with pytest.raises(AttributeError):
        agg_otpt.get_max_volume()

    with pytest.raises(AttributeError):
        agg_otpt.get_min_volume()

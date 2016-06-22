import numpy as np
import pytest
from fixtures import three_storage_model


def test_aggregated_storage(three_storage_model):
    """ Test `AggregatedStorage` correct sums multiple `Storage`. """
    m = three_storage_model

    agg_stg = m.node['Total Storage']
    stgs = [m.node['Storage {}'.format(num)] for num in range(3)]
    # Check initial volume is aggregated correclty
    np.testing.assert_allclose(agg_stg.initial_volume, np.sum(s.initial_volume for s in stgs))

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, np.sum(s.volume for s in stgs))
    current_pc = np.sum(s.volume for s in stgs) / (np.sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, np.sum(s.flow for s in stgs))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, np.sum(s.volume for s in stgs))
    current_pc = np.sum(s.volume for s in stgs) / (np.sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, np.sum(s.flow for s in stgs))


def test_aggregated_storage_scenarios(three_storage_model):
    """ Test `AggregatedScenario` correctly sums over scenarios. """
    from pywr.core import Scenario
    from pywr.parameters import ConstantScenarioParameter
    m = three_storage_model
    sc = Scenario(m, 'A', size=5)

    agg_stg = m.node['Total Storage']
    stgs = [m.node['Storage {}'.format(num)] for num in range(3)]
    # Setup scenario
    inpt1 = m.node['Input 1'].max_flow = ConstantScenarioParameter(sc, range(sc.size))

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, np.sum(s.volume for s in stgs))
    current_pc = np.sum(s.volume for s in stgs) / (np.sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, np.sum(s.flow for s in stgs))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_stg.volume, np.sum(s.volume for s in stgs))
    current_pc = np.sum(s.volume for s in stgs) / (np.sum(s.max_volume for s in stgs))
    np.testing.assert_allclose(agg_stg.current_pc, current_pc)
    np.testing.assert_allclose(agg_stg.flow, np.sum(s.flow for s in stgs))


def test_aggregated_storage_connectivity(three_storage_model):
    """ Test `AggregatedStorage` has the correct attributes. """
    m = three_storage_model
    agg_stg = m.node['Total Storage']
    inpt = m.node['Input 1']

    with pytest.raises(AttributeError):
        agg_stg.connect()

    with pytest.raises(TypeError):
        # If not `Connectable` the node won't have a iter_slots attribute
        inpt.connect(agg_stg)


def test_aggregated_storage_attributes(three_storage_model):
    """ Test `AggregatedStorage` has the correct attributes. """
    m = three_storage_model
    agg_stg = m.node['Total Storage']

    with pytest.raises(AttributeError):
        agg_stg.get_max_volume()

    with pytest.raises(AttributeError):
        agg_stg.get_min_volume()

    with pytest.raises(AttributeError):
        agg_stg.get_cost()


def test_aggregated_node(three_storage_model):
    """ Test `AggregatedNode` correct sums multiple `Node`. """
    m = three_storage_model

    agg_otpt = m.node['Total Output']
    otpts = [m.node['Output {}'.format(num)] for num in range(3)]

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, np.sum(o.flow for o in otpts))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, np.sum(o.flow for o in otpts))


def test_aggregated_node_scenarios(three_storage_model):
    """ Test `AggregatedScenario` correctly sums over scenarios. """
    from pywr.core import Scenario
    from pywr.parameters import ConstantScenarioParameter
    m = three_storage_model
    sc = Scenario(m, 'A', size=5)

    agg_otpt = m.node['Total Output']
    otpts = [m.node['Output {}'.format(num)] for num in range(3)]
    # Setup scenario
    inpt1 = m.node['Input 1'].max_flow = ConstantScenarioParameter(sc, range(sc.size))

    m.setup()
    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, np.sum(o.flow for o in otpts))

    m.step()

    # Finally check volume is summed correctly
    np.testing.assert_allclose(agg_otpt.flow, np.sum(o.flow for o in otpts))


def test_aggregated_node_connectivity(three_storage_model):
    """ Test `AggregatedStorage` has the correct attributes. """
    m = three_storage_model
    agg_otpt = m.node['Total Output']
    inpt = m.node['Input 1']

    with pytest.raises(AttributeError):
        agg_otpt.connect()

    with pytest.raises(TypeError):
        # If not `Connectable` the node won't have a iter_slots attribute
        inpt.connect(agg_otpt)


def test_aggregated_node_attributes(three_storage_model):
    """ Test `AggregatedStorage` has the correct attributes. """
    m = three_storage_model
    agg_otpt = m.node['Total Output']

    with pytest.raises(AttributeError):
        agg_otpt.get_max_volume()

    with pytest.raises(AttributeError):
        agg_otpt.get_min_volume()

    with pytest.raises(AttributeError):
        agg_otpt.get_cost()

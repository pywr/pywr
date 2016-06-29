from pywr.core import Model, Input, Output, Link, Scenario
from pywr.parameters import (CachedParameter, CachedTimeParameter,
    load_parameter, ConstantParameter, FunctionParameter)
from pywr.parameters._parameters import Parameter as BaseParameter
import pytest
from numpy.testing import assert_allclose
import pandas

@pytest.fixture
def model(solver):
    model = Model(
        solver=solver,
        start=pandas.to_datetime("2016-01-01"),
        end=pandas.to_datetime("2016-01-10"),
    )
    return model

@pytest.fixture
def simple_model(model):
    inpt = Input(model, "input")
    otpt = Output(model, "output", max_flow=20, cost=-1000)
    inpt.connect(otpt)
    return model

def create_function(data, model=None):
    def func(self, timestep, scenario_index):
        func.count += 1
        if isinstance(data, BaseParameter):
            value = data.value(timestep, scenario_index)
        else:
            value = data
        return value
    func.count = 0
    return func

def test_cache_both(simple_model):
    """
    Example of caching when it isn't needed - the function is only called
    once per timestep anyway
    """
    model = simple_model
    func = create_function(15.0)
    param = FunctionParameter(None, func)

    inpt = model.nodes["input"]
    inpt.max_flow = CachedParameter(param)

    model.run()
    assert_allclose(inpt.flow, 15.0)

    assert(func.count == 10)

def test_load_cache_both(simple_model):
    """
    As previous, but loading the parameter using `load_parameter`
    """
    model = simple_model

    data = {
        "cached": "both",
        "type": "constant",
        "values": 15.0
    }

    inpt = model.nodes["input"]
    inpt.max_flow = load_parameter(model, data)
    assert(inpt.max_flow.__class__ is CachedParameter)
    assert(isinstance(inpt.max_flow.parameter, ConstantParameter))

    model.run()
    assert_allclose(inpt.flow, 15)

def test_cache_both_shared(simple_model):
    """
    When a cached parameter is shared between nodes it should only be
    called once per timestep
    """
    model = simple_model
    func = create_function(15.0)
    param = FunctionParameter(None, func)

    inpt = model.nodes["input"]
    otpt = model.nodes["output"]
    inpt.max_flow = param
    otpt.max_flow = param

    model.run()
    assert_allclose(inpt.flow, 15.0)

    # inefficient - parameter is called twice per timestep
    assert(func.count == 20)

    func2 = create_function(25.0)
    param = FunctionParameter(None, func2)
    cached = CachedParameter(param)

    inpt.max_flow = cached
    otpt.max_flow = cached

    model.run()
    assert_allclose(inpt.flow, 25.0)

    # parameter is only called once per timestep
    assert(func2.count == 10)

def test_unknown_cache_type(model):
    data = {
        "cached": "fail",
        "type": "constant",
        "values": 42.0
    }
    with pytest.raises(ValueError):
        parameter = load_parameter(model, data)

# -*- coding: utf-8 -*-

import pywr
from pywr.core import Model, Input, Output, Link, Storage

import pandas
import datetime

import pytest

@pytest.fixture()
def model(request, solver):
    model = Model(solver=solver)
    return model

@pytest.fixture()
def simple_linear_model(request, solver):
    """
    Make a simple model with a single Input and Output.

    Input -> Link -> Output

    """
    model = Model(solver=solver)
    inpt = Input(model, name="Input")
    lnk = Link(model, name="Link", cost=1.0)
    inpt.connect(lnk)
    otpt = Output(model, name="Output")
    lnk.connect(otpt)

    return model

@pytest.fixture()
def simple_storage_model(request, solver):
    """
    Make a simple model with a single Input, Storage and Output.
    
    Input -> Storage -> Output
    """

    model = pywr.core.Model(
        parameters={
            'timestamp_start': pandas.to_datetime('2016-01-01'),
            'timestamp_finish': pandas.to_datetime('2016-01-05'),
            'timestep': datetime.timedelta(1),
        },
        solver=solver
    )

    inpt = Input(model, name="Input", max_flow=5.0, cost=-1)
    res = Storage(model, name="Storage", num_outputs=1, num_inputs=1, max_volume=20, volume=10)
    otpt = Output(model, name="Output", max_flow=8, cost=-999)
    
    inpt.connect(res)
    res.connect(otpt)
    
    return model

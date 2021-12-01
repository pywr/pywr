# -*- coding: utf-8 -*-

import pywr
from pywr.core import (
    Model,
    Input,
    Output,
    Link,
    Storage,
    AggregatedStorage,
    AggregatedNode,
)

import pandas
import datetime
from helpers import load_model
import pytest


@pytest.fixture()
def model(request):
    model = Model()
    return model


@pytest.fixture()
def simple_linear_model(request):
    """
    Make a simple model with a single Input and Output.

    Input -> Link -> Output

    """
    model = Model()
    inpt = Input(model, name="Input")
    lnk = Link(model, name="Link", cost=1.0)
    inpt.connect(lnk)
    otpt = Output(model, name="Output")
    lnk.connect(otpt)

    return model


@pytest.fixture()
def simple_storage_model(request):
    """
    Make a simple model with a single Input, Storage and Output.

    Input -> Storage -> Output
    """

    model = pywr.core.Model(
        start=pandas.to_datetime("2016-01-01"),
        end=pandas.to_datetime("2016-01-05"),
        timestep=datetime.timedelta(1),
    )

    inpt = Input(model, name="Input", max_flow=5.0, cost=-1)
    res = Storage(
        model, name="Storage", outputs=1, inputs=1, max_volume=20, initial_volume=10
    )
    otpt = Output(model, name="Output", max_flow=8, cost=-999)

    inpt.connect(res)
    res.connect(otpt)

    return model


@pytest.fixture()
def three_storage_model(request):
    """
    Make a simple model with three input, storage and output nodes. Also adds
    an `AggregatedStorage` and `AggregatedNode`.

        Input 0 -> Storage 0 -> Output 0
        Input 1 -> Storage 1 -> Output 1
        Input 2 -> Storage 2 -> Output 2


    """
    return load_model("three_storage.json")

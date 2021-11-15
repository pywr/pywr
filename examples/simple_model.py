#!/bin/python
"""
This script defines and runs a very simple model.

The network looks like this:

  A -->-- B -->-- C

"""
from pywr.model import Model
from pywr.nodes import Input, Output, Link


def create_model():
    # create a model
    model = Model(start="2016-01-01", end="2019-12-31", timestep=7)

    # create three nodes (an input, a link, and an output)
    A = Input(model, name="A", max_flow=10.0)
    B = Link(model, name="B", cost=10.0)
    C = Output(model, name="C", max_flow=5.0, cost=-20.0)

    # connect the nodes together
    A.connect(B)
    B.connect(C)

    return model


if __name__ == "__main__":
    # create the model and check it is valid
    model = create_model()
    model.check()

    # run the model
    result = model.run()
    print(result)

    # check the result of the model is as expected
    A = model.nodes["A"]
    assert abs(A.flow[0] - 5.0) < 0.000001

#!/bin/python
"""
This script sets up and runs the most simple model possible.

  A -->-- B -->-- C

"""
from pywr.core import Model, Input, Link, Output

def create_model():
    # create a model
    model = Model(start="2016-01-01", end="2019-12-31", timestep=7)

    # create three nodes (an input, a link, and an output)
    A = Input(model, name="A", max_flow=10.0)
    B = Link(model, name="B", cost=1.0)
    C = Output(model, name="C", max_flow=5.0, cost=-2.0)

    # connect nodes
    A.connect(B)
    B.connect(C)

    return model

if __name__ == '__main__':
    model = create_model()
    model.check()
    model.run()

    # check result was as expected
    assert(abs(model.nodes["A"].flow[0] - 5.0) < 0.000001)

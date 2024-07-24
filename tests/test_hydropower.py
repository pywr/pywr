"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""

import numpy as np

from helpers import load_model

def test_hydropower_results():
    """
        Use a simple model of a Reservoir to test that the area, level,
        volume, evaporation, rainfall behave as expected

        Catchment -> Link -> Reservoir -> Link 
                       |         |        |    
                       v         v        |    - > Link -> Output        |
                    Output     Turbine    |   /       \ 
                                     \    |  /         - -> Output
                                      \   V /              
         Catchment -> Reservoir -------> Link -> Output                                           
    """
    model = load_model("TestingReservoirAndTurbineNodes.json")
    model.run()

    df = model.recorders["__Storage reservoir 1__:recorder1"].to_dataframe()
    assert df.shape == (12, 1)

    res1_results = [round(c[0], 2) for c in  model.recorders["__Storage reservoir 1__:recorder1"].to_dataframe().values]
    assert res1_results == [19.15, 0.0, 20.0, 17.88, 0.0, 0.0, 0.0, 0.0,-15.0, -75.0, -24.17, 0.0]

    res2_results = [round(c[0], 2) for c in  model.recorders["__Storage reservoir 2__:recorder1"].to_dataframe().values]
    assert res2_results == [-56.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
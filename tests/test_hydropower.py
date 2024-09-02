"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""

import numpy as np

from helpers import load_model

def test_hydropower_results():
    """
        Test the additions made to the reservoir node, which creates nodes internally for ease of use.
        To test, take a model which contains a hydropower implementation using explitic nodes for evaporaton and rainfall,
        and compare to an identical model, except using the reservoir node which creates these nodes internally.
        This is a derivation of the 'two reservoir' model which is used in other tests.
    """
    model1 = load_model("TestingReservoirAndTurbineNodes_NewNodes.json")
    model2 = load_model("TestingReservoirAndTurbineNodes_OldNodes.json")

    model1.run()
    model2.run()

    df1 = model1.recorders["__Hydropower plant__:hydropower recorder"].to_dataframe()
    assert df1.shape == (12, 1)

    df2 = model2.recorders["__Hydropower plant__:hydropower recorder"].to_dataframe()
    assert df2.shape == (12, 1)

    hp1_results_NewNodes = [round(c[0], 2) for c in  model1.recorders["__Hydropower plant__:hydropower recorder"].to_dataframe().values]
    hp1_results_OldNodes = [round(c[0], 2) for c in  model2.recorders["__Hydropower plant__:hydropower recorder"].to_dataframe().values]
    assert hp1_results_NewNodes == hp1_results_OldNodes

    res2_results_NewNodes = [round(c[0], 2) for c in  model1.recorders["__Storage reservoir 2__:volume"].to_dataframe().values]
    res2_results_OldNodes = [round(c[0], 2) for c in  model2.recorders["__Storage reservoir 2__:volume"].to_dataframe().values]
    assert res2_results_NewNodes == res2_results_OldNodes

    evp2_results_NewNodes = [round(c[0], 2) for c in  model1.recorders["__Storage reservoir 2_evaporation__:evaporation"].to_dataframe().values]
    evp2_results_OldNodes = [round(c[0], 2) for c in  model2.recorders["__Storage reservoir 2_evaporation__:evaporation"].to_dataframe().values]
    assert evp2_results_NewNodes == evp2_results_OldNodes

    # Rainfall
    ra2_results_NewNodes = [round(c[0], 2) for c in  model1.recorders["__Storage reservoir 2_rainfall__:rainfall"].to_dataframe().values]
    ra2_results_OldNodes = [round(c[0], 2) for c in  model2.recorders["__Storage reservoir 2_rainfall__:rainfall"].to_dataframe().values]
    assert ra2_results_NewNodes == ra2_results_OldNodes

    # Reservoir 1
    res1_results_NewNodes = [round(c[0], 2) for c in  model1.recorders["__Storage reservoir 1__:volume"].to_dataframe().values]
    res1_results_OldNodes = [round(c[0], 2) for c in  model2.recorders["__Storage reservoir 1__:volume"].to_dataframe().values]
    assert res1_results_NewNodes == res1_results_OldNodes


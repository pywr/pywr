"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""

import numpy as np

from helpers import load_model

from pywr.recorders import (
    NumpyArrayStorageRecorder
)   


def test_hydropower_results():

    model = load_model("hydropower_verification.json")

    gerd_rec = NumpyArrayStorageRecorder(model, model.nodes["RES1"])

    model.run()

    gerd_st_df = gerd_rec.to_dataframe()
    gerd_turbine_df = model.recorders["__RES1_turbine__:hydropowerrecorder"].to_dataframe()

    """
        Verified values for start and end sequences of results period.
        Keys must match f"{}_df" var names, values are dicts of
        slice-as-tuple: np.array expected values.
    """
    expected = {
        "gerd_st": {
            (0,5): np.array([[46360.63694919],
                                [42965.88915086],
                                [39208.38512899],
                                [35785.00239353],
                                [32702.58927283]]),

            (-5,None): np.array([[40747.57945841],
                                    [49263.9095238 ],
                                    [53011.73528208],
                                    [53127.665874  ],
                                    [50834.14051615]])
        },
        "gerd_turbine": {
            (0,5): np.array([[1730.1752206 ],
                                [1595.41837885],
                                [1466.1889006 ],
                                [1329.45736668],
                                [1210.65347216]]),

            (-5,None): np.array([[1070.2307505 ],
                                    [1384.66567242],
                                    [1706.67934078],
                                    [1827.94504144],
                                    [1831.7610369 ]])
        }
    }

    assert gerd_st_df.shape == (240, 1)
    assert gerd_turbine_df.shape == (240, 1)

    for elem, ranges in expected.items():
        for s,v in ranges.items():
            df = locals()[f"{elem}_df"]
            assert np.all(np.isclose(df[slice(*s)].values, v))
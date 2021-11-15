from pywr.core import Model
from pywr.recorders import Recorder
from pywr.recorders._recorders import NodeRecorder
import pandas
import numpy as np


if __name__ == "__main__":

    m = Model.load("hydropower_example.json")
    stats = m.run()
    print(stats)

    print(m.recorders["turbine1_energy"].values())

    df = m.to_dataframe()
    print(df.head())

    from matplotlib import pyplot as plt

    df.plot(subplots=True)
    plt.show()

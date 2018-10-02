import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    store = pd.HDFStore('two_reservoir_moea.h5')

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette()

    for color, key in zip(colors, store.keys()):
        print(key)
        df = store[key]
        label = key.replace('/', '')
        df.plot.scatter(ax=ax, x='deficit', y='transferred', label=label, color=color)

    plt.show()

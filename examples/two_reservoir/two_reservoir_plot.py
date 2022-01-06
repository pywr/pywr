import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    store = pd.HDFStore("two_reservoir_moea.h5")

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette()

    for color, key in zip(colors, store.keys()):
        df = store[key]
        label = key.replace("/", "").replace("_", " ").capitalize()
        df.plot.scatter(ax=ax, x="deficit", y="transferred", label=label, color=color)

    ax.set_xlabel("Total deficit [$Mm^3$]")
    ax.set_ylabel("Total transfer [$Mm^3$]")
    ax.grid()
    fig.savefig("two_reservoir_pareto.png")

    plt.show()

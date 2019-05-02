from pywr.core import Model
from pywr.recorders import NumpyArrayNodeRecorder
import pandas
from matplotlib import pyplot as plt


def main():

    model = Model.load('simple_routing.json')

    recorders = []
    for node in model.nodes:
        r = NumpyArrayNodeRecorder(model, node)
        recorders.append(r)

    model.run()

    df = {r.node.name:r.to_dataframe() for r in recorders}
    df = pandas.concat(df, axis=1)

    df.plot()
    plt.show()


if __name__ == '__main__':
    main()

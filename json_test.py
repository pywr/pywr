import os
from pywr.core import Model, Storage

filename = 'model.json'
folder = os.path.dirname(os.path.abspath(filename))

with open(filename, 'r') as f:
    model = Model.load(f, path=folder)

#print(list(model.nodes))
#print(list(model.edges()))

recorder = model.nodes['reservoir1'].recorders[0]

model.check()
model.run()

for node in model.nodes:
    if isinstance(node, Storage):
        print(node, node.volume)
    else:
        print(node, node.flow)

recorder = model.recorders["aggregatedrecorder"]
print(recorder.value())

# import matplotlib.pyplot as plt
# plt.plot(recorder.data)
# plt.show()

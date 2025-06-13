# Demand savings

Demand restrictions are a common concept in water resource models. When the state of a water resource is poor
(for example, lower than normal reservoir levels) demand restrictions may be imposed on customers in order to reduce
the likelihood of a failure to supply.

The demand at a given timestep can be represented using the equation below:

<p style="text-align: center;">
d = d<sub>mean</sub> * p * (1-s)
</p>

where _d_ is the final demand to be calculated, _d<sub>mean</sub>_ is the baseline demand, _p_ is the annual
profile factor and _s_ is the demand restriction to apply (e.g. 5%).

Consider the following example. Each parameter is explained in the subsections below.

```json
{
  "parameters": {
    "demand_baseline": {
      "type": "constant",
      "value": 50
    },
    "demand_profile": {
      "type": "monthlyprofile",
      "values": [0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9, 0.9]
    },
    "demand_restriction_level": {
      "type": "controlcurveindex",
      "storage_node": "Central Reservoir",
      "control_curves": [
        "level1",
        "level2"
      ]
    },
    "level1": {
      "type": "constant",
      "value": 0.8
    },
    "level2": {
      "type": "constant",
      "value": 0.5
    },
    "demand_restriction_factor": {
      "type": "indexedarray",
      "index_parameter": "demand_restriction_level",
      "params": [
        {
          "type": "constant",
          "value": 1.0
        },
        {
          "type": "monthlyprofile",
          "values": [0.95, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95]
        },
        {
          "type": "monthlyprofile",
          "values": [0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8]
        }
      ]
    }
  }
}
```

## Baseline demand
The baseline demand _d<sub>mean</sub>_ is specified as a constant parameter called `"demand_baseline"`. This is
often the mean annual demand.

## Profile
_p_ is the profile implemented as a monthly profile called `"demand_profile"` that varies the demand throughout the year. The demand in May -
August is 1.2x the baseline demand, with the rest of the year at 0.9x the baseline, forming the common "top hat" 
profile shown below.

<p align="center">

```python exec="1" html="1"
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
import pandas

profile = [0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9, 0.9]

dates = pandas.date_range("2015-01-01", "2015-12-31")
values = [profile[d.month - 1] for d in dates]

fig, ax = plt.subplots()

ax.plot(np.arange(0, len(dates)), values, linewidth=2)

ax.set_xlabel("Day of year")
ax.set_ylabel("Demand factor")

ax.grid(True)
ax.set_ylim(0.0, 1.5)
ax.set_xlim(0, 365)
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```    
</p>

## Demand restriction level
The demand restriction level or index is often related to the storage of strategic reservoirs in relation to a
control curve (the expected reservoir volume at a given time of the year).

The level is implemented in the parameter `"demand_restriction_level"`
using a [pywr.parameters.ControlCurveIndexParameter][], which returns an integer 
based on the position of the storage in the `"Central Reservoir"` storage node with respect to the two 
control curves "level1" and "level2":

- 0 means no demand restrictions. The storage is above the curve in `"level1"`
- 1 (L1) is some restriction (or perhaps just publicity), when the storage is below `"level1"` (80% storage).
- 2 (L2) is for more severe restrictions when the storage is below `"level2"` (50% storage).
 
Control curves are commonly more complicated, as the expected level of a reservoir is usually lower in the summer than it is
in the winter.

## Demand restriction factor (_s_)
The demand restriction factor _s_ is determined by indexing an array of possible 
restriction profiles with `"demand_restriction_level"` using a [pywr.parameters.IndexedArrayParameter][]. In the example 
the list of profiles (in the `"params"` keys) corresponds to the L0, L1 and
L2 profiles respectively:

- At L0 a constant factor `1.0` is used to represent no restrictions. 
- At L1 there is a 10%  reduction in demand (`0.90` as a factor) during the summer months and a 5% reduction elsewhere (`0.95`). 
- At L2 there are further reductions to 75% or 80% depending on the month.

To understand how the index works, the following equivalent Python code may help:

```python
month = 5
demand_restriction_level = 1  # this can change between 0 and 2
demand_factors = [
    [1.0]*12, 
    [0.95, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95], 
    [0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8]
]
demand_restriction_factor = demand_factors[demand_restriction_level][month - 1]
```

## Final demand
Finally, the demand components can be combined as in the equation at the beginning using a [pywr.parameters.AggregatedParameter][]:

```json
{
  "parameters": {
    "demand_max_flow": {
      "type": "aggregated",
      "agg_func": "product",
      "parameters": [
        "demand_baseline",
        "demand_profile",
        "demand_restriction_factor"
      ]
    }
  }
}
```

At each timestep the value of each of the components is calculated, and the values are multiplied to give the final demand value.
The actual demand value will switch between the three profiles below depending on the resource state of the reservoir.

```python exec="1" html="1"
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
import pandas

baseline = 50.0

profile = np.array([0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2, 0.9, 0.9, 0.9, 0.9])

L0 = np.array([1.0] * 12)
L1 = np.array([0.95, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95])
L2 = np.array([0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8])

dates = pandas.date_range("2015-01-01", "2015-12-31")

L0_values = np.array([baseline * profile[d.month - 1] * L0[d.month - 1] for d in dates])
L1_values = np.array([baseline * profile[d.month - 1] * L1[d.month - 1] for d in dates])
L2_values = np.array([baseline * profile[d.month - 1] * L2[d.month - 1] for d in dates])

fig, ax = plt.subplots()

ax.plot(np.arange(0, len(dates)), L0_values, linewidth=2)
ax.plot(np.arange(0, len(dates)), L1_values, linewidth=2)
ax.plot(np.arange(0, len(dates)), L2_values, linewidth=2)

ax.set_xlabel("Day of year")
ax.set_ylabel("Demand [Ml/d]")

ax.grid(True)
ax.set_ylim(0.0, 70)
ax.set_xlim(0, 365)
ax.legend(["Level 0", "Level 1", "Level 2"], loc="lower right")
plt.tight_layout()


buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

This parameter can then be applied to the ``max_flow`` attribute a demand node:

```json
{
  "nodes": [
    {
      "type": "output",
      "name": "Demand",
      "max_flow": "demand_max_flow",
      "cost": -500
    }
  ]
}
```

When a model has more than one demand node, you can re-use the demand restriction level/factor for each demand node.
Pywr will only calculate the index once for each parameter.
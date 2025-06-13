# Annual profiles
Profile parameters return a repeating pattern of values. The following table summarises
all of them:

| Type                                                  | Resolution | Scenario-specific | Notes                                    |
|-------------------------------------------------------|------------|-------------------|------------------------------------------|
| [DailyProfileParameter](#daily-profile)               | Daily      | No                | 366 values, repeating annually           |
| [WeeklyProfileParameter](#weekly-profile)             | Weekly     | No                | 52 values                                |
| [MonthlyProfileParameter](#monthly-profile)           | Monthly    | No                | 12 values                                |
| [UniformDrawdownProfileParameter](#uniform-drawdown)  | Daily      | No                | Linearly from 1 to 0                     |
| [ScenarioDailyProfileParameter](#scenario-profiles)   | Daily      | Yes               | 2D array                                 |
| [ScenarioWeeklyProfileParameter](#scenario-profiles)  | Weekly     | Yes               | 2D array                                 |
| [ScenarioMonthlyProfileParameter](#scenario-profiles) | Monthly    | Yes               | 2D array                                 |
| [RbfProfileParameter](#radial-basis-function-profile) | Daily      | No                | Interpolated using radial basis function |
| WeightedAverageProfileParameter                       | Daily      | No                | Weighted by storage volumes              |

## Daily profile
The [pywr.parameters.DailyProfileParameter][] defines a repeating annual profile of 366 values 
with a daily resolution. This is ideal for modeling daily demand patterns or specific operational rules that change on a daily basis.

=== "Python"
    ```python
    import numpy as np
    from pywr.model import Model
    from pywr.parameters import DailyProfileParameter
      
    model = Model()
    DailyProfileParameter(
        model=model,
        name="Random data", 
        # generate a random sequence between 1 and 100 of 366 values
        values=np.random.rand(366)
    )
    ```

=== "JSON"

    ```json
    {
        "Random data": {
            "type": "DailyProfileParameter",
            "values": [1, 34, ..., 100]
        }
    }
    ```

!!!warning "Profile size"
    You must provide 366 values to account for leap years. The profile will throw an error
    if you provide a profile with length different from 366.

## Weekly profile
The [pywr.parameters.WeeklyProfileParameter][] defines a repeating annual profile of 52 values 
with a weekly resolution. 

=== "Python"
    ```python
    import numpy as np
    from pywr.model import Model
    from pywr.parameters import WeeklyProfileParameter
      
    model = Model()
    WeeklyProfileParameter(
        model=model,
        name="Random data", 
        # generate a random sequence of 52 values
        values=np.random.rand(52)
    )
    ```

=== "JSON"

    ```json
    {
        "Random data": {
            "type": "WeeklyProfileParameter",
            "values": [1, 34, ..., 100]
        }
    }
    ```

!!!info "Current week"
    The current week is calculated using the current day of the year (starting from 0) and 
    applying a floor division, which divides the day by 7 and rounds down to the nearest 
    whole number.

!!!warning "Profile size"
    You must provide 52 values; if the list has 53 values, it is truncated to 52. The profile will throw an
    error if you provide a profile with a size different from 52 or 53. The last week of
    the year will have more than 7 days, as 365 / 7 is not whole.

## Monthly profile
A [pywr.parameters.MonthlyProfileParameter][] is defined by a list or array of 12 values, where
each value corresponds to a month of the year, starting with January.

In this example, the demand will be `1` in January, `9` in February, and so on:

=== "Python"
    ```python
    from pywr.model import Model
    from pywr.parameters import MonthlyProfileParameter
      
    model = Model()
    MonthlyProfileParameter(
        model=model,
        name="Profile",
        values=[1.0, 9.0, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0]
    )
    ```

=== "JSON"

    ```json
    {
        "Random data": {
            "type": "MonthlyProfileParameter",
            "values": [1, 9, 45, 23, 120, 190, 300, 900, 200, 101, 32, 12]
        }
    }
    ```

By default, this creates a **piecewise profile** with a step change at the beginning of each month. 
This is shown in the first panel in the figure below.

<p align="center">

```python exec="1" html="1"
# generate profile chart by running a dummy model
import pandas as pd
from io import StringIO

from pywr.model import Model
from pywr.nodes import Output, Input
from pywr.parameters import MonthlyProfileParameter
import matplotlib.pyplot as plt
from pywr.recorders import NumpyArrayParameterRecorder

dfs = []
model = Model()
p = MonthlyProfileParameter(
    model=model,
    name="Profile",
    values=[1, 9, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0],
)
Input(model, name="I").connect(Output(model, name="O"))

r = NumpyArrayParameterRecorder(model, p)
model.run()
del model
df = r.to_dataframe()
dfs.append(df)

ax = pd.concat(dfs, axis=1).plot()
ax.set_ylabel("Profile value")
ax.get_legend().remove()

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>

### Interpolation
An optional `interp_day` parameter can be passed to the parameter to create a linearly piecewise-interpolated daily 
profile.

=== "Python"
    ```python
    from pywr.model import Model
    from pywr.parameters import MonthlyProfileParameter
      
    model = Model()
    MonthlyProfileParameter(
        model=model,
        name="Profile",
        interp_day="first",
        values=[1.0, 9.0, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0]
    )
    ```

=== "JSON"

    ```json
      {
        "Random data": {
            "type": "MonthlyProfileParameter",
            "interp_day": "first",
            "values": [1, 9, 45, 23, 120, 190, 300, 900, 200, 101, 32, 12]
        }
    }
    ```

When you set the value to 

- `"first"`, the profile starts on the first of the month: the first month is interpolated between `1` and `9`; the second month between `9` and `45` and so on.
- `"last"`, the profile starts on the last of the month: the first month is interpolated between `12` and `1`; the second month between `1` and `9` and so on.

These are the resulting profiles:

<p align="center">

```python exec="1" html="1"
# generate profile chart by running a dummy model
import pandas as pd
from io import StringIO

from pywr.model import Model
from pywr.nodes import Output, Input
from pywr.parameters import MonthlyProfileParameter
import matplotlib.pyplot as plt
from pywr.recorders import NumpyArrayParameterRecorder

dfs = []
options = ["first", "last"]
for interp_day in options:
    model = Model()
    p = MonthlyProfileParameter(
        model=model,
        name="Profile",
        values=[1, 9, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0],
        interp_day=interp_day,
    )
    Input(model, name="I").connect(Output(model, name="O"))

    r = NumpyArrayParameterRecorder(model, p)
    model.run()
    del model
    df = r.to_dataframe()
    df.columns = [str(interp_day)]
    dfs.append(df)

ax = pd.concat(dfs, axis=1).plot()
ax.set_ylabel("Profile value")
ax.set_title("Interpolated")

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>


## Uniform drawdown
This parameter provides a uniformly reducing value from one to zero. It returns a value of 1 on the
 reset day, and subsequently reduces by 1/366 every day afterward. 
 
=== "Python"
    ```python
    from pywr.model import Model
    from pywr.parameters import UniformDrawdownProfileParameter
      
    model = Model()
    UniformDrawdownProfileParameter(model=model, name="Profile")
    ```

=== "JSON"

    ```json
      {
        "Profile": {
            "type": "UniformDrawdownProfileParameter"
        }
    }
    ```

By default, the profile starts from `1` on the first of January:


<p align="center">

```python exec="1" html="1"
# generate profile chart by running a dummy model
from io import StringIO

from pywr.model import Model
from pywr.nodes import Output, Input
from pywr.parameters import UniformDrawdownProfileParameter
import matplotlib.pyplot as plt
from pywr.recorders import NumpyArrayParameterRecorder

model = Model()
p = UniformDrawdownProfileParameter(
    model=model,
    name="Profile",
)
Input(model, name="I").connect(Output(model, name="O"))

r = NumpyArrayParameterRecorder(model, p)
model.run()
del model

ax = r.to_dataframe().plot()
ax.set_ylabel("Profile value")

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>


!!!info
    This parameter is intended to be used with a [pywr.nodes.AnnualVirtualStorage][] node to provide a profile
    that represents perfect average utilisation of the annual volume. See [the cost section](../costs.md#annual-license)
    to understand how to combine the two components.

### Change the reset date
You can change the date when the profile starts from `1` (this is the reset date), by providing
the `"reset_day"` and `"reset_month"` options. For example to reset the profile on 1st April you
can use:

 
=== "Python"
    ```python
    from pywr.model import Model
    from pywr.parameters import UniformDrawdownProfileParameter
      
    model = Model()
    UniformDrawdownProfileParameter(
        model=model,
        name="Profile",
        reset_day=1,
        reset_month=4
    )
    ```

=== "JSON"

    ```json
      {
        "Profile": {
            "type": "UniformDrawdownProfileParameter",
            "reset_day": 1,
            "reset_month": 4
        }
    }
    ```

<p align="center">

```python exec="1" html="1"
# generate profile chart by running a dummy model
from io import StringIO

from pywr.model import Model
from pywr.nodes import Output, Input
from pywr.parameters import UniformDrawdownProfileParameter
import matplotlib.pyplot as plt
from pywr.recorders import NumpyArrayParameterRecorder

model = Model()
p = UniformDrawdownProfileParameter(
    model=model,
    name="Profile",
    reset_day=1,
    reset_month=4,
)
Input(model, name="I").connect(Output(model, name="O"))

r = NumpyArrayParameterRecorder(model, p)
model.run()
del model

ax = r.to_dataframe().plot()
ax.set_ylabel("Profile value")

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>

## Scenario profiles
Pywr also provides scenario-specific versions of the daily, monthly and weekly profile parameters, 
allowing for different profiles to be used under model scenarios. 
These are [pywr.parameters.ScenarioMonthlyProfileParameter][], [pywr.parameters.ScenarioDailyProfileParameter][],
and [pywr.parameters.ScenarioWeeklyProfileParameter][].

These parameters are defined similarly to their non-scenario counterparts, but their `values` argument is
structured to accept a 2D numpy array. For example, if you have two scenarios, you
can define the monthly profile to use in each scenario as:


=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model, Scenario
    from pywr.parameters import ScenarioMonthlyProfileParameter
      
    model = Model()
    scenario = Scenario(
        model=model,
        name="Demand", 
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    ScenarioMonthlyProfileParameter(
        model=model,
        name="Profile",
        scenario=scenario,
        values=np.array(
            [
                [1.0, 9.0, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0],
                [10.5, 11.2, 12.8, 14.5, 16.2, 18.0, 17.5, 16.8, 15.0, 13.5, 12.0, 11.0],
            ]
        ),  
    )
    ```

=== "JSON"

    ```json
      {
        "scenarios": [
            {
                "name": "Demand", 
                "ensemble_names": ["Low demand", "High demand"],
                "size": 2
            },
        ],
        "parameters": {
            "Profile": {
                "type": "ScenarioMonthlyProfileParameter",
                "scenario": "Demand",
                "values": [
                    [1.0, 9.0, 45.0, 23.0, 120.0, 190.0, 30.0, 90.0, 200.0, 101.0, 32.0, 12.0],
                    [10.5, 11.2, 12.8, 14.5, 16.2, 18.0, 17.5, 16.8, 15.0, 13.5, 12.0, 11.0],
                ]
            }
        }
    }
    ```

When Pywr runs the first scenario `"Low demand"`, the first array is used; otherwise the second
is used. The number of profiles (or first array dimension) must equal the scenario size. 

!!!info "Monthly interpolation"
    The `ScenarioMonthlyProfileParameter` does not accept the `interp_day` parameter
    as the `MonthlyProfileParameter`.

## Radial basis function profile
The [pywr.parameters.RbfProfileParameter][] interpolates a daily profile using a [radial basis function (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function)
and is mainly used in optimisation problems. The user can define the profile nodes (at least three 
points are needed) where:

- the independent variable is the day of the year. The first day must be 1 and the days between 1 and 365 and
strictly monotonically increasing.
- the dependent variable array must have the same size; the first value is repeated as 
the 366<sup>th</sup> value.

The following chart shows a profile created with three nodes, represented as dots, located at 
x=[1, 30, 120] and y=[0.4, 0.6, 0.7].


=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.parameters import RbfProfileParameter
      
    model = Model()
    RbfProfileParameter(
        model=model,
        name="Profile",
        days_of_year=[1, 30, 120],
        values=[0.4, 0.6, 0.7]
    )
    ```

=== "JSON"

    ```json
    {
        "Profile": {
            "type": "RbfProfileParameter",
            "days_of_year": [1, 30, 120] 
            "values": [0.4, 0.6, 0.7]
        }
    }
    ```

<p align="center">

```python exec="1" html="1"
# generate profile chart by running a dummy model
from io import StringIO

from pywr.model import Model
from pywr.nodes import Output, Input
from pywr.parameters import RbfProfileParameter
import matplotlib.pyplot as plt
from pywr.recorders import NumpyArrayParameterRecorder

model = Model()
days_of_year=[1, 30, 120]
values=[0.4, 0.6, 0.7]
p = RbfProfileParameter(
    model=model, name="Profile", days_of_year=days_of_year, values=values
)
Input(model, name="I").connect(Output(model, name="O"))

r = NumpyArrayParameterRecorder(model, p)
model.run()
del model

df = r.to_dataframe()
ax = df.plot()
ax.plot(df.index[days_of_year], values, "r.", markersize=10)
ax.set_ylabel("Profile value")
ax.set_ylim([0, 1])
ax.get_legend().remove()

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>


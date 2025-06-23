# Solvers
Pywr solves the allocation programme using a [Linear Programme](./key_concepts/problem.md) to determine
the flows at each timestep. The LP matrix is created during the model setup stage, and its structure cannot
change between timesteps. Changes to the objective coefficients and constraints bounds are allowed by 
using parameters, but the connectivity of the network remains constant during a simulation (i.e., you cannot
add or remove edges).

The LP matrix can be solved using either the [GLPK](https://www.gnu.org/software/glpk/)
or the [LpSolve](https://sourceforge.net/projects/lpsolve/) solver. 

With the GLPK solver, Pywr implements two types of linear programme:

- GLPK (route-based): this creates a flow variable for each path between a [pywr.nodes.Input][]
and [pywr.nodes.Output][] node in the network.
- GLPK (edge-based): this creates a variable for each edge in the network, connecting any node.

## Change the solver
There are three different ways you can set the type of solver you can use to run a model. Each option
accepts the following string for the solver name: `glpk`, `glpk-edge`, `lpsolve`.

!!!info "Default value"
    The default solver is always set to `glpk`.

### Python
When you load the model from JSON, you can set the `solver` option:

```python
from pywr.core import Model
model = Model.load("my_model.json", solver="glpk-edge")
```

### JSON document
In a JSON document you can set the `solver` key, whose value must be a dictionary with the solver name:

```json
{
  "solver": {
    "name": "lpsolve"
  }
}
```

### Environment variable
Otherwise you can set the following environment variable using Python or from the OS:

```python
import os
os.environ["PYWR_SOLVER"] = "glpk-edge"
```

## Change the solver options
The following options can be set for the glpk-based solvers, either as key in the `solver` dictionary
of the JSON file or as environment variables:

| JSON key               | Env variable                        | Default |                                                                                                                                                                                                                    |
|------------------------|-------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| set_fixed_flows_once   | PYWR_SOLVER_GLPK_FIXED_FLOWS_ONCE   | False   | When true and the node has a constant flow (i.e., both `min_flow` and `max_flow` are constants or [pywr.parameters.ConstantParameter][], the nodes constraints are updated only at reset instead at each timestep. |
| set_fixed_costs_once   | PYWR_SOLVER_GLPK_FIXED_COSTS_ONCE   | False   | Same as above but for a node's cost.                                                                                                                                                                               |
| set_fixed_factors_once | PYWR_SOLVER_GLPK_FIXED_FACTORS_ONCE | False   | Determine whether the factors of aggregated nodes should be updated in reset or at each time-step                                                                                                                  |
| use_unsafe_api         | PYWR_SOLVER_GLPK_UNSAFE_API         | False   | When usafe, Pywr does not perform data checks or error handling. See [this page for more information](./key_concepts/exceptions.md#version-117-and-above)                                                          |

All options accept a boolean value. The `lpsolve` solver has no option. 

## Access solver statistics
To access the solver statistics after a simulation, you can access the `solver_stats` property,
which contains a dictionary:

```python
model = Model()
...
stats = model.run()

print(stats.solver_stats["number_of_cols"])
print(stats.solver_stats["number_of_rows"])
```
 
The dictionary contains the following keys:

| Key                       | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| constraint_update_factors | Time to update constraint matrix values for aggregated nodes that have factors defined as parameters. |
| bounds_update_nonstorage  | Time to update non-storage properties ad aggregated nodes.                                            |
| bounds_update_storage     | Time to update storage node constraint and virtual storage node constraint.                           |
| objective_update          | Time to update the cost of each node in the model.                                                    |
| number_of_rows            | Matrix number of rows.                                                                                |
| number_of_cols            | Matrix number of columns.                                                                             |                                                                                                       |
| number_of_nonzero         | The number of non-zero elements in the constraint matrix                                              |                                                                                                       |
| number_of_routes          | Number of routes.                                                                                     |
| number_of_nodes           | Total number of nodes.                                                                                |

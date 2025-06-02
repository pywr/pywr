# Output

| What it does         | <span style="font-weight:normal;">This node removes water from the network using the `max_flow`, `min_flow` and `cost` options.</span> |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node from demand centres, zonal exports, termination or spill nodes.                                                          |
| **Pywr class**       | [pywr.nodes.Output][]                                                                                                                  |
                                                                                                                                  

## Available key options

| Name     | Description                                                | Required | Default value |
|----------|------------------------------------------------------------|----------|---------------|
| min_flow | Set the minimum amount of water that the nodes must remove | No       | 0             |
| max_flow | Set the maximum amount of water that the nodes must remove | No       | Inf           |
| cost     | The penality cost per unit flow via the node               | No       | 0             |


## Example
This is an example of an output used as demand centre:
```json
{
  "name": "Cardiff East",
  "type": "output",
  "max_flow": "Demand profile",
  "cost": -500
}
```
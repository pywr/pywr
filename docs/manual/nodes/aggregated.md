# Aggregated

## Aggregated node

| What it does         | <span style="font-weight:normal;">This node allows you to sum the flow delivered by other nodes in the network and limit the combined flow using the `max_flow`, `min_flow` and `cost` options.</span> |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to set constraints on the combined flow of a group of (possibly disconnected) nodes (for example for daily licenses from different sources).                                             |
| **Pywr class**       | [pywr.nodes.AggregatedNode][]                                                                                                                                                                          |

!!!info "Virtual node"
    This is a virtual node and cannot be connected to the network as a normal node.

### Available key options  
  
| Name          | Description                                                | Required | Default value |
|---------------|------------------------------------------------------------|----------|---------------|
| min_flow      | Set the minimum amount of water that the node must deliver | No       | 0             |
| max_flow      | Set the maximum amount of water that the node must deliver | No       | Inf           |
| storage_nodes | The list of node names to aggregated the flow of           | Yes      | -             |
| factors       | Scale the flow by the given weights                        | No       | []            |
 
If you set the `factors` property, this must be a list of size equal to the number of nodes in `nodes`.
The `factors`, `min_flow` and `max_flow` attributes can all be specified at the same time to 
constrain both the ratio, minimum and maximum flow via a group of nodes.

!!!danger "Hard constraints"
    Note that the constraint enforced by aggregated nodes is a "hard" 
    constraint; it must always be satisfied. This can result in complex and sometimes
    unintended behaviours.

### Examples  

#### Generic flow constraint
The aggregated node can be used to constrain the total flow via a group of nodes. This is useful, for example, in
abstraction schemes where the combined license for a group of sources is less than the sum of their individual licences.


In the example below, the aggregated node "D" ensures the total flow from 
A and B combined does not exceed 60. The `max_flow` attribute on the 
node "D" could also be a `Parameter` definition or reference.

```json
{
  "nodes": [
    {
      "name": "A",
      "type": "input",
      "max_flow": 30.0
    },
    {
      "name": "B",
      "type": "input",
      "max_flow": 40.0
    },
    {
      "name": "D",
      "type": "aggregated",
      "nodes": [
        "A",
        "B"
      ],
      "max_flow": 60.0
    }
  ]
}
```

#### Constraining flow using a ratio
Aggregated nodes allow a constraint to be added that ensures the flow via two or
more nodes conforms to a specific ratio. This is useful, for example, when 
modelling a blending constraint between multiple sources of water.

In the example below the aggregated node "D" constrains the flow in nodes "A" and "B" to be 
equal (`0.5 + 0.5 == 1`). Due to the constraint, the solution is flow from
A = 40, B = 40. There is no requirement that the
factors sum to `1.0` this example would work with factors of `50` and `50` 
(or any two equal numbers) instead.


```json
{
  "nodes": [
    {
      "name": "A",
      "type": "input"
    },
    {
      "name": "B",
      "type": "input",
      "max_flow": 40.0
    },
    {
      "name": "C",
      "type": "output",
      "max_flow": 100.0,
      "cost": -10.0
    },
    {
      "name": "D",
      "type": "aggregated",
      "nodes": [
        "A",
        "B"
      ],
      "factors": [
        0.5,
        0.5
      ]
    }
  ],
  "edges": [
    [
      "A",
      "C"
    ],
    [
      "B",
      "C"
    ]
  ]
}
```

Time-varying factors can also be used by referencing other parameters instead of using literal numbers as 
in the example.

More than two nodes can be included in the constraint. In the example below,
three nodes (A, B and F) are constrained to the ratio of `20%`, `30% `and `50%` respectively.

```json
{
    "name": "E",
    "type": "aggregated",
    "nodes": ["A", "B", "F"],
    "factors": [0.2, 0.3, 0.5]
}
```

## Aggregated storage

| What it does         | <span style="font-weight:normal;">This node sums the storages of storage or reservoir nodes and calculates the absolute and relative storage of the combined node.</span> |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to group reservoirs.                                                                                                                                        |
| **Pywr class**       | [pywr.nodes.AggregatedStorage][]                                                                                                                                          |

!!!info "Virtual node"
    This is a virtual node and cannot be connected to the network as a normal node.
                                                                                                                           
### Available key options  
  
| Name          | Description                                                | Required | Default value |
|---------------|------------------------------------------------------------|----------|---------------|
| storage_nodes | The list of storage node names to aggregated the volume of | Yes      | -             |
                                                                                     

## Example  
This is an example of a node to calculate the combined storage of two reservoirs:  
```json 
{  
  "name": "Combined storage",  
  "type": "AggregatedStorage",  
  "storage_nodes": [  
    "Reservoir 1",  
    "Reservoir 2"
  ]
}
```
# Aggregated

## Aggregated node

| What it does         | <span style="font-weight:normal;">This node allows you to sum the flow delivered by other nodes in the network and limit the combined flow using the `max_flow`, `min_flow` and `cost` options.</span> |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to apply a daily license using the abstraction from different sources or calculate the flow through different nodes.                                                                     |
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

### Example  
This is the node used to restrict the abstraction of a combined license for two intakes based on the flow gauged 
at gauging station:  
```json  
{
  "name": "Combined Daily Licence",
  "type": "aggregatednode",
  "max_flow": 10,
  "nodes": [
    "Intake 1",
    "Intake 2"
  ]
}
```

## Aggregated storage

| What it does         | <span style="font-weight:normal;">This node sums the storages of storage or reservoir nodes and calculates the absolute and relative storage of the combined nodes.</span> |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to group reservoirs.                                                                                                                                         |
| **Pywr class**       | [pywr.nodes.AggregatedStorage][]                                                                                                                                           |

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
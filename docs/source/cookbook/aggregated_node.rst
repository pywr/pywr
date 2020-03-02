Aggregated nodes
----------------

The `AggregatedNode` is a special node that allows constraints to be added to a model that apply to a group of (possibly disconnected) nodes, enforcing either a fixed ratio of flow and/or a total minimum or maximum flow.

Constraining flow using a ratio
===============================

Aggregated nodes allows a constraint to be added that ensures the flow via two or more nodes conforms to a specific ratio. This is useful, for example, when modelling a blending constraint between multiple sources of water.

In the example below the aggregated node "D" constrains the flow in nodes "A" and "B" to be equal (0.5 + 0.5 == 1). Due to the constraint the solution is flow from A = 40, B = 40, despite C demanding 100. There is no requirement that the factors sum to 1.0 this example would work with factors of 50 and 50 (or any two equal numbers) instead.

Note that the aggregated node "D" is not connected to any nodes via an edge. The constraint is applied regardless. The factors must also remain a constant throughout a simulation (i.e. they can not be `Parameter` definitions references). This limitation could be resolved in a future version of Pywr.

.. code-block:: javascript

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
            "nodes": ["A", "B"],
            "factors": [0.5, 0.5]
        }
    ],
    "edges": [
        ["A", "C"],
        ["B", "C"]
    ]

More than two nodes can be included in the constraint. In the example below three nodes (A, B and F) are constrained to the ratio of 20%-30%-50% respectively.

.. code-block:: javascript

    {
        "name": "E",
        "type": "aggregated",
        "nodes": ["A", "B", "F"],
        "factors": [0.2, 0.3, 0.5]
    }

Constraining flow using a maximum value
=======================================

The aggregated node can also be used to constrain the total flow via a group of nodes. This is useful, for example, in abstraction schemes where the combined license for a group of sources is less than the sum of their individual licences.

In the example below, the aggregated node "D" ensures the total flow from A and B combined does not exceed 60. The `max_flow` attribute on the node "D" could also be a `Parameter` definition or reference.

.. code-block:: javascript

    "nodes": [
        {
            "name": "A",
            "type": "input"
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
            "nodes": ["A", "B"],
            "max_flow": 60.0
        }
    ]

Additional information
======================

The `factors`, `min_flow` and `max_flow` attributes can all be specified for a single aggregated node to constrain both the ratio, minimum and maximum flow via a group of nodes.

Note that the constraint enforced by aggregated nodes is a "hard" constraint; it must be satisfied. This can result in complex and sometimes unintended behaviours.

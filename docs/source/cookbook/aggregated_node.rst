Aggregated nodes
----------------

The `AggregatedNode` is a special node that allows constraints to be added to a model that apply to a group of (possibly disconnected) nodes, enforcing either a fixed ratio or fixed maximum.

Constraining flow using a ratio
===============================

Aggregated nodes allows a constraint to be added that ensures the flow via two or more nodes conforms to a specific ratio. This is useful, for example, when modelling a blending constraint between multiple sources of water.

In the example below the aggregated node "D" constrains the flow in nodes "A" and "B" to be equal (0.5 + 0.5 == 1). Due to the constraint the solution is flow from A = 40, B = 40, despite C demanding 100.

Note that the aggregated node "D" is not connected to any nodes via an edge. The constraint is applied regardless.

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

In the example below, the aggregated node "D" ensures the total flow from A and B combined does not exceed 60.

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

The `factors` and `max_flow` parameters can both be specified for a single aggregated node to constrain both the ratio and maximum flow via a group of nodes.

Note that the constraint enforced by aggregated nodes is a "hard" constraint; it must be satisfied. This can result in complex and sometimes unintended behaviours.

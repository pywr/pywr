import tables
import numpy as np
from collections import defaultdict


def routes_to_sankey_links(
    filename,
    node_name,
    where="/",
    routes="/routes",
    rename_func=None,
    flow_threshold=0.0,
    ignore_circular=False,
    callback_func=None,
    **kwargs,
):
    """Function to return a list of links suitable to draw a Sankey diagram"""
    time_slice = kwargs.pop("time_slice", None)
    time_agg_func = kwargs.pop("time_agg_func", np.mean)

    scenario_slice = kwargs.pop("scenario_slice", None)
    scenario_agg_func = kwargs.pop("scenario_slice", np.mean)

    sources = defaultdict(lambda: defaultdict(lambda: 0.0))

    with tables.open_file(filename, mode="r") as h5:

        flows = h5.get_node(where, node_name)

        # Apply time slice and aggregation
        if time_slice is not None:
            flows = flows[time_slice, ...]

        if time_agg_func is not None and not isinstance(time_slice, int):
            flows = time_agg_func(flows, axis=0)

        if scenario_slice is not None:
            flows = flows[:, scenario_slice]

        if scenario_agg_func is not None:
            # Aggregate everything but the first axis
            axis = tuple(range(1, flows.ndim))
            flows = scenario_agg_func(flows, axis=axis)

        routes = h5.get_node(routes)

        assert flows.shape[0] == routes.shape[0]
        assert flows.ndim == 1

        for i in range(flows.shape[0]):
            row = routes[i]
            start = row["start"].decode("utf-8")
            end = row["end"].decode("utf-8")

            if callback_func is not None:
                ret = callback_func(start, end, flows[i])
                if ret is None:
                    continue

            if rename_func is not None:
                start = rename_func(start, True)
                end = rename_func(end, False)

            if ignore_circular and start == end:
                continue

            sources[start][end] += flows[i]

    links = []
    for source, targets in sources.items():
        for target, value in targets.items():
            if value < flow_threshold:
                continue
            links.append({"source": source, "target": target, "value": value})
    return links

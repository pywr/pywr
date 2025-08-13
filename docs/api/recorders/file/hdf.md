# Table (HDF)

::: pywr.recorders.TablesRecorder
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        # Hide unused methods
        - "!^agg_func$"
        - "!^constraint_lower_bounds$"
        - "!^constraint_upper_bounds$"
        - "!^is_constraint$"
        - "!^is_double_bounded_constraint$"
        - "!^is_equality_constraint$"
        - "!^is_lower_bounded_constraint$"
        - "!^is_objective$"
        - "!^is_constraint_violated$"
        - "!^is_upper_bounded_constraint$"
        - "!^aggregated_value$"
        - "!^values$"


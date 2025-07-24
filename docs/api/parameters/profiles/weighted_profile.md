# Weighted average

::: pywr.parameters.WeightedAverageProfileParameter
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        # Hide unused/private methods/attrs
        - "!^get_.*_variables$"
        - "!^set_.*_variables$"
        - "!^get_.*_bounds$"
        - "!^get_constant_value$"
        - "!^daily_values$"
        - "!^get_daily_values$"

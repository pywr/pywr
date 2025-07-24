# Other model's parameter index

::: pywr.parameters.OtherModelIndexParameterValueIndexParameter
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        # Hide unused methods
        - "!^get_.*_variables$"
        - "!^set_.*_variables$"
        - "!^get_.*_bounds$"
        - "!^get_constant_value$"
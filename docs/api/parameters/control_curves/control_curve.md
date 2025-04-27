# Control curve

::: pywr.parameters.ControlCurveParameter
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        # Hide unused methods
        - "!^get_integer_variables$"
        - "!^set_integer_variables$"
        - "!^get_integer_.*_bounds$"
        - "!^get_constant_value$"
        # Hide private methods
        - "!^_load_storage_node$"
        - "!^_load_control_curves$"
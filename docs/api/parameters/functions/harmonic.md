# Annual harmonic series

::: pywr.parameters.AnnualHarmonicSeriesParameter
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
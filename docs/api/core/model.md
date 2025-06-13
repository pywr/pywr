# Model
::: pywr.core.Model
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        - "!^_load.*$"
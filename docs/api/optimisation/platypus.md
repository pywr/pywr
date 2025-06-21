# Platypus wrapper

::: pywr.optimisation.platypus.PlatypusWrapper

::: pywr.optimisation.platypus.PywrRandomGenerator
    options:
      filters:
        - "!^__.*__$"      # Exclude all dunder methods
        - "^__init__$"     # Re-include only __init__
        - "!^_private"    # Optionally hide private methods
        # Hide unused methods
        - "!^_abc_impl$"

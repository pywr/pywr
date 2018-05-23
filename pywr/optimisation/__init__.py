from ..core import Model


def cache_variable_parameters(model):
    variables = []
    variable_map = [0, ]
    for var in model.variables:
        size = var.double_size + var.integer_size

        if size <= 0:
            raise ValueError('Variable parameter "{}" does not have a size > 0.'.format(var.name))

        variable_map.append(variable_map[-1] + size)
        variables.append(var)

    return variables, variable_map


def cache_constraints(model):
    constraints = []
    for r in model.constraints:
        constraints.append(r)

    return constraints


def cache_objectives(model):
    # This is done to make sure the order is fixed during optimisation.
    objectives = []
    for r in model.objectives:
        objectives.append(r)
    return objectives


# Global variables for individual process to cache the model and some of its data.
# TODO make this a registry using a unique id for each model. This way multiple
# models can be cached.
MODEL = None
VARIABLES = None
VARIABLE_MAP = None
OBJECTIVES = None
CONSTRAINTS = None


class BaseOptimisationWrapper(object):
    """ A helper class for running pywr optimisations with platypus.
    """

    def __init__(self, pywr_model_json, *args, **kwargs):
        super(BaseOptimisationWrapper, self).__init__(*args, **kwargs)
        self.pywr_model_json = pywr_model_json

    # The following properties enable attribute caching when repeat execution of the same model is undertaken.
    @property
    def model(self):
        global MODEL
        if MODEL is None:
            MODEL = self.make_model()
            MODEL.setup()
        return MODEL

    @property
    def model_variables(self):
        global VARIABLES, VARIABLE_MAP
        if VARIABLES is None:
            VARIABLES, VARIABLE_MAP = cache_variable_parameters(self.model)
        return VARIABLES

    @property
    def model_variable_map(self):
        global VARIABLES, VARIABLE_MAP
        if VARIABLES is None:
            VARIABLES, VARIABLE_MAP = cache_variable_parameters(self.model)
        return VARIABLE_MAP

    @property
    def model_objectives(self):
        global OBJECTIVES
        if OBJECTIVES is None:
            OBJECTIVES = cache_objectives(self.model)
        return OBJECTIVES

    @property
    def model_constraints(self):
        global CONSTRAINTS
        if CONSTRAINTS is None:
            CONSTRAINTS = cache_constraints(self.model)
        return CONSTRAINTS

    def make_model(self):
        m = Model.load(self.pywr_model_json)
        # Apply any user defined changes to the model
        self.customise_model(m)
        return m

    def customise_model(self, model):
        pass  # By default there is no customisation.


def clear_global_model_cache():
    """ Clear the module level model cache. """
    global MODEL, VARIABLES, VARIABLE_MAP, OBJECTIVES, CONSTRAINTS
    MODEL = None
    VARIABLES = None
    VARIABLE_MAP = None
    OBJECTIVES = None
    CONSTRAINTS = None

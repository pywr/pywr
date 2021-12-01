from ..core import Model
import uuid
import logging

logger = logging.getLogger(__name__)


def cache_variable_parameters(model):
    variables = []
    variable_map = [
        0,
    ]
    for var in model.variables:
        size = var.double_size + var.integer_size

        if size <= 0:
            raise ValueError(
                'Variable parameter "{}" does not have a size > 0.'.format(var.name)
            )

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


# Global variables for individual processes to cache the model and some of its data.
# The cache is keyed by a UID for each `BaseOptimisationWrapper`
class ModelCache:
    def __init__(self):
        self.model = None
        self.variables = None
        self.variable_map = None
        self.objectives = None
        self.constraints = None


MODEL_CACHE = {}


class BaseOptimisationWrapper(object):
    """A helper class for running pywr optimisations with platypus."""

    def __init__(self, pywr_model_json, *args, **kwargs):
        uid = kwargs.pop("uid", None)
        self.pywr_model_klass = kwargs.pop("model_klass", Model)
        super(BaseOptimisationWrapper, self).__init__(*args, **kwargs)
        self.pywr_model_json = pywr_model_json

        if uid is None:
            uid = uuid.uuid4().hex  # Create a unique ID for caching.
        self.uid = uid
        self.run_stats = None

    # The following properties enable attribute caching when repeat execution of the same model is undertaken.
    @property
    def _cached(self):
        global MODEL_CACHE
        try:
            cache = MODEL_CACHE[self.uid]
        except KeyError:
            model = self.make_model()
            model.setup()

            cache = ModelCache()
            cache.model = model
            cache.variables, cache.variable_map = cache_variable_parameters(model)
            cache.objectives = cache_objectives(model)
            cache.constraints = cache_constraints(model)
            MODEL_CACHE[self.uid] = cache
        return cache

    @property
    def model(self):
        return self._cached.model

    @property
    def model_variables(self):
        return self._cached.variables

    @property
    def model_variable_map(self):
        return self._cached.variable_map

    @property
    def model_objectives(self):
        return self._cached.objectives

    @property
    def model_constraints(self):
        return self._cached.constraints

    def make_model(self):
        m = self.pywr_model_klass.load(self.pywr_model_json)
        # Apply any user defined changes to the model
        self.customise_model(m)
        return m

    def customise_model(self, model):
        pass  # By default there is no customisation.


def clear_global_model_cache():
    """Clear the module level model cache."""
    global MODEL_CACHE
    MODEL_CACHE = {}

from ._parameters import Parameter, IndexParameter
import numpy as np
from scipy import stats


class DistributionParameter(Parameter):
    """ Parameter based on `scipy.stats` distributions.

    This parameter generates a values equal to the size of the associated scenario. Two sampling methods are
    implemented: (1) random sampling, the default or specified with `sampling_method="random"`, uses the `.rvs`
    function to generate random variables, or (2) a sampling of the distribution's CDF, specified with
    `sampling_method="quantiles"`, uses the `.ppf` with linear quantiles between `lower_quantile` and
    `upper_quantile`. The `random_state` is used with the random sampling method. The random sampling method
    will re-sample every time the model is reset (i.e. re-run).

    Parameters
    ==========
    distribution : instance of `scipy.stats` distribution.
        The distribution to use for generating the parameter values.
    scenario : `pywr.core.Scenario`
    sampling_method : str
    random_state : None or int
    """
    def __init__(self, model, distribution, scenario, *args, sampling_method='random', random_state=None,
                 lower_quantile=0.01, upper_quantile=0.99, **kwargs):
        super().__init__(model, *args, **kwargs)

        self.distribution = distribution
        self.scenario = scenario
        self.sampling_method = sampling_method
        self.random_state = random_state
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        # internal variable for storing the sampled variables for each scenario
        self._values = None
        self._scenario_index = None

    def setup(self):
        super().setup()
        self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

    def reset(self):

        nscenarios = self.scenario.size

        if self.sampling_method == 'random':
            # Take random variables
            self._values = self.distribution.rvs(size=nscenarios, random_state=self.random_state)
        elif self.sampling_method == 'quantiles':
            # Linear distribution of quantiles
            quantiles = np.linspace(self.lower_quantile, self.upper_quantile, nscenarios)
            # Generate values
            self._values = self.distribution.ppf(quantiles)
        else:
            raise ValueError('Sampling method "{}" not recognised.'.format(self.sampling_method))

    def value(self, ts, scenario_index):
        return self._values[scenario_index.indices[self._scenario_index]]

    @classmethod
    def load(cls, model, data):
        scenario = model.scenarios[data.pop('scenario')]
        distribution_data = data.pop('distribution')
        distribution_name = distribution_data.pop('name')
        distribution = getattr(stats, distribution_name)(**distribution_data)

        return cls(model, distribution=distribution, scenario=scenario, **data)
DistributionParameter.register()


class RandomFailureIndexParameter(IndexParameter):
    """An IndexParameter that returns random failure state using a probability distribution. """
    def __init__(self, model, mean_time_between_failures, mean_time_to_repair, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.mean_time_between_failures = mean_time_between_failures
        self.mean_time_to_repair = mean_time_to_repair
        # Private state
        self._days_to_next_event = None
        self._current_state = None
        self._failure_dist = None
        self._repair_dist = None

    def setup(self):
        super().setup()
        # Create an array to contain the remainings to failure or repair (event)
        num_comb = len(self.model.scenarios.combinations)
        self._days_to_next_event = np.empty([num_comb], np.float64)
        self._current_state = np.empty([num_comb], np.int)
        # Create exponential distributions for failure & repair
        self._failure_dist = stats.expon(scale=self.mean_time_between_failures)
        self._repair_dist = stats.expon(scale=self.mean_time_to_repair)

    def reset(self):
        # Start the model with a repair just completed. i.e. current failed (1), but repair about to complete.
        # The repair will finish in the first timestep (during before), and generate a new time to failure.
        self._days_to_next_event[...] = -1
        self._current_state[...] = 0

    def _make_next_state(self, current_state: int):
        """Generate the next state and days to next event."""
        if current_state == 1:
            new_state = 0
            days_to_next_event = np.floor(self._repair_dist.rvs(size=1))[0]
            if days_to_next_event < 1:
                new_state, days_to_next_event = self._make_next_state(new_state)
        elif current_state == 0:
            new_state = 1
            days_to_next_event = np.floor(self._failure_dist.rvs(size=1))[0]
            if days_to_next_event < 1:
                new_state, days_to_next_event = self._make_next_state(new_state)
        else:
            raise RuntimeError(f'Current state "{current_state}" is not 0 or 1.')
        return new_state, days_to_next_event

    def before(self):
        # Progress toward next event by the size of the timestep
        dt = self.model.timestepper.current.days
        self._days_to_next_event -= dt

        for i in range(self._current_state.shape[0]):
            if self._days_to_next_event[i] <= 0:
                # Event has happened; i.e. we switch state and generate a time to change state again.
                new_state, days_to_next_event = self._make_next_state(self._current_state[i])
                self._current_state[i] = new_state
                self._days_to_next_event[i] = days_to_next_event

    def index(self, timestep, scenario_index):
        return self._current_state[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        return cls(model, **data)
RandomFailureIndexParameter.register()

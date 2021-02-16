from helpers import load_model
from scipy import stats
import numpy as np
from numpy.testing import assert_allclose


class TestDistributionParameter:
    def test_json_random_run(self):
        """Test a model with a distribution parameter using random sampling. """
        model = load_model("monte_carlo_random.json")

        model.setup()
        expected_values = stats.norm.rvs(loc=15.0, scale=1.0, random_state=12345, size=100)
        supply1 = model.nodes['supply1']

        model.run()
        assert_allclose(supply1.flow, expected_values)

    def test_json_quantiles_run(self):
        """Test a model with a distribution parameter using quantiles. """
        model = load_model("monte_carlo_quantiles.json")

        model.setup()
        quantiles = np.linspace(0.1, 0.9, 101)
        expected_values = stats.norm.ppf(quantiles, loc=15.0, scale=1.0)
        supply1 = model.nodes['supply1']

        model.run()
        assert_allclose(supply1.flow, expected_values)


class TestRandomFailureIndexParameter:
    def test_json_run(self):
        """The test includes interpolation of river water level based on flow"""

        model = load_model("random_failure.json")
        model.run()

        flow = model.recorders['link_flow'].to_dataframe()
        print(flow.head(60))

        df = (flow > 0).sum(axis=0).mean()

        # TODO figure out a suitable numerical test
        print(df)
        print(365 - 25 * 365 / 30)
        print(365 * (1 - 5/29))
        assert False

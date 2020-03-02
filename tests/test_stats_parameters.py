from helpers import load_model


class TestRandomFailureIndexParameter:
    def test_json_run(self):
        """The test includes interpolation of river water level based on flow"""

        model = load_model("random_failure.json")
        model.run()

        flow = model.recorders['link_flow'].to_dataframe()
        print(flow)

        print((flow > 0).sum(axis=0).mean())

        print(365 - 5 * 365 / 25)
        print(365 * (1 - 5/29))

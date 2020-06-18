from pywr.parameters import Parameter, load_parameter
import datetime

class DiscountFactorParameter(Parameter):
    """Parameter that returns the current discount factor based on discount rate and a base year.

    Parameters
    ----------
    discount_rate : discount rate fixed across planning period
    base_year : base year in 'yyyy' format

    """

    def __init__(self, model, discount_rate, base_year):
        super(DiscountFactorParameter, self).__init__(model)
        self.discount_rate = discount_rate
        self.base_year = base_year

    def value(self, discount_rate, base_year):
        ct = self.model.timestepper.current #[0:4]
        cy = ct.year
        cyi = cy - self.base_year
        return 1/pow((1+self.discount_rate), cyi) # return discount factor of current year

    @classmethod
    def load(cls, model, data):
        discount_rate = load_parameter(model, data.pop("value"))
        base_year = load_parameter(model, data.pop("base_year"))
        return cls(model, discount_rate=discount_rate, base_year=base_year)

DiscountFactorParameter.register()
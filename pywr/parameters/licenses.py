#!/usr/bin/env python

import calendar, datetime
from ._parameters import Parameter as BaseParameter
import numpy as np

inf = float("inf")


class License(BaseParameter):
    """Base license class from which others inherit

    This class should not be instantiated directly. Instead, use one of the
    subclasses (e.g. DailyLicense).
    """

    def __new__(cls, *args, **kwargs):
        if cls is License:
            raise TypeError("License cannot be instantiated directly")
        else:
            return BaseParameter.__new__(cls)

    def __init__(self, model, node, **kwargs):
        super(License, self).__init__(model, **kwargs)
        self._node = node

    def resource_state(self, timestep):
        raise NotImplementedError()


class TimestepLicense(License):
    """License limiting volume for a single timestep

    This is the simplest kind of license. The volume available each timestep
    is a fixed value. There is no resource state, as use today does not
    impact availability tomorrow.
    """

    def __init__(self, model, node, amount, **kwargs):
        """Initialise a new TimestepLicense

        Parameters
        ----------
        amount : float
            The maximum volume available in each timestep
        """
        super(TimestepLicense, self).__init__(model, node, **kwargs)
        self._amount = amount

    def value(self, timestep, scenario_index):
        return self._amount

    def resource_state(self, timestep):
        return None


TimestepLicense.register()


# for now, assume a daily timestep
# in the future this will need to be more clever
class DailyLicense(TimestepLicense):
    pass


class StorageLicense(License):
    def __init__(self, model, node, amount, **kwargs):
        """A license with a volume to be spent over multiple timesteps

        This class should not be instantiated directly. Instead, use one of the
        subclasses such as AnnualLicense.

        Parameters
        ----------
        amount : float
            The volume of water available in each period
        """
        super(StorageLicense, self).__init__(model, node, **kwargs)
        self._amount = amount

    def setup(self):
        super(StorageLicense, self).setup()
        # Create a state array for the remaining licence volume.
        self._remaining = np.ones(len(self.model.scenarios.combinations)) * self._amount

    def value(self, timestep, scenario_index):
        return self._remaining[scenario_index.global_id]

    def after(self):
        timestep = self.model.timestepper.current
        self._remaining -= self._node.flow * timestep.days
        self._remaining[self._remaining < 0] = 0.0

    def reset(self):
        self._remaining[...] = self._amount

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        amount = data.pop("amount")
        return cls(model, node, amount=amount, **data)


StorageLicense.register()


class AnnualLicense(StorageLicense):
    """An annual license that apportions remaining volume equally for the rest of the year

    value = (volume remaining) / (days remaining) * (timestep length)

    Parameters
    ----------
    node : Node
        The node that consumes the licence
    amount : float
        The total annual volume for this license
    """

    def __init__(self, *args, **kwargs):
        super(AnnualLicense, self).__init__(*args, **kwargs)
        # Record year ready to reset licence when the year changes.
        self._prev_year = None

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        day_of_year = timestep.dayofyear
        days_in_year = 365 + int(calendar.isleap(timestep.year))
        if day_of_year == days_in_year:
            return self._remaining[i]
        else:
            days_remaining = days_in_year - (day_of_year - 1)
            return self._remaining[i] / days_remaining

    def before(self):
        # Reset licence if year changes.
        timestep = self.model.timestepper.current
        if self._prev_year != timestep.year:
            self.reset()

            # The number of days in the year before the first timestep of that year
            days_before_reset = timestep.dayofyear - 1
            # Adjust the license by the rate in previous timestep. This is needed for timesteps greater
            # than 1 day where the license reset is not exactly on the anniversary
            self._remaining[...] -= days_before_reset * self._node.prev_flow

            self._prev_year = timestep.year


AnnualLicense.register()


class AnnualExponentialLicense(AnnualLicense):
    """An annual license that returns a value based on an exponential function of the license's current state.

    The exponential function takes the form,

    .. math::
        f(t) = \mathit{max_value}e^{-x/k}

    Where :math:`x` is the ratio of actual daily averaged remaining license (as calculated by AnnualLicense) to the
    expected daily averaged remaining licence. I.e. if the license is on track the ratio is 1.0.
    """

    def __init__(self, model, node, amount, max_value, k=1.0, **kwargs):
        """

        Parameters
        ----------
        amount : float
            The total annual volume for this license
        max_value : float
            The maximum value that can be returned. This is used to scale the exponential function
        k : float
            A scale factor for the exponent of the exponential function
        """
        super(AnnualExponentialLicense, self).__init__(model, node, amount, **kwargs)
        self._max_value = max_value
        self._k = k

    def value(self, timestep, scenario_index):
        remaining = super(AnnualExponentialLicense, self).value(
            timestep, scenario_index
        )
        expected = self._amount / (365 + int(calendar.isleap(timestep.year)))
        x = remaining / expected
        return self._max_value * np.exp(-x / self._k)


AnnualExponentialLicense.register()


class AnnualHyperbolaLicense(AnnualLicense):
    """An annual license that returns a value based on an hyperbola (1/x) function of the license's current state.

    The hyperbola function takes the form,

    .. math::
        f(t) = \mathit{value}/x

    Where :math:`x` is the ratio of actual daily averaged remaining license (as calculated by AnnualLicense) to the
    expected daily averaged remaining licence. I.e. if the license is on track the ratio is 1.0.
    """

    def __init__(self, model, node, amount, value, **kwargs):
        """

        Parameters
        ----------
        amount : float
            The total annual volume for this license
        value : float
            The value used to scale the hyperbola function
        """
        super(AnnualHyperbolaLicense, self).__init__(model, node, amount, **kwargs)
        self._value = value

    def value(self, timestep, scenario_index):
        remaining = super(AnnualHyperbolaLicense, self).value(timestep, scenario_index)
        expected = self._amount / (365 + int(calendar.isleap(timestep.year)))
        x = remaining / expected
        try:
            return self._value / x
        except ZeroDivisionError:
            return inf


AnnualHyperbolaLicense.register()

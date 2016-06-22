#!/usr/bin/env python

import calendar, datetime
from ._parameters import Parameter as BaseParameter
import numpy as np

inf = float('inf')

class License(BaseParameter):
    """Base license class from which others inherit

    This class should not be instantiated directly. Instead, use one of the
    subclasses (e.g. DailyLicense).
    """
    def __new__(cls, *args, **kwargs):
        if cls is License:
            raise TypeError('License cannot be instantiated directly')
        else:
            return BaseParameter.__new__(cls)

    def __init__(self, node):
        self._node = node

    def resource_state(self, timestep):
        raise NotImplementedError()


class TimestepLicense(License):
    """License limiting volume for a single timestep

    This is the simplest kind of license. The volume available each timestep
    is a fixed value. There is no resource state, as use today does not
    impact availability tomorrow.
    """
    def __init__(self, node, amount):
        """Initialise a new TimestepLicense

        Parameters
        ----------
        amount : float
            The maximum volume available in each timestep
        """
        super(TimestepLicense, self).__init__(node)
        self._amount = amount

    def value(self, timestep, scenario_index):
        return self._amount

    def resource_state(self, timestep):
        return None


# for now, assume a daily timestep
# in the future this will need to be more clever
class DailyLicense(TimestepLicense):
    pass

class StorageLicense(License):
    def __init__(self, node, amount):
        """A license with a volume to be spent over multiple timesteps

        This class should not be instantiated directly. Instead, use one of the
        subclasses such as AnnualLicense.

        Parameters
        ----------
        amount : float
            The volume of water available in each period
        """
        super(StorageLicense, self).__init__(node)
        self._amount = amount

    def setup(self, model):
        # Create a state array for the remaining licence volume.
        self._remaining = np.ones(len(model.scenarios.combinations))*self._amount

    def value(self, timestep, scenario_index):
        return self._remaining[scenario_index.global_id]

    def after(self, timestep):
        self._remaining -= self._node.flow*timestep.days
        self._remaining[self._remaining < 0] = 0.0

    def reset(self):
        self._remaining[...] = self._amount


class AnnualLicense(StorageLicense):
    """An annual license"""
    def __init__(self, node, amount):
        """
        Parameters
        ----------
        amount : float
            The total annual volume for this license

        """
        super(AnnualLicense, self).__init__(node, amount)
        # Record year ready to reset licence when the year changes.
        self._prev_year = None

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        timetuple = timestep.datetime.timetuple()
        day_of_year = timetuple.tm_yday
        days_in_year = 365 + int(calendar.isleap(timestep.datetime.year))
        if day_of_year == days_in_year:
            return self._remaining[i]
        else:
            return self._remaining[i] / (days_in_year - day_of_year + 1)

    def before(self, timestep):
        # Reset licence if year changes.
        if self._prev_year != timestep.datetime.year:
            self.reset()

            # The number of days in the year before the first timestep of that year
            timetuple = timestep.datetime.timetuple()
            days_before_reset = timetuple.tm_yday - 1
            # Adjust the license by the rate in previous timestep. This is needed for timesteps greater
            # than 1 day where the license reset is not exactly on the anniversary
            self._remaining[...] -= days_before_reset*self._node.prev_flow

            self._prev_year = timestep.datetime.year


class AnnualExponentialLicense(AnnualLicense):
    """ An annual license that returns a value based on an exponential function of the license's current state.

    The exponential function takes the form,

    .. math::
        f(t) = \mathit{max_value}e^{-x/k}

    Where :math:`x` is the ratio of actual daily averaged remaining license (as calculated by AnnualLicense) to the
    expected daily averaged remaining licence. I.e. if the license is on track the ratio is 1.0.
    """
    def __init__(self, node, amount, max_value, k=1.0):
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
        super(AnnualExponentialLicense, self).__init__(node, amount)
        self._max_value = max_value
        self._k = k

    def value(self, timestep, scenario_index):
        remaining = super(AnnualExponentialLicense, self).value(timestep, scenario_index)
        expected = self._amount / (365 + int(calendar.isleap(timestep.datetime.year)))
        x = remaining / expected
        return self._max_value * np.exp(-x / self._k)


class AnnualHyperbolaLicense(AnnualLicense):
    """ An annual license that returns a value based on an hyperbola (1/x) function of the license's current state.

    The hyperbola function takes the form,

    .. math::
        f(t) = \mathit{value}/x

    Where :math:`x` is the ratio of actual daily averaged remaining license (as calculated by AnnualLicense) to the
    expected daily averaged remaining licence. I.e. if the license is on track the ratio is 1.0.
    """
    def __init__(self, node, amount, value):
        """

        Parameters
        ----------
        amount : float
            The total annual volume for this license
        value : float
            The value used to scale the hyperbola function
        """
        super(AnnualHyperbolaLicense, self).__init__(node, amount)
        self._value = value

    def value(self, timestep, scenario_index):
        remaining = super(AnnualHyperbolaLicense, self).value(timestep, scenario_index)
        expected = self._amount / (365 + int(calendar.isleap(timestep.datetime.year)))
        x = remaining / expected
        try:
            return self._value / x
        except ZeroDivisionError:
            return inf

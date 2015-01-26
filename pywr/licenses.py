#!/usr/bin/env python

import calendar

inf = float('inf')

class License(object):
    '''Base license class from which others inherit
    
    This class should not be instantiated directly. Instead, use one of the
    subclasses (e.g. DailyLicense).
    '''
    def __new__(cls, *args, **kwargs):
        if cls is License:
            raise TypeError('License cannot be instantiated directly')
        else:
            return object.__new__(cls, *args, **kwargs)
    def available(self, index):
        raise NotImplementedError()
    def resource_state(self, index):
        raise NotImplementedError()
    def commit(self, value):
        pass
    def refresh(self):
        pass

class DailyLicense(License):
    '''Daily license'''
    def __init__(self, amount):
        self._amount = amount
    def available(self, index):
        return self._amount # assumes daily timestep
    def resource_state(self, index):
        return None

class AnnualLicense(License):
    '''Annual license'''
    def __init__(self, amount):
        self._amount = amount
        self.refresh()
    def available(self, index):
        return self._remaining
    def resource_state(self, index):
        timetuple = index.timetuple()
        day_of_year = timetuple.tm_yday
        days_in_year = 365 + int(calendar.isleap(index.year))
        if day_of_year == days_in_year:
            return inf
        else:
            expected_remaining = self._amount - ((day_of_year-1) * self._amount / days_in_year)
            if expected_remaining == self._amount:
                return 1.0
            else:
                return self._remaining / expected_remaining
    def commit(self, value):
        self._remaining -= value
    def refresh(self):
        self._remaining = self._amount

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
            return object.__new__(cls)
    def available(self, index):
        raise NotImplementedError()
    def resource_state(self, index):
        raise NotImplementedError()
    def commit(self, value):
        pass
    def refresh(self):
        pass

    @classmethod
    def from_xml(cls, xml):
        lic_type = xml.get('type')
        amount = float(xml.text)
        lic_types = {
            'annual': AnnualLicense,
            'daily': DailyLicense,
        }
        lic = lic_types[lic_type](amount)
        return lic

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
            return self._remaining / expected_remaining
    def commit(self, value):
        self._remaining -= value
    def refresh(self):
        self._remaining = self._amount

class LicenseCollection(License):
    '''A collection of Licences'''
    def __init__(self, licenses=None):
        if licenses is None:
            self._licenses = []
        else:
            self._licenses = licenses
            self._licenses = set(self._licenses)
    def add(self, license):
        self._licenses.add(license)
    def remove(self, license):
        self._licenses.remove(license)
    def __len__(self):
        return len(self._licenses)
    def available(self, index):
        min_available = float('inf')
        for license in self._licenses:
            min_available = min(license.available(index), min_available)
        return min_available
    def resource_state(self, index):
        resource_states = []
        for license in self._licenses:
            resource_states.append(license.resource_state(index))
        resource_states = [r for r in resource_states if r is not None]
        return min(resource_states)
    def commit(self, value):
        for license in self._licenses:
            license.commit(value)
    def refresh(self):
        for license in self._licenses:
            license.refresh()

    @classmethod
    def from_xml(cls, xml):
        licenses = set()
        for xml_lic in xml.getchildren():
            license = License.from_xml(xml_lic)
            licenses.add(license)
        return LicenseCollection(licenses)

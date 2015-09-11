#!/usr/bin/env python

import calendar
import xml.etree.ElementTree as ET

inf = float('inf')

class License(object):
    """Base license class from which others inherit

    This class should not be instantiated directly. Instead, use one of the
    subclasses (e.g. DailyLicense).
    """
    def __new__(cls, *args, **kwargs):
        if cls is License:
            raise TypeError('License cannot be instantiated directly')
        else:
            return object.__new__(cls)
    def available(self, timestep):
        raise NotImplementedError()
    def resource_state(self, timestep):
        raise NotImplementedError()
    def commit(self, scenario_index, value):
        pass
    def reset(self):
        pass

    @classmethod
    def from_xml(cls, xml):
        lic_type = xml.get('type')
        amount = float(xml.text)
        lic_types = {
            'annual': AnnualLicense,
            'daily': TimestepLicense,
            'timestep': TimestepLicense,
        }
        lic = lic_types[lic_type](amount)
        return lic

class TimestepLicense(License):
    """License limiting volume for a single timestep

    This is the simplest kind of license. The volume available each timestep
    is a fixed value. There is no resource state, as use today does not
    impact availability tomorrow.
    """
    def __init__(self, amount):
        """Initialise a new TimestepLicense

        Parameters
        ----------
        amount : float
            The maximum volume available in each timestep
        """
        self._amount = amount
    def available(self, timestep):
        return self._amount
    def resource_state(self, timestep):
        return None

    def xml(self):
        xml = ET.Element('license')
        xml.set('type', 'timestep')
        xml.text = str(self._amount)
        return xml

# for now, assume a daily timestep
# in the future this will need to be more clever
class DailyLicense(TimestepLicense):
    pass

class StorageLicense(License):
    def __init__(self, amount):
        """A license with a volume to be spent over multiple timesteps

        This class should not be instantiated directly. Instead, use one of the
        subclasses such as AnnualLicense.

        Parameters
        ----------
        amount : float
            The volume of water available in each period
        """
        super(StorageLicense, self).__init__()
        self._amount = amount
        self.reset()
    def available(self, timestep):
        return self._remaining
    def commit(self, scenario_index, value):
        self._remaining -= value
    def reset(self):
        self._remaining = self._amount

class AnnualLicense(StorageLicense):
    """An annual license"""
    def resource_state(self, timestep):
        timetuple = timestep.datetime.timetuple()
        day_of_year = timetuple.tm_yday
        days_in_year = 365 + int(calendar.isleap(timestep.datetime.year))
        if day_of_year == days_in_year:
            return inf
        else:
            expected_remaining = self._amount - ((day_of_year-1) * self._amount / days_in_year)
            return self._remaining / expected_remaining

    def xml(self):
        xml = ET.Element('license')
        xml.set('type', 'annual')
        xml.text = str(self._amount)
        return xml

class LicenseCollection(License):
    """A collection of Licences

    This object behaves like a set. Licenses can be added to or removed from it.
    The amount of water available in the current timestep is the minimum
    amount available from the child licenses. Similarly, the resource state is
    the minimum of the resource states of the child licenses.
    """
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
    def available(self, timestep):
        min_available = inf
        for license in self._licenses:
            min_available = min(license.available(timestep), min_available)
        return min_available
    def resource_state(self, timestep):
        resource_states = []
        for license in self._licenses:
            resource_states.append(license.resource_state(timestep))
        resource_states = [r for r in resource_states if r is not None]
        return min(resource_states)
    def commit(self, scenario_index, value):
        for license in self._licenses:
            license.commit(scenario_index, value)
    def reset(self):
        for license in self._licenses:
            license.reset()

    def xml(self):
        xml = ET.Element('licensecollection')
        for license in self._licenses:
            xml.append(license.xml())
        return xml

    @classmethod
    def from_xml(cls, xml):
        licenses = set()
        for xml_lic in xml.getchildren():
            license = License.from_xml(xml_lic)
            licenses.add(license)
        return LicenseCollection(licenses)

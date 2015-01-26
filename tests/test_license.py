#!/usr/bin/env python

import pytest
from datetime import datetime

from pywr.licenses import License, DailyLicense, AnnualLicense, LicenseCollection

def test_base_license():
    with pytest.raises(TypeError):
        lic = License()

def test_daily_license():
    '''Test daily licence'''
    
    lic = DailyLicense(42.0)
    assert(isinstance(lic, License))
    assert(lic.available(datetime(2015, 1, 1)) == 42.0)
    
    # daily licences don't have resource state
    assert(lic.resource_state(datetime(2015, 1, 1)) is None)

def test_annual_license():
    '''Test annual license'''
    
    lic = AnnualLicense(365.0)
    assert(isinstance(lic, License))
    assert(lic.available(datetime(2015, 1, 1)) == 365.0)
    assert(lic.resource_state(datetime(2015, 1, 1)) == 1.0)
    
    # use some water and check the remaining decreases
    lic.commit(181.0)
    assert(lic.available(datetime(2015, 1, 1)) == 184.0)
    
    # check resource state
    assert(lic.resource_state(datetime(2015, 7, 1)) == 1.0) # as expected
    assert(lic.resource_state(datetime(2015, 8, 1)) > 1.0) # better than expected
    assert(lic.resource_state(datetime(2015, 6, 1)) < 1.0) # worse than expected
    
    # on last day, resource state is inf
    assert(lic.resource_state(datetime(2015, 12, 31)) == float('inf'))

    # after a refresh, licence is restored to original state
    lic.refresh()
    assert(lic.available(datetime(2015, 1, 1)) == 365.0)
    assert(lic.resource_state(datetime(2015, 1, 1)) == 1.0)

def test_license_collection():
    '''Test license collection'''
    daily_lic = DailyLicense(42.0)
    annual_lic = AnnualLicense(365.0)
    collection = LicenseCollection([daily_lic, annual_lic])

    assert(len(collection) == 2)
    
    assert(collection.available(datetime(2015, 1, 1)) == 42.0)
    assert(collection.resource_state(datetime(2015, 1, 1)) == 1.0)
    assert(collection.resource_state(datetime(2015, 2, 1)) > 1.0)
    
    collection.commit(360.0)
    assert(collection.available(datetime(2015, 12, 1)) == 5.0)
    assert(collection.resource_state(datetime(2015, 12, 1)) < 1.0)

#!/usr/bin/env python

import pytest
from datetime import datetime

from pywr.licenses import License, DailyLicense, AnnualLicense

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

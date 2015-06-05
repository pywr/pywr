#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pywr.core
import pywr.licenses

import xml.etree.ElementTree as ET
import pandas
import dateutil
import datetime

# TODO: these should be moved into from_xml functions on the classes

def parse_parameter(model, xml):
    parameter_type = xml.get('type')
    key = xml.get('key')
    if parameter_type == 'const' or parameter_type == 'constant':
        try:
            value = float(xml.text)
        except:
            value = xml.text
        return key, pywr.core.Parameter(value=value)
    elif parameter_type == 'timeseries':
        name = xml.text
        return key, model.data[name]
    elif parameter_type == 'datetime':
        return key, pandas.to_datetime(xml.text)
    elif parameter_type == 'timedelta':
        return key, datetime.timedelta(float(xml.text))
    else:
        raise NotImplementedError('Unknown parameter type: {}'.format(parameter_type))

def parse_variable(model, xml):
    key = xml.get('key')
    value = float(xml.text)
    var = pywr.core.Variable(initial=value)
    return key, var

def parse_licensecollection(xml):
    collection = pywr.licenses.LicenseCollection([])
    for xml_lic in xml.getchildren():
        lic_type = xml_lic.get('type')
        value = float(xml_lic.text)
        lic_types = {
            'annual': pywr.licenses.AnnualLicense,
            'daily': pywr.licenses.DailyLicense,
        }
        lic = lic_types[lic_type](value)
        collection.add(lic)
    return collection

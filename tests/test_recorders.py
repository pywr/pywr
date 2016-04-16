# -*- coding: utf-8 -*-
"""
Test the Recorder object API

"""
from __future__ import print_function
import pywr.core
import numpy as np
import pytest
from numpy.testing import assert_allclose
from fixtures import simple_linear_model, simple_storage_model
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, AggregatedRecorder, \
                           CSVRecorder, TablesRecorder, TotalDeficitNodeRecorder, TotalFlowRecorder


def test_numpy_recorder(simple_linear_model):
    """
    Test the NumpyArrayNodeRecorder
    """
    model = simple_linear_model
    otpt = model.node['Output']

    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec = NumpyArrayNodeRecorder(model, otpt)

    # test retrieval of recorder
    assert model.recorders['numpyarraynoderecorder.Output'] == rec
    # test changing name of recorder
    rec.name = 'timeseries.Output'
    assert model.recorders['timeseries.Output'] == rec
    with pytest.raises(KeyError):
        model.recorders['numpyarraynoderecorder.Output']

    model.run()

    assert rec.data.shape == (365, 1)
    assert np.all((rec.data - 10.0) < 1e-12)


def test_numpy_storage_recorder(simple_storage_model):
    """
    Test the NumpyArrayStorageRecorder
    """
    model = simple_storage_model

    res = model.node['Storage']

    rec = NumpyArrayStorageRecorder(model, res)

    model.run()

    assert(rec.data.shape == (5, 1))
    assert_allclose(rec.data, np.array([[7, 4, 1, 0, 0]]).T, atol=1e-7)


def test_csv_recorder(simple_linear_model, tmpdir):
    """
    Test the CSV Recorder

    """
    model = simple_linear_model
    otpt = model.node['Output']
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0

    csvfile = tmpdir.join('output.csv')
    # By default the CSVRecorder saves all nodes in alphabetical order
    # and scenario index 0.
    rec = CSVRecorder(model, str(csvfile))

    model.run()

    import csv
    with open(str(csvfile), 'r') as fh:
        dialect = csv.Sniffer().sniff(fh.read(1024))
        fh.seek(0)
        reader = csv.reader(fh, dialect)
        for irow, row in enumerate(reader):
            if irow == 0:
                expected = ['Datetime', 'Input', 'Link', 'Output']
                actual = row
            else:
                dt = model.timestepper.start+(irow-1)*model.timestepper.delta
                expected = [dt.isoformat()]
                actual = [row[0]]
                assert np.all((np.array([float(v) for v in row[1:]]) - 10.0) < 1e-12)
            assert expected == actual


def test_hdf5_recorder(simple_linear_model, tmpdir):
    """
    Test the TablesRecorder

    """
    model = simple_linear_model
    otpt = model.node['Output']
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0

    h5file = tmpdir.join('output.h5')
    import tables
    with tables.open_file(str(h5file), 'w') as h5f:
        rec = TablesRecorder(model, h5f.root)

        model.run()

        for node_name in model.node.keys():
            ca = h5f.get_node('/', node_name)
            assert ca.shape == (365, 1)
            assert np.all((ca[...] - 10.0) < 1e-12)


def test_total_deficit_node_recorder(simple_linear_model):
    """
    Test TotalDeficitNodeRecorder
    """
    model = simple_linear_model
    otpt = model.node['Output']
    otpt.max_flow = 30.0
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec = TotalDeficitNodeRecorder(model, otpt)

    model.step()
    assert_allclose(20.0, rec.value(), atol=1e-7)

    model.step()
    assert_allclose(40.0, rec.value(), atol=1e-7)


def test_total_flow_node_recorder(simple_linear_model):
    """
    Test TotalDeficitNodeRecorder
    """
    model = simple_linear_model
    otpt = model.node['Output']
    otpt.max_flow = 30.0
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec = TotalFlowRecorder(model, otpt)

    model.step()
    assert_allclose(10.0, rec.value(), atol=1e-7)

    model.step()
    assert_allclose(20.0, rec.value(), atol=1e-7)


def test_aggregated_recorder(simple_linear_model):
    model = simple_linear_model
    otpt = model.node['Output']
    otpt.max_flow = 30.0
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec1 = TotalFlowRecorder(model, otpt)
    rec2 = TotalDeficitNodeRecorder(model, otpt)

    rec = AggregatedRecorder(model, [rec1, rec2], agg_func=np.max)

    model.step()
    assert_allclose(20.0, rec.value(), atol=1e-7)

    model.step()
    assert_allclose(40.0, rec.value(), atol=1e-7)


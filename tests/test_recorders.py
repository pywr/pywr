# -*- coding: utf-8 -*-
"""
Test the Recorder object API

"""
from __future__ import print_function
import pywr.core
from pywr.core import Model, Input, Output, Scenario
import numpy as np
import pandas
import pytest
from numpy.testing import assert_allclose
from fixtures import simple_linear_model, simple_storage_model
from pywr.recorders import (NumpyArrayNodeRecorder, NumpyArrayStorageRecorder,
    AggregatedRecorder, CSVRecorder, TablesRecorder, TotalDeficitNodeRecorder,
    TotalFlowNodeRecorder, MeanFlowRecorder, NumpyArrayParameterRecorder,
    NumpyArrayIndexParameterRecorder, MeanParameterRecorder, load_recorder)
from pywr.parameters import DailyProfileParameter, FunctionParameter
from helpers import load_model


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

    df = rec.to_dataframe()
    assert df.shape == (365, 1)
    assert np.all((df.values - 10.0) < 1e-12)


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


    df = rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, np.array([[7, 4, 1, 0, 0]]).T, atol=1e-7)


def test_numpy_parameter_recorder(simple_linear_model):
    """
    Test the NumpyArrayParameterRecorder
    """
    from pywr.parameters import DailyProfileParameter

    model = simple_linear_model
    # using leap year simplifies tests
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-12-31")
    otpt = model.node['Output']

    p = DailyProfileParameter(np.arange(366, dtype=np.float64), )
    p.name = 'daily profile'
    model.node['Input'].max_flow = p
    otpt.cost = -2.0
    rec = NumpyArrayParameterRecorder(model, model.node['Input'].max_flow)

    # test retrieval of recorder
    assert model.recorders['numpyarrayparameterrecorder.daily profile'] == rec

    model.run()

    assert rec.data.shape == (366, 1)
    assert_allclose(rec.data, np.arange(366, dtype=np.float64)[:, np.newaxis])

    df = rec.to_dataframe()
    assert df.shape == (366, 1)
    assert_allclose(df.values, np.arange(366, dtype=np.float64)[:, np.newaxis])


def test_numpy_index_parameter_recorder(simple_storage_model):
    """
    Test the NumpyArrayIndexParameterRecorder
    """
    from pywr.parameters.control_curves import ControlCurveIndexParameter

    model = simple_storage_model

    res = model.node['Storage']

    p = ControlCurveIndexParameter(res, [5.0/20.0, 2.5/20.0])

    res_rec = NumpyArrayStorageRecorder(model, res)
    lvl_rec = NumpyArrayIndexParameterRecorder(model, p)

    model.run()

    assert(res_rec.data.shape == (5, 1))
    assert_allclose(res_rec.data, np.array([[7, 4, 1, 0, 0]]).T, atol=1e-7)
    assert (lvl_rec.data.shape == (5, 1))
    assert_allclose(lvl_rec.data, np.array([[0, 1, 2, 2, 2]]).T, atol=1e-7)


    df = lvl_rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, np.array([[0, 1, 2, 2, 2]]).T, atol=1e-7)


def test_parameter_mean_recorder(simple_linear_model):
    model = simple_linear_model
    # using leap year simplifies test
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-12-31")

    node = model.nodes["Input"]
    values = np.arange(0, 366, dtype=np.float64)
    node.max_flow = DailyProfileParameter(values)

    timesteps = 3
    rec = MeanParameterRecorder(model, node.max_flow, timesteps)

    model.run()

    assert_allclose(rec.data[[0, 1, 2, 3, 364], 0], [0, 0.5, 1, 2, 363])

def test_parameter_mean_recorder_json(simple_linear_model):
    model = simple_linear_model
    node = model.nodes["Input"]
    values = np.arange(0, 366, dtype=np.float64)
    parameter = DailyProfileParameter(values, name="input_max_flow")
    model.parameters.append(parameter) # HACK
    node.max_flow = parameter

    data = {
        "type": "meanparameter",
        "parameter": "input_max_flow",
        "timesteps": 3,
    }

    rec = load_recorder(model, data)


def test_concatenated_dataframes(simple_storage_model):
    """
    Test that Model.to_dataframe returns something sensible.

    """
    model = simple_storage_model

    scA = Scenario(model, 'A', size=2)
    scB = Scenario(model, 'B', size=3)

    res = model.node['Storage']
    rec1 = NumpyArrayStorageRecorder(model, res)
    otpt = model.node['Output']
    rec2 = NumpyArrayNodeRecorder(model, otpt)
    # The following can't return a DataFrame; is included to check
    # it doesn't cause any issues
    rec3 = TotalDeficitNodeRecorder(model, otpt)

    model.run()

    df = model.to_dataframe()
    assert df.shape == (5, 2*2*3)
    assert df.columns.names == ['Recorder', 'A', 'B']


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
    assert_allclose(20.0, rec.aggregated_value(), atol=1e-7)

    model.step()
    assert_allclose(40.0, rec.aggregated_value(), atol=1e-7)


def test_total_flow_node_recorder(simple_linear_model):
    """
    Test TotalDeficitNodeRecorder
    """
    model = simple_linear_model
    otpt = model.node['Output']
    otpt.max_flow = 30.0
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec = TotalFlowNodeRecorder(model, otpt)

    model.step()
    assert_allclose(10.0, rec.aggregated_value(), atol=1e-7)

    model.step()
    assert_allclose(20.0, rec.aggregated_value(), atol=1e-7)


def test_aggregated_recorder(simple_linear_model):
    model = simple_linear_model
    otpt = model.node['Output']
    otpt.max_flow = 30.0
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec1 = TotalFlowNodeRecorder(model, otpt)
    rec2 = TotalDeficitNodeRecorder(model, otpt)

    rec = AggregatedRecorder(model, [rec1, rec2], agg_func="max")

    model.step()
    assert_allclose(20.0, rec.aggregated_value(), atol=1e-7)

    model.step()
    assert_allclose(40.0, rec.aggregated_value(), atol=1e-7)


def test_reset_timestepper_recorder(solver):
    model = Model(
        solver=solver,
        start=pandas.to_datetime('2016-01-01'),
        end=pandas.to_datetime('2016-01-01')
    )

    inpt = Input(model, "input", max_flow=10)
    otpt = Output(model, "output", max_flow=50, cost=-10)
    inpt.connect(otpt)

    rec = NumpyArrayNodeRecorder(model, otpt)

    model.run()

    model.timestepper.end = pandas.to_datetime("2016-01-02")

    model.run()

def test_mean_flow_recorder(solver):
    model = Model(solver=solver)
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-01-04")

    inpt = Input(model, "input")
    otpt = Output(model, "output")
    inpt.connect(otpt)

    rec_flow = NumpyArrayNodeRecorder(model, inpt)
    rec_mean = MeanFlowRecorder(model, node=inpt, timesteps=3)

    inpt.max_flow = inpt.min_flow = FunctionParameter(inpt, lambda model, t, si: 2 + t.index)
    model.run()

    expected = [
        2.0,
        (2.0 + 3.0) / 2,
        (2.0 + 3.0 + 4.0) / 3,
        (3.0 + 4.0 + 5.0) / 3,  # zeroth day forgotten
    ]

    for value, expected_value in zip(rec_mean.data, expected):
        assert_allclose(value, expected_value)

def test_mean_flow_recorder_json(solver):
    model = load_model("mean_flow_recorder.json", solver=solver)

    # TODO: it's not possible to define a FunctionParameter in JSON yet
    supply1 = model.nodes["supply1"]
    supply1.max_flow = supply1.min_flow = FunctionParameter(supply1, lambda model, t, si: 2 + t.index)

    assert(len(model.recorders) == 3)

    rec_flow = model.recorders["Supply"]
    rec_mean = model.recorders["Mean Flow"]
    rec_check = model.recorders["Supply 2"]

    model.run()

    assert_allclose(rec_flow.data[:,0], [2.0, 3.0, 4.0, 5.0])
    assert_allclose(rec_mean.data[:,0], [2.0, 2.5, 3.0, 4.0])
    assert_allclose(rec_check.data[:,0], [50.0, 50.0, 60.0, 60.0])

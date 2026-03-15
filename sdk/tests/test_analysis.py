import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from neurocore.result import RunResult
from neurocore import analysis

@pytest.fixture
def mock_result():
    return RunResult(
        total_spikes=10,
        timesteps=100,
        spike_trains={
            0: [5, 15, 25, 35, 45],
            1: [10, 20, 30],
            2: [50, 60],
        },
        placement=None,
        backend="simulator",
    )

class TestFiringRates:
    def test_per_neuron(self, mock_result):
        rates = analysis.firing_rates(mock_result)
        assert rates[0] == pytest.approx(5 / 100)
        assert rates[1] == pytest.approx(3 / 100)
        assert rates[2] == pytest.approx(2 / 100)

    def test_hardware_aggregate(self):
        result = RunResult(
            total_spikes=500, timesteps=100,
            spike_trains={}, placement=None, backend="chip",
        )
        rates = analysis.firing_rates(result)
        assert rates["aggregate"] == pytest.approx(5.0)

class TestSpikeCountTimeseries:
    def test_basic(self, mock_result):
        ts = analysis.spike_count_timeseries(mock_result, bin_size=10)
        assert len(ts) == 10
        assert ts[0] == 1
        assert ts[1] == 2

    def test_empty(self):
        result = RunResult(0, 100, {}, None, "chip")
        ts = analysis.spike_count_timeseries(result)
        assert len(ts) == 0

class TestISIHistogram:
    def test_basic(self, mock_result):
        counts, edges = analysis.isi_histogram(mock_result, bins=5)
        assert len(counts) == 5
        assert counts.sum() > 0

    def test_empty(self):
        result = RunResult(0, 100, {}, None, "simulator")
        counts, edges = analysis.isi_histogram(result)
        assert len(counts) == 0

class TestRasterPlot:
    def test_raster_no_display(self, mock_result):
        import matplotlib
        matplotlib.use("Agg")
        fig = analysis.raster_plot(mock_result, show=False)
        assert fig is not None

    def test_raster_hardware_fails(self):
        result = RunResult(100, 50, {}, None, "chip")
        with pytest.raises(Exception):
            result.raster_plot()

from .exceptions import NeurocoreError

class RunResult:

    def __init__(self, total_spikes, timesteps, spike_trains, placement, backend):
        self.total_spikes = total_spikes
        self.timesteps = timesteps
        self.spike_trains = spike_trains
        self.placement = placement
        self.backend = backend

    def raster_plot(self, filename=None, show=True, populations=None):
        if not self.spike_trains:
            raise NeurocoreError(
                "Per-neuron spike data not available. "
                "Hardware only returns total spike count. "
                "Use Simulator backend for raster plots.")
        from . import analysis
        return analysis.raster_plot(self, filename, show, populations)

    def firing_rates(self, population=None):
        from . import analysis
        return analysis.firing_rates(self, population)

    def spike_count_timeseries(self, bin_size=1):
        from . import analysis
        return analysis.spike_count_timeseries(self, bin_size)

    def isi_histogram(self, bins=50):
        from . import analysis
        return analysis.isi_histogram(self, bins)

    def to_dataframe(self):
        from . import analysis
        return analysis.to_dataframe(self)

    def __repr__(self):
        return (f"RunResult(total_spikes={self.total_spikes}, "
                f"timesteps={self.timesteps}, backend='{self.backend}')")

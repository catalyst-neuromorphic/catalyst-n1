"""RunResult container for spike data and analysis access."""

from .exceptions import NeurocoreError


class RunResult:
    """Encapsulates results from a run() call."""

    def __init__(self, total_spikes, timesteps, spike_trains, placement, backend):
        self.total_spikes = total_spikes
        self.timesteps = timesteps
        self.spike_trains = spike_trains  # {global_neuron_id: [timestep_list]}
        self.placement = placement
        self.backend = backend

    def raster_plot(self, filename=None, show=True, populations=None):
        """Generate a matplotlib spike raster plot.

        Only available with Simulator backend (hardware doesn't report
        per-neuron spike data).
        """
        if not self.spike_trains:
            raise NeurocoreError(
                "Per-neuron spike data not available. "
                "Hardware only returns total spike count. "
                "Use Simulator backend for raster plots.")
        from . import analysis
        return analysis.raster_plot(self, filename, show, populations)

    def firing_rates(self, population=None):
        """Compute mean firing rate (spikes/timestep) per neuron."""
        from . import analysis
        return analysis.firing_rates(self, population)

    def spike_count_timeseries(self, bin_size=1):
        """Total spikes per time bin across all neurons."""
        from . import analysis
        return analysis.spike_count_timeseries(self, bin_size)

    def isi_histogram(self, bins=50):
        """Inter-spike interval distribution."""
        from . import analysis
        return analysis.isi_histogram(self, bins)

    def to_dataframe(self):
        """Export spike data as a pandas DataFrame."""
        from . import analysis
        return analysis.to_dataframe(self)

    def __repr__(self):
        return (f"RunResult(total_spikes={self.total_spikes}, "
                f"timesteps={self.timesteps}, backend='{self.backend}')")

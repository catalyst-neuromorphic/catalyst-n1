import numpy as np
from .constants import NEURONS_PER_CORE

def raster_plot(result, filename=None, show=True, populations=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    colors = ["#00ffcc", "#ff6b6b", "#ffd93d", "#6bcfff",
              "#c084fc", "#ff9f43", "#2ed573", "#ff6348"]

    if populations and result.placement:
        for idx, pop in enumerate(populations):
            color = colors[idx % len(colors)]
            for local_i in range(pop.size):
                key = (pop.id, local_i)
                if key in result.placement.neuron_map:
                    core, neuron = result.placement.neuron_map[key]
                    gid = core * NEURONS_PER_CORE + neuron
                    if gid in result.spike_trains:
                        times = result.spike_trains[gid]
                        ax.scatter(times, [gid] * len(times), s=1,
                                   c=color, marker="|", linewidths=0.5)
            ax.scatter([], [], s=20, c=color, marker="|", label=pop.label)
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                  edgecolor="#333", labelcolor="white")
    else:
        for gid, times in result.spike_trains.items():
            ax.scatter(times, [gid] * len(times), s=1,
                       c="#00ffcc", marker="|", linewidths=0.5)

    ax.set_xlabel("Timestep", color="white", fontsize=10)
    ax.set_ylabel("Neuron ID", color="white", fontsize=10)
    ax.set_title(f"Spike Raster ({result.total_spikes} spikes, "
                 f"{result.timesteps} timesteps)",
                 color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, facecolor="#0a0a1a")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def firing_rates(result, population=None):
    if not result.spike_trains:
        if result.timesteps > 0:
            return {"aggregate": result.total_spikes / result.timesteps}
        return {"aggregate": 0.0}

    rates = {}
    if population and result.placement:
        for local_i in range(population.size):
            key = (population.id, local_i)
            if key in result.placement.neuron_map:
                core, neuron = result.placement.neuron_map[key]
                gid = core * NEURONS_PER_CORE + neuron
                n_spikes = len(result.spike_trains.get(gid, []))
                rates[gid] = n_spikes / result.timesteps if result.timesteps > 0 else 0.0
    else:
        for gid, times in result.spike_trains.items():
            rates[gid] = len(times) / result.timesteps if result.timesteps > 0 else 0.0
    return rates

def spike_count_timeseries(result, bin_size=1):
    if not result.spike_trains:
        return np.array([])

    n_bins = (result.timesteps + bin_size - 1) // bin_size
    counts = np.zeros(n_bins, dtype=np.int32)
    for times in result.spike_trains.values():
        for t in times:
            bin_idx = t // bin_size
            if bin_idx < n_bins:
                counts[bin_idx] += 1
    return counts

def isi_histogram(result, bins=50):
    if not result.spike_trains:
        return np.array([]), np.array([])

    intervals = []
    for times in result.spike_trains.values():
        sorted_t = sorted(times)
        for i in range(1, len(sorted_t)):
            intervals.append(sorted_t[i] - sorted_t[i - 1])

    if not intervals:
        return np.array([]), np.array([])

    return np.histogram(intervals, bins=bins)

def to_dataframe(result):
    import pandas as pd

    rows = []
    for gid, times in result.spike_trains.items():
        core = gid // NEURONS_PER_CORE
        local = gid % NEURONS_PER_CORE
        for t in times:
            rows.append({
                "timestep": t,
                "neuron_id": gid,
                "core": core,
                "local_neuron": local,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("timestep").reset_index(drop=True)
    return df

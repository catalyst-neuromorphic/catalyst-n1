"""Neurocore Project Dashboard — Full system visualization."""

import sys
sys.path.insert(0, r"C:\Users\mrwab\neuromorphic-chip\sdk")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import numpy as np
from collections import defaultdict

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE

net = nc.Network()
exc = net.population(64, params={"threshold": 500, "leak": 2, "refrac": 2}, label="Excitatory")
inh = net.population(16, params={"threshold": 400, "leak": 2, "refrac": 2}, label="Inhibitory")

net.connect(exc, exc, topology="random_sparse", p=0.15, weight=300, seed=42)
net.connect(exc, inh, topology="fixed_fan_out", fan_out=16, weight=250, seed=42)
net.connect(inh, exc, topology="fixed_fan_out", fan_out=32, weight=-200, seed=42)

sim = nc.Simulator()
sim.deploy(net)
compiled = sim._compiled

# Run with sustained input, collecting per-timestep data
spike_trains = defaultdict(list)
potential_log = {0: [], 10: [], 64: []}  # track a few neurons' membrane potential
spike_counts_per_ts = []
total = 0

for t in range(200):
    sim.inject(exc[:16], current=600)
    # Log membrane potentials before running
    for gid in potential_log:
        potential_log[gid].append(int(sim._potential[gid]))
    result = sim.run(1)
    total += result.total_spikes
    spike_counts_per_ts.append(result.total_spikes)
    for gid, times in result.spike_trains.items():
        spike_trains[gid].extend([t])

from neurocore.result import RunResult
combined = RunResult(total, 200, dict(spike_trains), compiled.placement, "simulator")

BG = "#0a0a1a"
PANEL_BG = "#0f1029"
GRID_COLOR = "#1a1a3a"
TEXT_COLOR = "#e0e0e0"
ACCENT1 = "#00ffcc"  # cyan/green - excitatory
ACCENT2 = "#ff6b6b"  # red/coral - inhibitory
ACCENT3 = "#ffd93d"  # gold
ACCENT4 = "#6bcfff"  # light blue
ACCENT5 = "#c084fc"  # purple

fig = plt.figure(figsize=(24, 16), facecolor=BG)
fig.suptitle("NEUROCORE  —  Neuromorphic Chip Project Dashboard",
             fontsize=22, color=ACCENT1, fontweight="bold",
             fontfamily="monospace", y=0.98)
fig.text(0.5, 0.955, "128-core × 256-neuron spiking neural processor  |  "
         "P1–P11 complete  |  STDP · Graded Spikes · Dendritic Compartments · 32K neurons",
         ha="center", fontsize=10, color="#666", fontfamily="monospace")

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                       left=0.04, right=0.97, top=0.93, bottom=0.04)

ax_arch = fig.add_subplot(gs[0, 0:2])
ax_arch.set_facecolor(PANEL_BG)
ax_arch.set_xlim(-0.5, 15.5)
ax_arch.set_ylim(-0.5, 9.5)
ax_arch.set_aspect("equal")
ax_arch.set_title("Chip Architecture — 4×4 Core Mesh (FPGA overlay)",
                   color=TEXT_COLOR, fontsize=11, fontfamily="monospace", pad=10)
ax_arch.axis("off")

# Draw 4x4 mesh of cores (showing 16 of 128 possible)
core_positions = {}
for row in range(4):
    for col in range(4):
        cx = col * 4 + 1.5
        cy = (3 - row) * 2.5 + 1
        core_id = row * 4 + col
        core_positions[core_id] = (cx, cy)

        # Core box
        color = ACCENT1 if core_id < compiled.placement.num_cores_used else "#1a2a3a"
        alpha = 0.9 if core_id < compiled.placement.num_cores_used else 0.3
        rect = FancyBboxPatch((cx - 1.3, cy - 0.8), 2.6, 1.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax_arch.add_patch(rect)

        # Core label
        ax_arch.text(cx, cy + 0.3, f"Core {core_id}", ha="center", va="center",
                     fontsize=7, color=color, fontweight="bold", fontfamily="monospace",
                     alpha=alpha)
        ax_arch.text(cx, cy - 0.1, "256 LIF neurons", ha="center", va="center",
                     fontsize=5.5, color=color, fontfamily="monospace", alpha=alpha * 0.7)
        ax_arch.text(cx, cy - 0.4, "32-slot fanout", ha="center", va="center",
                     fontsize=5.5, color=color, fontfamily="monospace", alpha=alpha * 0.7)

        # Mesh connections (right and down)
        if col < 3:
            ncx = (col + 1) * 4 + 1.5
            ax_arch.annotate("", xy=(ncx - 1.4, cy), xytext=(cx + 1.4, cy),
                            arrowprops=dict(arrowstyle="<->", color="#334", lw=0.8))
        if row < 3:
            ncy = (3 - row - 1) * 2.5 + 1
            ax_arch.annotate("", xy=(cx, ncy + 0.9), xytext=(cx, cy - 0.9),
                            arrowprops=dict(arrowstyle="<->", color="#334", lw=0.8))

ax_topo = fig.add_subplot(gs[0, 2:4])
ax_topo.set_facecolor(PANEL_BG)
ax_topo.set_title("E/I Network Topology — 64 exc + 16 inh",
                   color=TEXT_COLOR, fontsize=11, fontfamily="monospace", pad=10)
ax_topo.set_xlim(-1.5, 1.5)
ax_topo.set_ylim(-1.5, 1.5)
ax_topo.set_aspect("equal")
ax_topo.axis("off")

# Place excitatory neurons in a ring
exc_positions = {}
for i in range(64):
    angle = 2 * np.pi * i / 64
    x = np.cos(angle) * 1.1
    y = np.sin(angle) * 1.1
    exc_positions[i] = (x, y)
    ax_topo.plot(x, y, "o", color=ACCENT1, markersize=3, alpha=0.7)

# Place inhibitory neurons in inner ring
inh_positions = {}
for i in range(16):
    angle = 2 * np.pi * i / 16
    x = np.cos(angle) * 0.5
    y = np.sin(angle) * 0.5
    inh_positions[i] = (x, y)
    ax_topo.plot(x, y, "s", color=ACCENT2, markersize=5, alpha=0.9)

# Draw a sample of connections (not all — too dense)
rng = np.random.default_rng(42)
# E->E (sparse sample)
adj = compiled.adjacency
drawn = 0
for src_gid, targets in adj.items():
    if drawn > 200:
        break
    src_local = src_gid % NEURONS_PER_CORE
    if src_local >= 64:
        continue
    for tgt_gid, w, comp in targets:
        tgt_local = tgt_gid % NEURONS_PER_CORE
        if tgt_local < 64 and rng.random() < 0.15:
            sx, sy = exc_positions[src_local]
            tx, ty = exc_positions[tgt_local]
            ax_topo.plot([sx, tx], [sy, ty], "-", color=ACCENT1, alpha=0.04, lw=0.5)
            drawn += 1

# E->I connections (sample)
drawn = 0
for src_gid, targets in adj.items():
    if drawn > 80:
        break
    src_local = src_gid % NEURONS_PER_CORE
    if src_local >= 64:
        continue
    for tgt_gid, w, comp in targets:
        tgt_local = tgt_gid % NEURONS_PER_CORE
        if 64 <= tgt_local < 80 and rng.random() < 0.2:
            sx, sy = exc_positions[src_local]
            tx, ty = inh_positions[tgt_local - 64]
            ax_topo.plot([sx, tx], [sy, ty], "-", color=ACCENT3, alpha=0.08, lw=0.5)
            drawn += 1

# I->E connections (sample)
drawn = 0
for src_gid, targets in adj.items():
    if drawn > 80:
        break
    src_local = src_gid % NEURONS_PER_CORE
    if not (64 <= src_local < 80):
        continue
    for tgt_gid, w, comp in targets:
        tgt_local = tgt_gid % NEURONS_PER_CORE
        if tgt_local < 64 and rng.random() < 0.15:
            sx, sy = inh_positions[src_local - 64]
            tx, ty = exc_positions[tgt_local]
            ax_topo.plot([sx, tx], [sy, ty], "-", color=ACCENT2, alpha=0.08, lw=0.5)
            drawn += 1

# Legend
ax_topo.plot([], [], "o", color=ACCENT1, markersize=5, label="Excitatory (64)")
ax_topo.plot([], [], "s", color=ACCENT2, markersize=5, label="Inhibitory (16)")
ax_topo.plot([], [], "-", color=ACCENT1, alpha=0.5, label="E→E (p=0.15)")
ax_topo.plot([], [], "-", color=ACCENT3, alpha=0.5, label="E→I (fan=16)")
ax_topo.plot([], [], "-", color=ACCENT2, alpha=0.5, label="I→E (fan=32)")
ax_topo.legend(loc="lower right", fontsize=7, facecolor=PANEL_BG,
               edgecolor="#333", labelcolor=TEXT_COLOR, framealpha=0.9)

ax_raster = fig.add_subplot(gs[1, :])
ax_raster.set_facecolor(PANEL_BG)
ax_raster.set_title("Spike Raster — 200 timesteps, sustained drive to exc[:16]",
                     color=TEXT_COLOR, fontsize=11, fontfamily="monospace", pad=10)

for gid, times in spike_trains.items():
    local = gid % NEURONS_PER_CORE
    if local < 64:
        color = ACCENT1
    else:
        color = ACCENT2
    ax_raster.scatter(times, [gid] * len(times), s=0.8, c=color, marker="|", linewidths=0.4)

ax_raster.set_xlabel("Timestep", color=TEXT_COLOR, fontsize=9, fontfamily="monospace")
ax_raster.set_ylabel("Neuron ID", color=TEXT_COLOR, fontsize=9, fontfamily="monospace")
ax_raster.tick_params(colors="#666", labelsize=7)
for spine in ax_raster.spines.values():
    spine.set_color("#222")
ax_raster.set_xlim(0, 200)

# Patches for legend
exc_patch = mpatches.Patch(color=ACCENT1, label="Excitatory")
inh_patch = mpatches.Patch(color=ACCENT2, label="Inhibitory")
ax_raster.legend(handles=[exc_patch, inh_patch], loc="upper right", fontsize=7,
                 facecolor=PANEL_BG, edgecolor="#333", labelcolor=TEXT_COLOR)

ax_rate = fig.add_subplot(gs[2, 0])
ax_rate.set_facecolor(PANEL_BG)
ax_rate.set_title("Firing Rate Distribution", color=TEXT_COLOR, fontsize=10,
                   fontfamily="monospace", pad=8)

rates = combined.firing_rates()
exc_rates = [rates.get(gid, 0) for gid in range(64)]
inh_rates = [rates.get(gid, 0) for gid in range(64, 80)]

ax_rate.hist(exc_rates, bins=15, color=ACCENT1, alpha=0.7, label="Exc", edgecolor="#0a0a1a")
ax_rate.hist(inh_rates, bins=8, color=ACCENT2, alpha=0.7, label="Inh", edgecolor="#0a0a1a")
ax_rate.set_xlabel("Firing rate (spikes/ts)", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_rate.set_ylabel("Count", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_rate.tick_params(colors="#666", labelsize=7)
ax_rate.legend(fontsize=7, facecolor=PANEL_BG, edgecolor="#333", labelcolor=TEXT_COLOR)
for spine in ax_rate.spines.values():
    spine.set_color("#222")

ax_ts = fig.add_subplot(gs[2, 1])
ax_ts.set_facecolor(PANEL_BG)
ax_ts.set_title("Network Activity Over Time", color=TEXT_COLOR, fontsize=10,
                 fontfamily="monospace", pad=8)

ax_ts.fill_between(range(200), spike_counts_per_ts, color=ACCENT1, alpha=0.3)
ax_ts.plot(spike_counts_per_ts, color=ACCENT1, lw=1, alpha=0.9)

# Moving average
window = 10
if len(spike_counts_per_ts) >= window:
    ma = np.convolve(spike_counts_per_ts, np.ones(window)/window, mode="valid")
    ax_ts.plot(range(window-1, 200), ma, color=ACCENT3, lw=2, label=f"{window}-pt avg")
    ax_ts.legend(fontsize=7, facecolor=PANEL_BG, edgecolor="#333", labelcolor=TEXT_COLOR)

ax_ts.set_xlabel("Timestep", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_ts.set_ylabel("Spikes", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_ts.tick_params(colors="#666", labelsize=7)
for spine in ax_ts.spines.values():
    spine.set_color("#222")

ax_mem = fig.add_subplot(gs[2, 2])
ax_mem.set_facecolor(PANEL_BG)
ax_mem.set_title("Membrane Potential Traces", color=TEXT_COLOR, fontsize=10,
                  fontfamily="monospace", pad=8)

colors_mem = [ACCENT1, ACCENT4, ACCENT2]
labels_mem = ["exc[0] (driven)", "exc[10] (recurrent)", "inh[0]"]
for idx, (gid, color, label) in enumerate(zip([0, 10, 64], colors_mem, labels_mem)):
    trace = potential_log[gid]
    ax_mem.plot(trace, color=color, lw=0.8, alpha=0.9, label=label)

ax_mem.axhline(y=500, color=ACCENT1, lw=0.5, ls="--", alpha=0.3, label="exc threshold")
ax_mem.axhline(y=400, color=ACCENT2, lw=0.5, ls="--", alpha=0.3, label="inh threshold")
ax_mem.set_xlabel("Timestep", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_mem.set_ylabel("Potential", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_mem.tick_params(colors="#666", labelsize=7)
ax_mem.legend(fontsize=6, facecolor=PANEL_BG, edgecolor="#333", labelcolor=TEXT_COLOR, loc="upper right")
ax_mem.set_xlim(0, 200)
for spine in ax_mem.spines.values():
    spine.set_color("#222")

ax_isi = fig.add_subplot(gs[2, 3])
ax_isi.set_facecolor(PANEL_BG)
ax_isi.set_title("Inter-Spike Interval Distribution", color=TEXT_COLOR, fontsize=10,
                  fontfamily="monospace", pad=8)

counts_isi, edges_isi = combined.isi_histogram(bins=20)
if counts_isi:
    centers = (edges_isi[:-1] + edges_isi[1:]) / 2
    widths = edges_isi[1:] - edges_isi[:-1]
    ax_isi.bar(centers, counts_isi, width=widths * 0.9, color=ACCENT5, alpha=0.8,
               edgecolor="#0a0a1a")

ax_isi.set_xlabel("ISI (timesteps)", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_isi.set_ylabel("Count", color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
ax_isi.tick_params(colors="#666", labelsize=7)
for spine in ax_isi.spines.values():
    spine.set_color("#222")

stats_text = (
    f"Total spikes: {total:,}\n"
    f"Active neurons: {len([r for r in rates.values() if r > 0])}/80\n"
    f"Connections: {len(compiled.prog_conn_cmds):,}\n"
    f"Cores used: {compiled.placement.num_cores_used}\n"
    f"SDK v{nc.__version__}"
)
fig.text(0.97, 0.04, stats_text, ha="right", va="bottom",
         fontsize=8, color="#555", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG,
                   edgecolor="#222", alpha=0.9))

# Save
output = r"C:\Users\mrwab\neuromorphic-chip\sdk\neurocore_dashboard.png"
plt.savefig(output, dpi=180, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Dashboard saved to: {output}")

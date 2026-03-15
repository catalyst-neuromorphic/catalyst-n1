"""Visualize async vs sync mode — the key P12 feature."""

import sys
sys.path.insert(0, r"C:\Users\mrwab\neuromorphic-chip\sdk")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

import neurocore as nc
from neurocore.result import RunResult
from neurocore.constants import NEURONS_PER_CORE

BG = "#0a0a1a"
PANEL = "#0f1029"
TEXT = "#e0e0e0"
CYAN = "#00ffcc"
RED = "#ff6b6b"
GOLD = "#ffd93d"
BLUE = "#6bcfff"
PURPLE = "#c084fc"
GREEN = "#2ed573"

def run_chain(async_mode):
    net = nc.Network()
    pops = []
    for i in range(8):
        p = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 1},
                           label=f"N{i}")
        pops.append(p)
    for i in range(7):
        net.connect(pops[i], pops[i+1], topology="all_to_all", weight=200)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(async_mode=async_mode)

    trains = defaultdict(list)
    total = 0
    for t in range(12):
        if t == 0:
            sim.inject(pops[0], current=200)
        result = sim.run(1)
        total += result.total_spikes
        for gid, times in result.spike_trains.items():
            trains[gid].extend([t])
    return trains, total, sim._compiled.placement, pops

sync_trains, sync_total, placement, pops = run_chain(False)
async_trains, async_total, _, _ = run_chain(True)

def run_ei(async_mode, timesteps=150):
    net = nc.Network()
    exc = net.population(64, params={"threshold": 500, "leak": 2, "refrac": 2}, label="Excitatory")
    inh = net.population(16, params={"threshold": 400, "leak": 2, "refrac": 2}, label="Inhibitory")
    net.connect(exc, exc, topology="random_sparse", p=0.15, weight=300, seed=42)
    net.connect(exc, inh, topology="fixed_fan_out", fan_out=16, weight=250, seed=42)
    net.connect(inh, exc, topology="fixed_fan_out", fan_out=32, weight=-200, seed=42)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(async_mode=async_mode)

    trains = defaultdict(list)
    counts = []
    total = 0
    for t in range(timesteps):
        sim.inject(exc[:16], current=600)
        result = sim.run(1)
        total += result.total_spikes
        counts.append(result.total_spikes)
        for gid, times in result.spike_trains.items():
            trains[gid].extend([t])
    return dict(trains), counts, total, sim._compiled.placement, exc, inh

sync_ei_trains, sync_ei_counts, sync_ei_total, ei_place, exc, inh = run_ei(False)
async_ei_trains, async_ei_counts, async_ei_total, _, _, _ = run_ei(True)

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle("NEUROCORE  —  Async Event-Driven Mode (Phase 12 GALS)",
             fontsize=20, color=CYAN, fontweight="bold", fontfamily="monospace", y=0.98)
fig.text(0.5, 0.955, "Togglable via set_learning(async_mode=True)  |  "
         "Cores fire only on pending spikes  |  Quiescence detection ends timestep",
         ha="center", fontsize=9, color="#666", fontfamily="monospace")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.25,
                       left=0.05, right=0.96, top=0.93, bottom=0.05)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
ax1.set_title("SYNC Mode — 8-Neuron Chain", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

for gid, times in sync_trains.items():
    neuron = gid % NEURONS_PER_CORE
    ax1.scatter(times, [neuron] * len(times), s=120, c=CYAN, marker="|", linewidths=2.5)
    for t in times:
        ax1.annotate(f"N{neuron}", (t + 0.15, neuron), fontsize=7, color="#888",
                     fontfamily="monospace")

ax1.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax1.set_ylabel("Neuron", color=TEXT, fontsize=9, fontfamily="monospace")
ax1.set_xlim(-0.5, 11.5)
ax1.set_ylim(-0.5, 7.5)
ax1.set_yticks(range(8))
ax1.set_yticklabels([f"N{i}" for i in range(8)])
ax1.tick_params(colors="#666", labelsize=8)
for spine in ax1.spines.values():
    spine.set_color("#222")

# Arrow showing propagation direction
ax1.annotate("", xy=(7.5, 7), xytext=(0.5, 0),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5, ls="--"))
ax1.text(5, 2.5, f"7 timesteps\n{sync_total} total spikes", fontsize=10,
         color=GOLD, fontfamily="monospace", ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=GOLD, alpha=0.8))

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)
ax2.set_title("ASYNC Mode — 8-Neuron Chain (same network)", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

for gid, times in async_trains.items():
    neuron = gid % NEURONS_PER_CORE
    ax2.scatter(times, [neuron] * len(times), s=120, c=GREEN, marker="|", linewidths=2.5)
    for t in times:
        ax2.annotate(f"N{neuron}", (t + 0.15, neuron), fontsize=7, color="#888",
                     fontfamily="monospace")

ax2.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax2.set_ylabel("Neuron", color=TEXT, fontsize=9, fontfamily="monospace")
ax2.set_xlim(-0.5, 11.5)
ax2.set_ylim(-0.5, 7.5)
ax2.set_yticks(range(8))
ax2.set_yticklabels([f"N{i}" for i in range(8)])
ax2.tick_params(colors="#666", labelsize=8)
for spine in ax2.spines.values():
    spine.set_color("#222")

# All spikes at t=0
ax2.text(0.5, 4, f"1 timestep!\n{async_total} spikes\n(micro-steps)", fontsize=10,
         color=GREEN, fontfamily="monospace", ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=GREEN, alpha=0.8))

ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
ax3.set_title(f"SYNC E/I Network — {sync_ei_total:,} spikes / 150 ts",
              color=TEXT, fontsize=12, fontfamily="monospace", pad=10)

for gid, times in sync_ei_trains.items():
    local = gid % NEURONS_PER_CORE
    color = CYAN if local < 64 else RED
    ax3.scatter(times, [gid] * len(times), s=0.6, c=color, marker="|", linewidths=0.3)

ax3.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax3.set_ylabel("Neuron ID", color=TEXT, fontsize=9, fontfamily="monospace")
ax3.tick_params(colors="#666", labelsize=7)
for spine in ax3.spines.values():
    spine.set_color("#222")
exc_p = mpatches.Patch(color=CYAN, label="Exc")
inh_p = mpatches.Patch(color=RED, label="Inh")
ax3.legend(handles=[exc_p, inh_p], loc="upper right", fontsize=7,
           facecolor=PANEL, edgecolor="#333", labelcolor=TEXT)

ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
ax4.set_title(f"ASYNC E/I Network — {async_ei_total:,} spikes / 150 ts",
              color=TEXT, fontsize=12, fontfamily="monospace", pad=10)

for gid, times in async_ei_trains.items():
    local = gid % NEURONS_PER_CORE
    color = GREEN if local < 64 else PURPLE
    ax4.scatter(times, [gid] * len(times), s=0.6, c=color, marker="|", linewidths=0.3)

ax4.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax4.set_ylabel("Neuron ID", color=TEXT, fontsize=9, fontfamily="monospace")
ax4.tick_params(colors="#666", labelsize=7)
for spine in ax4.spines.values():
    spine.set_color("#222")
exc_p2 = mpatches.Patch(color=GREEN, label="Exc (async)")
inh_p2 = mpatches.Patch(color=PURPLE, label="Inh (async)")
ax4.legend(handles=[exc_p2, inh_p2], loc="upper right", fontsize=7,
           facecolor=PANEL, edgecolor="#333", labelcolor=TEXT)

ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor(PANEL)
ax5.set_title("Network Activity — Sync vs Async", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

window = 5
sync_ma = np.convolve(sync_ei_counts, np.ones(window)/window, mode="valid")
async_ma = np.convolve(async_ei_counts, np.ones(window)/window, mode="valid")
x = range(window - 1, 150)

ax5.fill_between(x, sync_ma, alpha=0.15, color=CYAN)
ax5.plot(x, sync_ma, color=CYAN, lw=1.5, label=f"Sync ({sync_ei_total:,} spikes)")
ax5.fill_between(x, async_ma, alpha=0.15, color=GREEN)
ax5.plot(x, async_ma, color=GREEN, lw=1.5, label=f"Async ({async_ei_total:,} spikes)")

ax5.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax5.set_ylabel("Spikes / ts (5-pt avg)", color=TEXT, fontsize=9, fontfamily="monospace")
ax5.tick_params(colors="#666", labelsize=7)
ax5.legend(fontsize=8, facecolor=PANEL, edgecolor="#333", labelcolor=TEXT)
for spine in ax5.spines.values():
    spine.set_color("#222")

ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor(PANEL)
ax6.set_title("P12 Async Architecture — GALS Event Loop", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 8)
ax6.axis("off")

# Draw the async FSM flow
boxes = [
    (5, 7, "IDLE", "#555"),
    (5, 5.5, "ASYNC_ACTIVE\n(main loop)", GREEN),
    (1.5, 3.5, "INJECT\n(drain PCIF)", BLUE),
    (5, 3.5, "ROUTE\n(inter-core)", GOLD),
    (8.5, 3.5, "RESTART\n(deferred)", PURPLE),
    (5, 1.5, "QUIESCENT\n(timestep done)", CYAN),
]

for bx, by, label, color in boxes:
    rect = mpatches.FancyBboxPatch((bx - 1.1, by - 0.55), 2.2, 1.1,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, alpha=0.15,
                                    edgecolor=color, linewidth=1.5)
    ax6.add_patch(rect)
    ax6.text(bx, by, label, ha="center", va="center", fontsize=7.5,
             color=color, fontweight="bold", fontfamily="monospace")

# Arrows
arrow_style = dict(arrowstyle="->", lw=1.2)
arrows = [
    ((5, 6.4), (5, 6.1), "#555"),         # IDLE → ACTIVE
    ((3.8, 5.2), (2.6, 4.1), BLUE),       # ACTIVE → INJECT
    ((5, 4.9), (5, 4.1), GOLD),           # ACTIVE → ROUTE
    ((6.2, 5.2), (7.4, 4.1), PURPLE),     # ACTIVE → RESTART
    ((2.6, 3.0), (3.8, 5.0), BLUE),       # INJECT → ACTIVE (back)
    ((4.0, 3.8), (3.8, 5.0), GOLD),       # ROUTE → ACTIVE (back, shifted)
    ((7.4, 3.0), (6.2, 5.0), PURPLE),     # RESTART → ACTIVE (back)
    ((5, 4.9), (5, 2.1), CYAN),           # ACTIVE → QUIESCENT
]

for start, end, color in arrows:
    ax6.annotate("", xy=end, xytext=start,
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

# Labels on arrows
ax6.text(2.2, 4.8, "PCIF\nnon-empty", fontsize=6, color=BLUE,
         fontfamily="monospace", ha="center")
ax6.text(5.7, 4.5, "capture\nFIFO", fontsize=6, color=GOLD,
         fontfamily="monospace", ha="center")
ax6.text(7.8, 4.8, "core\nspiked", fontsize=6, color=PURPLE,
         fontfamily="monospace", ha="center")
ax6.text(3.8, 2.3, "all quiet", fontsize=6, color=CYAN,
         fontfamily="monospace", ha="center")

# Key insight callout
ax6.text(5, 0.5,
         "Key: chains collapse into micro-steps within 1 timestep\n"
         "Quiescence = all cores idle + no restarts + all FIFOs empty",
         ha="center", va="center", fontsize=7, color="#888",
         fontfamily="monospace", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a1a",
                   edgecolor="#333", alpha=0.8))

# Save
output = r"C:\Users\mrwab\neuromorphic-chip\sdk\async_dashboard.png"
plt.savefig(output, dpi=180, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Saved to: {output}")

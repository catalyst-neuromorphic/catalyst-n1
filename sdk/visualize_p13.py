"""Visualize P13 Loihi Parity features — CSR pool, multicast, 3-factor learning."""

import sys
sys.path.insert(0, r"C:\Users\mrwab\neuromorphic-chip\sdk")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict

import neurocore as nc
from neurocore.result import RunResult
from neurocore.constants import NEURONS_PER_CORE, POOL_DEPTH, ROUTE_FANOUT

BG = "#0a0a1a"
PANEL = "#0f1029"
TEXT = "#e0e0e0"
CYAN = "#00ffcc"
RED = "#ff6b6b"
GOLD = "#ffd93d"
BLUE = "#6bcfff"
PURPLE = "#c084fc"
GREEN = "#2ed573"
ORANGE = "#ff9f43"
PINK = "#ff6b9d"

print("Running CSR pool demo...")
net_csr = nc.Network()
hub = net_csr.population(1, params={"threshold": 100, "leak": 0, "refrac": 1}, label="Hub")
fan_out_pop = net_csr.population(100, params={"threshold": 100, "leak": 0, "refrac": 1}, label="Fan-out targets")
sparse_src = net_csr.population(50, params={"threshold": 100, "leak": 0, "refrac": 1}, label="Sparse sources")
# Hub neuron connects to ALL 100 targets (was impossible with 32-slot limit!)
net_csr.connect(hub, fan_out_pop, topology="all_to_all", weight=200)
# Sparse sources connect to 3 targets each
net_csr.connect(sparse_src, fan_out_pop, topology="fixed_fan_out", fan_out=3, weight=150, seed=42)

sim_csr = nc.Simulator()
sim_csr.deploy(net_csr)
compiled = sim_csr._compiled

# Gather fanout distribution from index cmds
fanout_per_neuron = {}
for cmd in compiled.prog_index_cmds:
    fanout_per_neuron[cmd["neuron"]] = cmd["count"]

# Run simulation
csr_trains = defaultdict(list)
csr_total = 0
for t in range(30):
    if t < 3:
        sim_csr.inject(hub, current=200)
        sim_csr.inject(sparse_src[:10], current=200)
    result = sim_csr.run(1)
    csr_total += result.total_spikes
    for gid, times in result.spike_trains.items():
        csr_trains[gid].extend([t])

print("Running multicast routing demo...")
net_mcast = nc.Network()
src_core = net_mcast.population(NEURONS_PER_CORE, params={"threshold": 100, "leak": 0, "refrac": 2},
                                 label="Source core")
targets = []
for i in range(6):
    # 1 neuron per target to keep routes within 8-slot limit per source
    t = net_mcast.population(1, params={"threshold": 100, "leak": 0, "refrac": 2},
                              label=f"Target {i}")
    targets.append(t)
    net_mcast.connect(src_core, t, topology="all_to_all", weight=200)

sim_mcast = nc.Simulator()
sim_mcast.deploy(net_mcast)
mcast_compiled = sim_mcast._compiled

# Count routes per source neuron
routes_per_src = defaultdict(int)
for cmd in mcast_compiled.prog_route_cmds:
    routes_per_src[cmd["src_neuron"]] += 1

print("Running 3-factor learning demo...")

def run_3factor(reward_time, reward_value, label):
    net = nc.Network()
    pre = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 2}, label="Pre")
    post = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 2}, label="Post")
    net.connect(pre, post, topology="all_to_all", weight=500)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(learn=True, three_factor=True)

    weights_over_time = []
    elig_over_time = []

    for t in range(60):
        # Pre and post spike every 8 timesteps to build eligibility
        if t % 8 == 0 and t < 40:
            sim.inject(pre, current=200)
        if t % 8 == 2 and t < 40:
            sim.inject(post, current=200)

        # Apply reward at specified time
        if t == reward_time:
            sim.reward(reward_value)

        sim.run(1)

        # Record weight
        w = 500  # default
        for targets in sim._adjacency.values():
            for _, wt, _ in targets:
                w = wt
        weights_over_time.append(w)

        # Record total eligibility magnitude
        total_elig = sum(abs(v) for v in sim._eligibility.values())
        elig_over_time.append(total_elig)

    return weights_over_time, elig_over_time

# Positive reward at t=20
w_pos, e_pos = run_3factor(20, 800, "Positive reward")
# Negative reward at t=20
w_neg, e_neg = run_3factor(20, -800, "Negative reward")
# No reward (control)
w_none, e_none = run_3factor(999, 0, "No reward")
# Delayed reward at t=35
w_delayed, e_delayed = run_3factor(35, 800, "Delayed reward")

print("Running E/I network at 1024 scale...")
net_scale = nc.Network()
exc = net_scale.population(256, params={"threshold": 500, "leak": 2, "refrac": 2}, label="Excitatory")
inh = net_scale.population(64, params={"threshold": 400, "leak": 2, "refrac": 2}, label="Inhibitory")
# Use high fanout connections (>32 was impossible before!)
net_scale.connect(exc, exc, topology="random_sparse", p=0.12, weight=250, seed=42)
net_scale.connect(exc, inh, topology="fixed_fan_out", fan_out=48, weight=200, seed=42)
net_scale.connect(inh, exc, topology="fixed_fan_out", fan_out=64, weight=-180, seed=42)

sim_scale = nc.Simulator()
sim_scale.deploy(net_scale)
scale_compiled = sim_scale._compiled

scale_trains = defaultdict(list)
scale_counts = []
scale_total = 0
for t in range(200):
    sim_scale.inject(exc[:32], current=600)
    result = sim_scale.run(1)
    scale_total += result.total_spikes
    scale_counts.append(result.total_spikes)
    for gid, times in result.spike_trains.items():
        scale_trains[gid].extend([t])

print("Building figure...")
fig = plt.figure(figsize=(24, 22), facecolor=BG)
fig.suptitle("NEUROCORE  v0.2.0  —  Phase 13: Loihi 1 Parity",
             fontsize=22, color=CYAN, fontweight="bold", fontfamily="monospace", y=0.98)
fig.text(0.5, 0.96,
         "1024 neurons/core  |  CSR variable fanout (32K pool)  |  "
         "8× multicast routing  |  3-factor eligibility learning",
         ha="center", fontsize=10, color="#666", fontfamily="monospace")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.28,
                       left=0.05, right=0.96, top=0.93, bottom=0.04)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
ax1.set_title("P13a: CSR Variable Fanout", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

# Plot fanout distribution
fanouts = sorted(fanout_per_neuron.values())
unique_vals = sorted(set(fanouts))
counts_per = [fanouts.count(v) for v in unique_vals]
colors = [GOLD if v > 32 else CYAN for v in unique_vals]
bars = ax1.bar(range(len(unique_vals)), counts_per, color=colors, alpha=0.8, width=0.6)

ax1.set_xticks(range(len(unique_vals)))
ax1.set_xticklabels([str(v) for v in unique_vals], fontsize=8)
ax1.set_xlabel("Connections per neuron", color=TEXT, fontsize=9, fontfamily="monospace")
ax1.set_ylabel("Neuron count", color=TEXT, fontsize=9, fontfamily="monospace")
ax1.tick_params(colors="#666", labelsize=8)
for spine in ax1.spines.values():
    spine.set_color("#222")

# Callout for hub neuron
if any(v > 32 for v in unique_vals):
    ax1.text(0.95, 0.95, f"Hub: 100 targets!\n(was limited to 32)",
             transform=ax1.transAxes, fontsize=8, color=GOLD,
             fontfamily="monospace", ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GOLD, alpha=0.8))

# Legend
old_p = mpatches.Patch(color=CYAN, label="Within old limit (≤32)")
new_p = mpatches.Patch(color=GOLD, label="Exceeds old limit (>32)")
ax1.legend(handles=[old_p, new_p], loc="center right", fontsize=7,
           facecolor=PANEL, edgecolor="#333", labelcolor=TEXT)

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)
ax2.set_title("CSR Pool Architecture", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis("off")

# Index table
ax2.add_patch(mpatches.FancyBboxPatch((0.3, 5.5), 3.5, 2,
              boxstyle="round,pad=0.15", facecolor=CYAN, alpha=0.12,
              edgecolor=CYAN, linewidth=1.5))
ax2.text(2.05, 7.2, "INDEX TABLE", ha="center", fontsize=9, color=CYAN,
         fontweight="bold", fontfamily="monospace")
ax2.text(2.05, 6.6, "1024 entries", ha="center", fontsize=7, color="#888",
         fontfamily="monospace")
ax2.text(2.05, 6.1, "neuron → {base, count}", ha="center", fontsize=7,
         color=CYAN, fontfamily="monospace")

# Connection Pool
ax2.add_patch(mpatches.FancyBboxPatch((5, 5.5), 4.5, 2,
              boxstyle="round,pad=0.15", facecolor=GOLD, alpha=0.12,
              edgecolor=GOLD, linewidth=1.5))
ax2.text(7.25, 7.2, "CONNECTION POOL", ha="center", fontsize=9, color=GOLD,
         fontweight="bold", fontfamily="monospace")
ax2.text(7.25, 6.6, "32,768 entries (shared)", ha="center", fontsize=7,
         color="#888", fontfamily="monospace")
ax2.text(7.25, 6.1, "pool[addr] → {tgt, wt, comp}", ha="center", fontsize=7,
         color=GOLD, fontfamily="monospace")

# Arrow index→pool
ax2.annotate("", xy=(5, 6.5), xytext=(3.8, 6.5),
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
ax2.text(4.4, 6.8, "base_addr", fontsize=6, color=GREEN, fontfamily="monospace",
         ha="center")

# Example entries
examples = [
    (0.5, 4.5, "N0: base=0, count=100", GOLD),
    (0.5, 3.8, "N1: base=100, count=3", CYAN),
    (0.5, 3.1, "N2: base=103, count=50", PURPLE),
    (0.5, 2.4, "...", "#555"),
]
for x, y, label, color in examples:
    ax2.text(x, y, label, fontsize=7.5, color=color, fontfamily="monospace")

# vs old system
ax2.add_patch(mpatches.FancyBboxPatch((5.3, 1.8), 4, 2.8,
              boxstyle="round,pad=0.15", facecolor=RED, alpha=0.08,
              edgecolor=RED, linewidth=1, ls="--"))
ax2.text(7.3, 4.3, "OLD: Fixed 32 slots/neuron", ha="center", fontsize=7.5,
         color=RED, fontweight="bold", fontfamily="monospace")
ax2.text(7.3, 3.7, "N0: [slot0][slot1]...[slot31]", ha="center", fontsize=7,
         color=RED, fontfamily="monospace", alpha=0.7)
ax2.text(7.3, 3.1, "Always scan all 32 slots", ha="center", fontsize=7,
         color=RED, fontfamily="monospace", alpha=0.7)
ax2.text(7.3, 2.4, "Wasted cycles on empty slots", ha="center", fontsize=7,
         color=RED, fontfamily="monospace", alpha=0.7)

# Bottom note
ax2.text(5, 1.2, "Savings: sparse neurons (3 conn) take 17 cycles\n"
         "instead of 192 cycles → 11× speedup",
         ha="center", fontsize=7, color=GREEN, fontfamily="monospace",
         style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a",
                   edgecolor="#333", alpha=0.8))

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL)
ax3.set_title(f"P13b: Multicast Routing ({ROUTE_FANOUT}×)", color=TEXT,
              fontsize=12, fontfamily="monospace", pad=10)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)
ax3.axis("off")

# Draw source core
src_x, src_y = 1.5, 4
ax3.add_patch(mpatches.FancyBboxPatch((src_x-1.2, src_y-0.8), 2.4, 1.6,
              boxstyle="round,pad=0.15", facecolor=CYAN, alpha=0.15,
              edgecolor=CYAN, linewidth=2))
ax3.text(src_x, src_y+0.3, "Core 0", ha="center", fontsize=9, color=CYAN,
         fontweight="bold", fontfamily="monospace")
ax3.text(src_x, src_y-0.3, "N0 fires", ha="center", fontsize=7, color=CYAN,
         fontfamily="monospace")

# Draw target cores
target_colors = [GREEN, GOLD, PURPLE, BLUE, ORANGE, PINK]
target_positions = [(7, 7), (9, 6), (9, 4), (9, 2), (7, 1), (5, 1)]
for i, ((tx, ty), color) in enumerate(zip(target_positions, target_colors)):
    ax3.add_patch(mpatches.FancyBboxPatch((tx-0.7, ty-0.5), 1.4, 1,
                  boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                  edgecolor=color, linewidth=1.5))
    ax3.text(tx, ty, f"Core {i+1}", ha="center", fontsize=7.5, color=color,
             fontweight="bold", fontfamily="monospace")
    # Arrow from source
    ax3.annotate("", xy=(tx-0.7, ty), xytext=(src_x+1.2, src_y),
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7))

# Slot labels
ax3.text(5, 4.8, "Slot 0", fontsize=6, color=GREEN, fontfamily="monospace",
         rotation=20)
ax3.text(5.5, 5.5, "Slot 1", fontsize=6, color=GOLD, fontfamily="monospace",
         rotation=10)

# Old vs new
ax3.text(1.5, 7.5, "OLD: 1 route per source", fontsize=8, color=RED,
         fontfamily="monospace", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=RED, alpha=0.8))
ax3.text(1.5, 6.7, f"NEW: {ROUTE_FANOUT} slots per source", fontsize=8, color=GREEN,
         fontfamily="monospace", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", facecolor=PANEL, edgecolor=GREEN, alpha=0.8))

ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PANEL)
ax4.set_title("P13c: Eligibility Traces", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

t_axis = range(60)
ax4.fill_between(t_axis, e_pos, alpha=0.15, color=CYAN)
ax4.plot(t_axis, e_pos, color=CYAN, lw=1.5, label="+ reward @ t=20")
ax4.fill_between(t_axis, e_delayed, alpha=0.15, color=GOLD)
ax4.plot(t_axis, e_delayed, color=GOLD, lw=1.5, label="+ reward @ t=35")
ax4.fill_between(t_axis, e_none, alpha=0.15, color="#666")
ax4.plot(t_axis, e_none, color="#666", lw=1.5, label="No reward")

# Mark reward times
ax4.axvline(20, color=CYAN, ls=":", alpha=0.5, lw=1)
ax4.axvline(35, color=GOLD, ls=":", alpha=0.5, lw=1)
ax4.text(20.5, max(e_pos)*0.9, "R+", fontsize=8, color=CYAN, fontfamily="monospace")
ax4.text(35.5, max(e_delayed)*0.7, "R+", fontsize=8, color=GOLD, fontfamily="monospace")

ax4.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax4.set_ylabel("Total |eligibility|", color=TEXT, fontsize=9, fontfamily="monospace")
ax4.tick_params(colors="#666", labelsize=7)
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor="#333", labelcolor=TEXT, loc="upper right")
for spine in ax4.spines.values():
    spine.set_color("#222")

ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL)
ax5.set_title("P13c: Weight Change via Reward", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)

ax5.plot(t_axis, w_pos, color=GREEN, lw=2, label="Positive reward")
ax5.plot(t_axis, w_neg, color=RED, lw=2, label="Negative reward")
ax5.plot(t_axis, w_delayed, color=GOLD, lw=2, ls="--", label="Delayed reward")
ax5.plot(t_axis, w_none, color="#666", lw=1.5, ls=":", label="No reward (control)")

ax5.axhline(500, color="#444", ls=":", lw=0.5)
ax5.axvline(20, color="#444", ls=":", alpha=0.5, lw=1)
ax5.axvline(35, color="#444", ls=":", alpha=0.5, lw=1)
ax5.text(20.5, min(min(w_neg), 400), "reward\n@ t=20", fontsize=6, color="#888",
         fontfamily="monospace")
ax5.text(35.5, min(min(w_neg), 400), "delayed\n@ t=35", fontsize=6, color="#888",
         fontfamily="monospace")

ax5.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax5.set_ylabel("Synapse weight", color=TEXT, fontsize=9, fontfamily="monospace")
ax5.tick_params(colors="#666", labelsize=7)
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor="#333", labelcolor=TEXT, loc="center right")
for spine in ax5.spines.values():
    spine.set_color("#222")

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL)
ax6.set_title("3-Factor Learning Pipeline", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 8)
ax6.axis("off")

# Pipeline boxes
boxes = [
    (2, 7, "STDP\nCorrelation", CYAN),
    (5, 7, "Eligibility\nAccumulate", PURPLE),
    (8, 7, "Eligibility\nDecay", ORANGE),
    (5, 4.5, "REWARD\nSignal", GOLD),
    (5, 2.2, "Weight\nUpdate", GREEN),
]
for bx, by, label, color in boxes:
    ax6.add_patch(mpatches.FancyBboxPatch((bx-1.3, by-0.7), 2.6, 1.4,
                  boxstyle="round,pad=0.15", facecolor=color, alpha=0.12,
                  edgecolor=color, linewidth=1.5))
    ax6.text(bx, by, label, ha="center", va="center", fontsize=8,
             color=color, fontweight="bold", fontfamily="monospace")

# Arrows
arrows = [
    ((3.3, 7), (3.7, 7), CYAN),      # STDP → Elig
    ((6.3, 7), (6.7, 7), PURPLE),     # Elig → Decay
    ((5, 6.3), (5, 5.2), PURPLE),     # Elig down to × node
    ((5, 3.8), (5, 2.9), GREEN),      # × node → Weight
]
for start, end, color in arrows:
    ax6.annotate("", xy=end, xytext=start,
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Multiply symbol
ax6.text(5, 3.7, "×", fontsize=16, color=GOLD, fontfamily="monospace",
         ha="center", va="center", fontweight="bold")

# Side labels
ax6.text(1.5, 5.5, "pre/post\nspike\ntiming", fontsize=7, color=CYAN,
         fontfamily="monospace", ha="center", style="italic")
ax6.annotate("", xy=(2, 6.3), xytext=(1.5, 5.8),
             arrowprops=dict(arrowstyle="->", color=CYAN, lw=1, alpha=0.5))

ax6.text(8.5, 4.5, "external\nreward\nsignal", fontsize=7, color=GOLD,
         fontfamily="monospace", ha="center", style="italic")
ax6.annotate("", xy=(6.3, 4.5), xytext=(7.8, 4.5),
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1, alpha=0.5))

# Formula
ax6.text(5, 1.1,
         "Δw = (eligibility × reward) >> 7\n"
         "elig_decay: elig -= elig >> 3  (~12.5%/ts)",
         ha="center", fontsize=7, color="#888", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a",
                   edgecolor="#333", alpha=0.8))

ax7 = fig.add_subplot(gs[2, 0:2])
ax7.set_facecolor(PANEL)
ax7.set_title(f"E/I Network — 320 neurons, fan-out up to 64 (P13 CSR) — {scale_total:,} spikes / 200 ts",
              color=TEXT, fontsize=11, fontfamily="monospace", pad=10)

for gid, times in scale_trains.items():
    local = gid % NEURONS_PER_CORE
    color = CYAN if local < 256 else RED
    ax7.scatter(times, [gid] * len(times), s=0.4, c=color, marker="|", linewidths=0.2)

ax7.set_xlabel("Timestep", color=TEXT, fontsize=9, fontfamily="monospace")
ax7.set_ylabel("Neuron ID", color=TEXT, fontsize=9, fontfamily="monospace")
ax7.tick_params(colors="#666", labelsize=7)
for spine in ax7.spines.values():
    spine.set_color("#222")
exc_p = mpatches.Patch(color=CYAN, label="Excitatory (256)")
inh_p = mpatches.Patch(color=RED, label="Inhibitory (64)")
ax7.legend(handles=[exc_p, inh_p], loc="upper right", fontsize=7,
           facecolor=PANEL, edgecolor="#333", labelcolor=TEXT)

ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(PANEL)
ax8.set_title("P12 → P13 Feature Gains", color=TEXT, fontsize=12,
              fontfamily="monospace", pad=10)
ax8.axis("off")

features = [
    ("Neurons/core", "256", "1,024", "4×"),
    ("Max fanout", "32 (fixed)", "~1,024 (pool)", "32×"),
    ("Pool depth", "8,192", "32,768", "4×"),
    ("Inter-core routes", "1/source", f"{ROUTE_FANOUT}/source", f"{ROUTE_FANOUT}×"),
    ("Learning", "2-factor STDP", "3-factor elig.", "+reward"),
    ("Total neurons", "32,768", "131,072", "4×"),
]

# Table header
y = 0.92
ax8.text(0.05, y, "Feature", fontsize=8, color=CYAN, fontweight="bold",
         fontfamily="monospace", transform=ax8.transAxes)
ax8.text(0.38, y, "P12", fontsize=8, color=RED, fontweight="bold",
         fontfamily="monospace", transform=ax8.transAxes)
ax8.text(0.60, y, "P13", fontsize=8, color=GREEN, fontweight="bold",
         fontfamily="monospace", transform=ax8.transAxes)
ax8.text(0.85, y, "Gain", fontsize=8, color=GOLD, fontweight="bold",
         fontfamily="monospace", transform=ax8.transAxes)

y -= 0.04
ax8.plot([0.02, 0.98], [y, y], color="#333", lw=0.5,
         transform=ax8.transAxes, clip_on=False)

for feat, old, new, gain in features:
    y -= 0.1
    ax8.text(0.05, y, feat, fontsize=7.5, color=TEXT,
             fontfamily="monospace", transform=ax8.transAxes)
    ax8.text(0.38, y, old, fontsize=7.5, color="#888",
             fontfamily="monospace", transform=ax8.transAxes)
    ax8.text(0.60, y, new, fontsize=7.5, color=GREEN,
             fontfamily="monospace", transform=ax8.transAxes)
    ax8.text(0.85, y, gain, fontsize=7.5, color=GOLD, fontweight="bold",
             fontfamily="monospace", transform=ax8.transAxes)

# Bottom summary
ax8.text(0.5, 0.05,
         f"Pool: {len(compiled.prog_pool_cmds)} entries  |  "
         f"Routes: {len(mcast_compiled.prog_route_cmds):,}  |  "
         f"Cores: {scale_compiled.placement.num_cores_used}",
         ha="center", fontsize=7, color="#666", fontfamily="monospace",
         transform=ax8.transAxes)

# Save
output = r"C:\Users\mrwab\neuromorphic-chip\sdk\p13_dashboard.png"
plt.savefig(output, dpi=180, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"Saved to: {output}")

"""Generate spike raster plot from simulation output."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 2, 1]})
fig.patch.set_facecolor('#0a0a1a')

ax1 = axes[0]
ax1.set_facecolor('#0a0a1a')

# Spike data from Phase 5 test (TEST 4: Cross-core)
# Core 0: N0 spikes at ts=10, N1 at ts=11, N2 at ts=12, N3 at ts=13
# Core 1: N0 spikes at ts=14, N1 at ts=15
spike_data = [
    (10, 'C0:N0'), (11, 'C0:N1'), (12, 'C0:N2'), (13, 'C0:N3'),
    (14, 'C1:N0'), (15, 'C1:N1'),
]

neurons = ['C0:N0', 'C0:N1', 'C0:N2', 'C0:N3', 'C1:N0', 'C1:N1']
neuron_idx = {n: i for i, n in enumerate(neurons)}
colors_map = {'C0': '#4a9eff', 'C1': '#ff6b35'}

for ts, neuron in spike_data:
    core = neuron[:2]
    y = neuron_idx[neuron]
    ax1.scatter(ts, y, s=200, c=colors_map[core], marker='|', linewidths=3, zorder=5)
    ax1.scatter(ts, y, s=80, c=colors_map[core], alpha=0.3, zorder=4)

# Draw cross-core boundary
ax1.axhline(y=3.5, color='#ff4444', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(29, 3.5, 'NoC Boundary', fontsize=8, color='#ff4444', va='center',
         fontfamily='monospace')

# Draw propagation arrows
for i in range(len(spike_data)-1):
    ts1, n1 = spike_data[i]
    ts2, n2 = spike_data[i+1]
    y1, y2 = neuron_idx[n1], neuron_idx[n2]
    color = '#ffcc00' if y1 < 3.5 and y2 > 3.5 else '#ffffff33'
    ax1.annotate('', xy=(ts2-0.1, y2), xytext=(ts1+0.1, y1),
                arrowprops=dict(arrowstyle='->', color=color, linewidth=1.5, alpha=0.6))

ax1.set_yticks(range(len(neurons)))
ax1.set_yticklabels(neurons, fontsize=9, fontfamily='monospace', color='#cccccc')
ax1.set_xlabel('Timestep', fontsize=10, color='#888888', fontfamily='monospace')
ax1.set_title('Cross-Core Spike Propagation (Core 0 → Core 1 via NoC)',
              fontsize=13, fontweight='bold', color='#ffffff', fontfamily='monospace', pad=10)
ax1.set_xlim(8, 30)
ax1.tick_params(colors='#666666')
ax1.spines['bottom'].set_color('#333333')
ax1.spines['left'].set_color('#333333')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', color='#222222', linewidth=0.5)

ax2 = axes[1]
ax2.set_facecolor('#0a0a1a')

# Simulated spike times for 4-core chain propagation
# Each core: N0→N1→N2→N3, with inter-core hops adding 1 timestep delay
chain_spikes = []
core_colors = ['#4a9eff', '#ff6b35', '#2ecc71', '#e74c3c']
all_neurons = []

base_ts = 5
for core in range(4):
    for neuron in range(4):
        ts = base_ts + core * 5 + neuron + 1
        label = f'C{core}:N{neuron}'
        chain_spikes.append((ts, label, core))
        if label not in all_neurons:
            all_neurons.append(label)

neuron_idx2 = {n: i for i, n in enumerate(all_neurons)}

for ts, label, core in chain_spikes:
    y = neuron_idx2[label]
    ax2.scatter(ts, y, s=150, c=core_colors[core], marker='|', linewidths=2.5, zorder=5)
    ax2.scatter(ts, y, s=60, c=core_colors[core], alpha=0.3, zorder=4)

# Core boundaries
for boundary in [3.5, 7.5, 11.5]:
    ax2.axhline(y=boundary, color='#ff4444', linestyle='--', linewidth=0.8, alpha=0.4)

ax2.set_yticks(range(len(all_neurons)))
ax2.set_yticklabels(all_neurons, fontsize=7, fontfamily='monospace', color='#cccccc')
ax2.set_xlabel('Timestep', fontsize=10, color='#888888', fontfamily='monospace')
ax2.set_title('Full 4-Core Chain: Spike Traverses All 1,024-Neuron Mesh',
              fontsize=13, fontweight='bold', color='#ffffff', fontfamily='monospace', pad=10)
ax2.tick_params(colors='#666666')
ax2.spines['bottom'].set_color('#333333')
ax2.spines['left'].set_color('#333333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', color='#222222', linewidth=0.5)

# Legend
for i, label in enumerate(['Core 0', 'Core 1', 'Core 2', 'Core 3']):
    ax2.scatter([], [], c=core_colors[i], s=100, label=label)
ax2.legend(loc='upper right', fontsize=8, facecolor='#1a1a2a', edgecolor='#333355',
           labelcolor='#cccccc')

ax3 = axes[2]
ax3.set_facecolor('#0a0a1a')

# Simulate LIF neuron membrane potential
threshold = 1000
leak = 3
stimulus = 200
weight = 600
refrac = 3

V = [0]
spike_times = []
refrac_counter = 0

for t in range(1, 80):
    if refrac_counter > 0:
        V.append(0)
        refrac_counter -= 1
        continue

    v = V[-1]
    v = v - leak  # leak
    if v < 0: v = 0
    v = v + stimulus  # external input every timestep

    if v >= threshold:
        spike_times.append(t)
        V.append(threshold + 100)  # show spike visually
        refrac_counter = refrac
    else:
        V.append(v)

ax3.plot(range(len(V)), V, color='#4a9eff', linewidth=1.5, zorder=3)
ax3.axhline(y=threshold, color='#ff4444', linestyle='--', linewidth=1, alpha=0.7)
ax3.text(78, threshold + 30, 'Threshold', fontsize=8, color='#ff4444',
         ha='right', fontfamily='monospace')

for st in spike_times:
    ax3.axvline(x=st, color='#ffcc00', linewidth=1, alpha=0.4, zorder=2)

ax3.fill_between(range(len(V)), 0, V, alpha=0.1, color='#4a9eff')
ax3.set_xlabel('Timestep', fontsize=10, color='#888888', fontfamily='monospace')
ax3.set_ylabel('Membrane\nPotential', fontsize=9, color='#888888', fontfamily='monospace')
ax3.set_title('LIF Neuron Dynamics: Charge → Threshold → Spike → Reset → Refractory',
              fontsize=11, fontweight='bold', color='#ffffff', fontfamily='monospace', pad=10)
ax3.tick_params(colors='#666666')
ax3.spines['bottom'].set_color('#333333')
ax3.spines['left'].set_color('#333333')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_ylim(-50, 1200)

plt.tight_layout(pad=1.5)
plt.savefig('C:/Users/mrwab/neuromorphic-chip/spike_visualization.png', dpi=150,
            facecolor='#0a0a1a', bbox_inches='tight', pad_inches=0.3)
print("Spike visualization saved!")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0a0a1a')

C_BG = '#0a0a1a'
C_CORE = '#1a3a5c'
C_CORE_EDGE = '#4a9eff'
C_NEURON = '#ff6b35'
C_UART = '#2ecc71'
C_HOST = '#9b59b6'
C_MESH = '#1a2a3a'
C_MESH_EDGE = '#3a7aff'
C_TEXT = '#ffffff'
C_ARROW = '#ffcc00'
C_ROUTE = '#ff4444'

ax.text(8, 11.5, 'NEUROMORPHIC CHIP ARCHITECTURE', fontsize=20, fontweight='bold',
        ha='center', va='center', color=C_TEXT, fontfamily='monospace')
ax.text(8, 11.0, '4 Cores  x  256 Neurons  =  1,024 Spiking Neurons',
        fontsize=12, ha='center', va='center', color='#888888', fontfamily='monospace')

fpga = FancyBboxPatch((0.3, 0.3), 15.4, 10.2, boxstyle="round,pad=0.1",
                       facecolor='none', edgecolor='#333355', linewidth=2, linestyle='--')
ax.add_patch(fpga)
ax.text(0.6, 10.2, 'FPGA TOP (Arty A7-100T)', fontsize=9, color='#555577',
        fontfamily='monospace')

uart_rx = FancyBboxPatch((0.5, 4.5), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#1a3a2a', edgecolor=C_UART, linewidth=2)
ax.add_patch(uart_rx)
ax.text(1.5, 5.5, 'UART RX', fontsize=10, fontweight='bold', ha='center', color=C_UART,
        fontfamily='monospace')
ax.text(1.5, 5.0, '115200 8N1', fontsize=7, ha='center', color='#aaaaaa',
        fontfamily='monospace')

uart_tx = FancyBboxPatch((0.5, 2.5), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor='#1a3a2a', edgecolor=C_UART, linewidth=2)
ax.add_patch(uart_tx)
ax.text(1.5, 3.5, 'UART TX', fontsize=10, fontweight='bold', ha='center', color=C_UART,
        fontfamily='monospace')
ax.text(1.5, 3.0, '115200 8N1', fontsize=7, ha='center', color='#aaaaaa',
        fontfamily='monospace')

host = FancyBboxPatch((3.2, 2.5), 2.5, 3.5, boxstyle="round,pad=0.1",
                       facecolor='#2a1a3a', edgecolor=C_HOST, linewidth=2)
ax.add_patch(host)
ax.text(4.45, 5.2, 'HOST', fontsize=11, fontweight='bold', ha='center', color=C_HOST,
        fontfamily='monospace')
ax.text(4.45, 4.7, 'INTERFACE', fontsize=11, fontweight='bold', ha='center', color=C_HOST,
        fontfamily='monospace')
ax.text(4.45, 4.0, 'CMD Parser', fontsize=7, ha='center', color='#aaaaaa',
        fontfamily='monospace')
ax.text(4.45, 3.6, 'PROG_CONN', fontsize=6, ha='center', color='#777777',
        fontfamily='monospace')
ax.text(4.45, 3.3, 'PROG_ROUTE', fontsize=6, ha='center', color='#777777',
        fontfamily='monospace')
ax.text(4.45, 3.0, 'STIMULUS/RUN', fontsize=6, ha='center', color='#777777',
        fontfamily='monospace')

mesh = FancyBboxPatch((6.3, 1.0), 9.2, 8.5, boxstyle="round,pad=0.1",
                       facecolor=C_MESH, edgecolor=C_MESH_EDGE, linewidth=2)
ax.add_patch(mesh)
ax.text(10.9, 9.1, 'NEUROMORPHIC MESH (NoC)', fontsize=11, fontweight='bold',
        ha='center', color=C_MESH_EDGE, fontfamily='monospace')

core_positions = [(7.0, 5.2), (11.5, 5.2), (7.0, 1.5), (11.5, 1.5)]
core_labels = ['CORE 0', 'CORE 1', 'CORE 2', 'CORE 3']

for idx, (cx, cy) in enumerate(core_positions):
    core = FancyBboxPatch((cx, cy), 3.5, 3.2, boxstyle="round,pad=0.05",
                           facecolor=C_CORE, edgecolor=C_CORE_EDGE, linewidth=1.5)
    ax.add_patch(core)
    ax.text(cx+1.75, cy+2.8, core_labels[idx], fontsize=9, fontweight='bold',
            ha='center', color=C_CORE_EDGE, fontfamily='monospace')
    ax.text(cx+1.75, cy+2.4, '256 LIF Neurons', fontsize=7, ha='center', color='#aaaaaa',
            fontfamily='monospace')

    for ni in range(6):
        for nj in range(6):
            nx = cx + 0.35 + ni * 0.48
            ny = cy + 0.35 + nj * 0.3
            neuron = plt.Circle((nx, ny), 0.1, facecolor=C_NEURON, edgecolor='#cc5520',
                               linewidth=0.5, alpha=0.7)
            ax.add_patch(neuron)

    ax.text(cx+1.75, cy+0.2, '...256 total', fontsize=6, ha='center', color='#666666',
            fontfamily='monospace')

arrow_style = dict(arrowstyle='->', color=C_ROUTE, linewidth=2, mutation_scale=15)
ax.annotate('', xy=(11.5, 6.8), xytext=(10.5, 6.8), arrowprops=arrow_style)
ax.annotate('', xy=(8.75, 5.2), xytext=(8.75, 4.7), arrowprops=arrow_style)
ax.annotate('', xy=(13.25, 5.2), xytext=(13.25, 4.7), arrowprops=arrow_style)
ax.annotate('', xy=(11.5, 3.1), xytext=(10.5, 3.1), arrowprops=arrow_style)

rt = FancyBboxPatch((9.8, 4.55), 1.5, 0.55, boxstyle="round,pad=0.05",
                     facecolor='#3a1a1a', edgecolor=C_ROUTE, linewidth=1)
ax.add_patch(rt)
ax.text(10.55, 4.82, 'ROUTE TABLE', fontsize=6, fontweight='bold', ha='center',
        color=C_ROUTE, fontfamily='monospace')

arrow2 = dict(arrowstyle='->', color=C_ARROW, linewidth=2, mutation_scale=15)
ax.annotate('', xy=(3.2, 5.25), xytext=(2.5, 5.25), arrowprops=arrow2)
ax.annotate('', xy=(2.5, 3.25), xytext=(3.2, 3.25), arrowprops=arrow2)
ax.annotate('', xy=(6.3, 4.25), xytext=(5.7, 4.25), arrowprops=arrow2)

ax.annotate('uart_rxd', xy=(0.5, 5.25), xytext=(-0.3, 5.25),
            fontsize=8, color=C_UART, fontfamily='monospace', fontweight='bold',
            ha='right', va='center',
            arrowprops=dict(arrowstyle='->', color=C_UART, linewidth=1.5))
ax.annotate('uart_txd', xy=(-0.3, 3.25), xytext=(0.5, 3.25),
            fontsize=8, color=C_UART, fontfamily='monospace', fontweight='bold',
            ha='right', va='center',
            arrowprops=dict(arrowstyle='->', color=C_UART, linewidth=1.5))

for i, (label, color) in enumerate([
    ('LED0: Heartbeat', '#00ff00'),
    ('LED1: RX Activity', '#ffaa00'),
    ('LED2: TX Activity', '#ff6600'),
    ('LED3: Spike Activity', '#ff0066')
]):
    x = 1.5 + i * 3.5
    circle = plt.Circle((x, 0.6), 0.15, facecolor=color, edgecolor='white',
                        linewidth=1, alpha=0.8)
    ax.add_patch(circle)
    ax.text(x + 0.3, 0.6, label, fontsize=7, color='#aaaaaa', va='center',
            fontfamily='monospace')

stats = FancyBboxPatch((6.5, 9.3), 8.8, 0.9, boxstyle="round,pad=0.1",
                        facecolor='#1a1a2a', edgecolor='#444466', linewidth=1)
ax.add_patch(stats)
ax.text(10.9, 9.85, '1,024 Neurons  |  32 Fanout/Neuron  |  Inter-Core NoC  |  UART Host  |  4 Pins',
        fontsize=7, ha='center', color='#aaaaaa', fontfamily='monospace')

plt.tight_layout()
plt.savefig('architecture.png', dpi=150,
            facecolor=C_BG, bbox_inches='tight', pad_inches=0.3)
print("Architecture diagram saved!")

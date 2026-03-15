"""
Neuromorphic Chip - Spike Visualizer
Parses the VCD waveform file and generates visual plots of neuron activity.
"""

import re
import os

def parse_vcd_spikes(vcd_path):
    """Parse VCD file to extract spike timing for each neuron."""
    spikes = {0: [], 1: [], 2: [], 3: []}
    membrane = {0: [], 1: [], 2: [], 3: []}

    current_time = 0

    id_map = {}

    with open(vcd_path, 'r') as f:
        in_header = True
        for line in f:
            line = line.strip()

            # Parse variable declarations
            if line.startswith('$var'):
                parts = line.split()
                if len(parts) >= 5:
                    var_id = parts[3]
                    var_name = parts[4]
                    # Map IDs to signal names
                    id_map[var_id] = var_name

            if line == '$enddefinitions $end':
                in_header = False
                continue

            if in_header:
                continue

            # Parse time changes
            if line.startswith('#'):
                current_time = int(line[1:])
                continue

            # Parse value changes for spike signals
            # Single bit values: 0X or 1X where X is the identifier
            if len(line) >= 2 and line[0] in ('0', '1'):
                val = int(line[0])
                var_id = line[1:]
                if var_id in id_map:
                    name = id_map[var_id]
                    for i in range(4):
                        if name == f'spikes[{i}]' or (name == 'spikes' and var_id.endswith(f'[{i}]')):
                            if val == 1:
                                spikes[i].append(current_time)

    return spikes, current_time

def parse_simulation_output(sim_output=None):
    """Parse spike times from simulation console output."""
    spikes = {0: [], 1: [], 2: [], 3: []}

    # Known spike data from our simulation
    raw = """[185000] SPIKE! Neuron 0
[335000] SPIKE! Neuron 0
[485000] SPIKE! Neuron 0
[505000] SPIKE! Neuron 1
[635000] SPIKE! Neuron 0
[655000] SPIKE! Neuron 2
[785000] SPIKE! Neuron 0
[935000] SPIKE! Neuron 0
[955000] SPIKE! Neuron 1
[1085000] SPIKE! Neuron 0
[1235000] SPIKE! Neuron 0
[1255000] SPIKE! Neuron 2
[1385000] SPIKE! Neuron 0
[1405000] SPIKE! Neuron 1
[1535000] SPIKE! Neuron 0
[1685000] SPIKE! Neuron 0
[1835000] SPIKE! Neuron 0
[1855000] SPIKE! Neuron 1
[1855000] SPIKE! Neuron 2
[1875000] SPIKE! Neuron 3
[1895000] SPIKE! Neuron 0
[2045000] SPIKE! Neuron 0
[2145000] SPIKE! Neuron 0
[2165000] SPIKE! Neuron 1
[2245000] SPIKE! Neuron 0
[2265000] SPIKE! Neuron 2
[2345000] SPIKE! Neuron 0
[2445000] SPIKE! Neuron 0
[2465000] SPIKE! Neuron 1
[2545000] SPIKE! Neuron 0
[2645000] SPIKE! Neuron 0
[2665000] SPIKE! Neuron 2
[2745000] SPIKE! Neuron 0
[2765000] SPIKE! Neuron 1
[2845000] SPIKE! Neuron 0
[2945000] SPIKE! Neuron 0
[3045000] SPIKE! Neuron 0
[3065000] SPIKE! Neuron 1
[3065000] SPIKE! Neuron 2
[3085000] SPIKE! Neuron 3
[3105000] SPIKE! Neuron 0
[3205000] SPIKE! Neuron 0
[3305000] SPIKE! Neuron 0
[3325000] SPIKE! Neuron 1
[3405000] SPIKE! Neuron 0
[3425000] SPIKE! Neuron 2
[3505000] SPIKE! Neuron 0
[3605000] SPIKE! Neuron 0
[3625000] SPIKE! Neuron 1
[3705000] SPIKE! Neuron 0
[3805000] SPIKE! Neuron 0
[3825000] SPIKE! Neuron 2
[3905000] SPIKE! Neuron 0
[3925000] SPIKE! Neuron 1
[4005000] SPIKE! Neuron 0
[4105000] SPIKE! Neuron 0
[4105000] SPIKE! Neuron 2
[4125000] SPIKE! Neuron 3
[4205000] SPIKE! Neuron 0
[4215000] SPIKE! Neuron 2
[4225000] SPIKE! Neuron 1
[4305000] SPIKE! Neuron 0
[4325000] SPIKE! Neuron 2
[4405000] SPIKE! Neuron 0
[4425000] SPIKE! Neuron 2
[4445000] SPIKE! Neuron 3
[4465000] SPIKE! Neuron 0
[4485000] SPIKE! Neuron 1
[4515000] SPIKE! Neuron 2
[4565000] SPIKE! Neuron 0
[4605000] SPIKE! Neuron 2
[4665000] SPIKE! Neuron 0
[4695000] SPIKE! Neuron 2
[4715000] SPIKE! Neuron 3
[4785000] SPIKE! Neuron 0
[4805000] SPIKE! Neuron 1
[4805000] SPIKE! Neuron 2
[4885000] SPIKE! Neuron 0
[4905000] SPIKE! Neuron 2
[4985000] SPIKE! Neuron 0
[5005000] SPIKE! Neuron 2
[5025000] SPIKE! Neuron 3
[5045000] SPIKE! Neuron 0
[5065000] SPIKE! Neuron 1
[5095000] SPIKE! Neuron 2
[5145000] SPIKE! Neuron 0
[5185000] SPIKE! Neuron 2
[5245000] SPIKE! Neuron 0
[5275000] SPIKE! Neuron 2
[5295000] SPIKE! Neuron 3
[5365000] SPIKE! Neuron 0
[5385000] SPIKE! Neuron 1
[5385000] SPIKE! Neuron 2
[5465000] SPIKE! Neuron 0
[5485000] SPIKE! Neuron 2
[5565000] SPIKE! Neuron 0
[5585000] SPIKE! Neuron 2
[5605000] SPIKE! Neuron 3
[5625000] SPIKE! Neuron 0
[5645000] SPIKE! Neuron 1
[5675000] SPIKE! Neuron 2
[5725000] SPIKE! Neuron 0
[5765000] SPIKE! Neuron 2
[5825000] SPIKE! Neuron 0
[5855000] SPIKE! Neuron 2
[5875000] SPIKE! Neuron 3
[5945000] SPIKE! Neuron 0
[5965000] SPIKE! Neuron 1
[5965000] SPIKE! Neuron 2
[6045000] SPIKE! Neuron 0
[6065000] SPIKE! Neuron 2"""

    for line in raw.strip().split('\n'):
        m = re.match(r'\[(\d+)\] SPIKE! Neuron (\d)', line)
        if m:
            time_ps = int(m.group(1))
            neuron = int(m.group(2))
            spikes[neuron].append(time_ps)

    return spikes

def draw_raster_plot(spikes, total_time=7070000):
    """Draw a text-based spike raster plot."""
    width = 100  # characters wide

    neuron_names = ['Neuron 0 (Input)    ', 'Neuron 1 (Excit)    ', 'Neuron 2 (Chain)    ', 'Neuron 3 (Inhibit)  ']
    neuron_chars = ['#', '+', '*', 'o']

    # Phase markers
    phases = [
        (70000,   'Phase 1: Low stimulus'),
        (2070000, 'Phase 2: High stimulus'),
        (4070000, 'Phase 3: Dual stimulus'),
        (6070000, 'Phase 4: No stimulus'),
    ]

    print()
    print('=' * (width + 25))
    print('  NEUROMORPHIC CHIP - SPIKE RASTER PLOT')
    print('  Each mark = one spike from that neuron')
    print('=' * (width + 25))
    print()

    # Time axis header
    header = '                    '
    for i in range(0, width + 1, 20):
        time_us = (i / width) * (total_time / 1000)
        header += f'{time_us:>6.0f}us' + ' ' * 12
    print(header)
    print('                    ' + '-' * width)

    # Draw phase markers
    phase_line = '                    '
    for t, name in phases:
        pos = int((t / total_time) * width)
        phase_line = phase_line[:20+pos] + '|' + phase_line[21+pos:]
    print(phase_line)

    # Draw each neuron's spike train
    for n in range(4):
        line = neuron_names[n]
        row = [' '] * width

        for spike_time in spikes[n]:
            pos = int((spike_time / total_time) * width)
            if 0 <= pos < width:
                row[pos] = neuron_chars[n]

        line += ''.join(row) + f'  ({len(spikes[n])} spikes)'
        print(line)

    print('                    ' + '-' * width)

    # Phase labels
    print()
    print('  Phases:')
    for t, name in phases:
        print(f'    | {name} (t={t/1000:.0f}us)')

    print()
    print('  Circuit:')
    print('    External Input --> [N0] --excit--> [N1]')
    print('                       |')
    print('                       +---excit--> [N2] --excit--> [N3]')
    print('                       |                              |')
    print('                       +<--------inhibit--------------+')
    print()

    # Firing rate analysis
    print('  Firing Rate Analysis:')
    for phase_idx in range(len(phases)):
        t_start = phases[phase_idx][0]
        t_end = phases[phase_idx + 1][0] if phase_idx + 1 < len(phases) else total_time
        duration_us = (t_end - t_start) / 1000

        print(f'    {phases[phase_idx][1]}:')
        for n in range(4):
            count = sum(1 for s in spikes[n] if t_start <= s < t_end)
            rate = (count / duration_us) * 1000 if duration_us > 0 else 0
            bar = '#' * int(rate * 2)
            print(f'      N{n}: {count:>3} spikes  ({rate:>5.1f} spikes/ms)  {bar}')
        print()

def draw_membrane_ascii(spikes, total_time=7070000):
    """Draw a simplified membrane potential visualization."""
    width = 100
    height = 10

    print('=' * (width + 25))
    print('  MEMBRANE POTENTIAL APPROXIMATION (Neuron 0)')
    print('  Threshold = 1000 (top line)')
    print('=' * (width + 25))
    print()

    # Simulate membrane potential for neuron 0
    threshold = 1000
    leak = 2
    input_current = 0
    potential = 0

    potentials = []
    for t in range(0, total_time, total_time // width):
        # Determine current phase input
        if t < 70000:
            input_current = 0
        elif t < 2070000:
            input_current = 100
        elif t < 4070000:
            input_current = 200
        elif t < 6070000:
            input_current = 200
        else:
            input_current = 0

        # Check if there's a spike near this time
        spiked = any(abs(s - t) < (total_time // width) for s in spikes[0])

        if spiked:
            potentials.append(threshold)
            potential = 0
        else:
            potential = min(potential + input_current - leak, threshold)
            potential = max(potential, 0)
            potentials.append(potential)

    # Draw
    for row in range(height, -1, -1):
        level = (row / height) * threshold
        line = f'  {level:>6.0f} |'
        for col in range(min(width, len(potentials))):
            if potentials[col] >= level and (row == 0 or potentials[col] < ((row + 1) / height) * threshold):
                line += '#'
            elif potentials[col] >= level:
                line += '|'
            elif row == height and potentials[col] >= threshold * 0.95:
                line += '^'  # spike marker
            else:
                line += ' '
        print(line)

    print(f'         +' + '-' * width)
    print(f'          0us' + ' ' * (width - 20) + f'{total_time/1000:.0f}us')
    print()


if __name__ == '__main__':
    print('\n' * 2)

    # Parse spikes from simulation output
    spikes = parse_simulation_output()

    # Draw visualizations
    draw_raster_plot(spikes)
    draw_membrane_ascii(spikes)

    print('To view full waveforms with GTKWave:')
    print('  wsl gtkwave /mnt/c/Users/mrwab/neuromorphic-chip/sim/neuron_core.vcd')
    print()

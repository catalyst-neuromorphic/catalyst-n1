"""Simulator vs Chip Comparison Benchmark
==========================================
Demonstrates both backends with the same network, comparing spike counts.

When no FPGA is connected, runs simulator-only and shows expected chip commands.

Features demonstrated: Backend abstraction, deploy/inject/run API, RunResult.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.compiler import Compiler


def build_network():
    """Build a moderately complex E/I network."""
    net = nc.Network()
    exc = net.population(64, params={
        "threshold": 800, "leak": 5, "refrac": 3,
    }, label="excitatory")
    inh = net.population(16, params={
        "threshold": 600, "leak": 2, "refrac": 2,
    }, label="inhibitory")

    net.connect(exc, exc, topology="random_sparse", p=0.1, weight=200, seed=42)
    net.connect(exc, inh, topology="all_to_all", weight=150)
    net.connect(inh, exc, topology="all_to_all", weight=-300, compartment=0)

    return net, exc, inh


def run_simulator(net, exc, inh, timesteps=100):
    """Run on the software simulator."""
    sim = nc.Simulator()
    sim.deploy(net)

    # Inject stimulus to first 8 excitatory neurons
    sim.inject(exc[:8], current=1200)
    result = sim.run(timesteps)
    return result


def main():
    print("=" * 60)
    print("  Simulator vs Chip Comparison Benchmark")
    print("=" * 60)

    net, exc, inh = build_network()
    timesteps = 100

    # Compile and show network summary
    compiled = Compiler().compile(net)
    print(f"\nNetwork: {net.total_neurons()} neurons "
          f"({net.populations[0].size} exc + {net.populations[1].size} inh)")
    print(f"Compiled: {compiled.summary()}")

    # Run simulator
    print(f"\n--- Simulator ({timesteps} timesteps) ---")
    t0 = time.perf_counter()
    result = run_simulator(net, exc, inh, timesteps)
    elapsed = time.perf_counter() - t0

    print(f"Total spikes: {result.total_spikes}")
    print(f"Active neurons: {len(result.spike_trains)}/{net.total_neurons()}")
    print(f"Elapsed: {elapsed * 1000:.1f}ms")

    rates = result.firing_rates()
    if rates:
        max_rate = max(rates.values())
        avg_rate = sum(rates.values()) / len(rates)
        print(f"Max firing rate: {max_rate:.2f} Hz")
        print(f"Avg firing rate: {avg_rate:.2f} Hz (active neurons only)")

    timeseries = result.spike_count_timeseries()
    peak_t = max(range(len(timeseries)), key=lambda i: timeseries[i])
    print(f"Peak activity: timestep {peak_t} ({timeseries[peak_t]} spikes)")

    # Show what would be sent to FPGA
    print(f"\n--- Chip Commands (would be sent via UART) ---")
    print(f"PROG_NEURON commands: {len(compiled.prog_neuron_cmds)}")
    print(f"PROG_INDEX commands:  {len(compiled.prog_index_cmds)}")
    print(f"PROG_POOL commands:   {len(compiled.prog_pool_cmds)}")
    print(f"PROG_ROUTE commands:  {len(compiled.prog_route_cmds)}")
    print(f"PROG_DELAY commands:  {len(compiled.prog_delay_cmds)}")
    total_bytes = (len(compiled.prog_neuron_cmds) * 7
                   + len(compiled.prog_index_cmds) * 10
                   + len(compiled.prog_pool_cmds) * 9
                   + len(compiled.prog_route_cmds) * 10)
    print(f"Total deploy payload: ~{total_bytes} bytes")

    # Try chip backend (will fail without hardware)
    print(f"\n--- Chip Backend ---")
    try:
        chip = nc.Chip(port="COM3")
        chip.deploy(net)
        chip.inject(exc[:8], current=1200)
        chip_result = chip.run(timesteps)
        print(f"Chip spikes: {chip_result.total_spikes}")
        print(f"Match: {'YES' if chip_result.total_spikes == result.total_spikes else 'NO'}")
        chip.close()
    except Exception as e:
        print(f"No FPGA connected ({type(e).__name__})")
        print("  Run with --port <port> when FPGA is attached")

    print("\nDone!")


if __name__ == "__main__":
    main()

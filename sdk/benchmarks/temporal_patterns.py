"""Temporal Patterns Benchmark
==============================
Demonstrates P17 axon delays for temporal pattern detection.

A source population sends spikes through connections with varying delays,
causing target neurons to receive coincident inputs at different times.

Features demonstrated: Axon delays, spike timing, temporal coding.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE


def main():
    print("=" * 60)
    print("  Temporal Pattern Detection Benchmark (P17 Delays)")
    print("=" * 60)

    net = nc.Network()

    # Input neurons fire at different times via stimulus timing
    inputs = net.population(4, params={"threshold": 100, "leak": 0, "refrac": 5},
                            label="inputs")

    # Coincidence detector: fires only when multiple delayed inputs arrive together
    detector = net.population(1, params={"threshold": 800, "leak": 50, "refrac": 3},
                              label="detector")

    # Each input has a different delay so they arrive at the detector simultaneously
    # Input 0: delay=5, Input 1: delay=3, Input 2: delay=1, Input 3: delay=0
    for i, delay in enumerate([5, 3, 1, 0]):
        # Connect individual input neuron to detector
        src_slice = inputs[i]
        # Use a separate connection for each delay value
        net.connect(inputs, detector, topology="one_to_one",
                    weight=300, delay=delay) if i == 0 else None

    net2 = nc.Network()
    i0 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in0")
    i1 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in1")
    i2 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in2")
    i3 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in3")
    det = net2.population(1, params={"threshold": 800, "leak": 50, "refrac": 3},
                          label="detector")

    # Different delays: if inputs fire at the same time, arrivals stagger
    # If inputs fire in sequence (i0@t0, i1@t2, i2@t4, i3@t5),
    # with delays (5,3,1,0), all arrive at t=5 -> coincidence!
    net2.connect(i0, det, topology="all_to_all", weight=300, delay=5)
    net2.connect(i1, det, topology="all_to_all", weight=300, delay=3)
    net2.connect(i2, det, topology="all_to_all", weight=300, delay=1)
    net2.connect(i3, det, topology="all_to_all", weight=300, delay=0)

    sim = nc.Simulator()
    sim.deploy(net2)

    # Test 1: Staggered inputs that arrive simultaneously at detector
    print("\nTest 1: Temporally coded pattern (inputs staggered to coincide)")
    sim.inject(i0, current=200)  # fires at t=0
    sim.run(2)
    sim.inject(i1, current=200)  # fires at t=2
    sim.run(2)
    sim.inject(i2, current=200)  # fires at t=4
    sim.run(1)
    sim.inject(i3, current=200)  # fires at t=5
    result = sim.run(10)

    p = result.placement
    det_gid = p.neuron_map[(det.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(det.id, 0)][1]
    det_spikes = result.spike_trains.get(det_gid, [])
    print(f"  Detector spikes: {len(det_spikes)} (expect >= 1 from coincidence)")

    # Test 2: Simultaneous inputs (arrive at different times -> no coincidence)
    sim2 = nc.Simulator()
    sim2.deploy(net2)
    print("\nTest 2: Simultaneous inputs (delays spread arrivals)")
    sim2.inject(i0, current=200)
    sim2.inject(i1, current=200)
    sim2.inject(i2, current=200)
    sim2.inject(i3, current=200)
    result2 = sim2.run(15)
    det_spikes2 = result2.spike_trains.get(det_gid, [])
    print(f"  Detector spikes: {len(det_spikes2)} (spread arrivals, may or may not fire)")

    # Summary
    print(f"\nNetwork: {net2.total_neurons()} neurons, "
          f"4 delay connections (0,1,3,5 timesteps)")
    print("Done!")


if __name__ == "__main__":
    main()

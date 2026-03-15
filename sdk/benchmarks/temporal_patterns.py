import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE

def main():
    print("Temporal Pattern Detection Benchmark (P17 Delays)")

    net = nc.Network()
    i0 = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in0")
    i1 = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in1")
    i2 = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in2")
    i3 = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 10}, label="in3")
    det = net.population(1, params={"threshold": 800, "leak": 50, "refrac": 3},
                          label="detector")

    net.connect(i0, det, topology="all_to_all", weight=300, delay=5)
    net.connect(i1, det, topology="all_to_all", weight=300, delay=3)
    net.connect(i2, det, topology="all_to_all", weight=300, delay=1)
    net.connect(i3, det, topology="all_to_all", weight=300, delay=0)

    sim = nc.Simulator()
    sim.deploy(net)

    print("\nTest 1: Temporally coded pattern (inputs staggered to coincide)")
    sim.inject(i0, current=200)
    sim.run(2)
    sim.inject(i1, current=200)
    sim.run(2)
    sim.inject(i2, current=200)
    sim.run(1)
    sim.inject(i3, current=200)
    result = sim.run(10)

    p = result.placement
    det_gid = p.neuron_map[(det.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(det.id, 0)][1]
    det_spikes = result.spike_trains.get(det_gid, [])
    print(f"  Detector spikes: {len(det_spikes)} (expect >= 1 from coincidence)")

    sim2 = nc.Simulator()
    sim2.deploy(net)
    print("\nTest 2: Simultaneous inputs (delays spread arrivals)")
    sim2.inject(i0, current=200)
    sim2.inject(i1, current=200)
    sim2.inject(i2, current=200)
    sim2.inject(i3, current=200)
    result2 = sim2.run(15)
    det_spikes2 = result2.spike_trains.get(det_gid, [])
    print(f"  Detector spikes: {len(det_spikes2)} (spread arrivals, may or may not fire)")

    print(f"\nNetwork: {net.total_neurons()} neurons, "
          f"4 delay connections (0,1,3,5 timesteps)")
    print("Done!")

if __name__ == "__main__":
    main()

import os
import sys
import time
import argparse
import numpy as np

_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SDK_DIR not in sys.path:
    sys.path.insert(0, _SDK_DIR)

import neurocore as nc
from neurocore.simulator import Simulator
from neurocore.constants import (
    NEURONS_PER_CORE, WEIGHT_MIN, WEIGHT_MAX,
    DEFAULT_THRESHOLD, DEFAULT_LEAK,
)

def test_all_core_saturation(num_cores=16, timesteps=1000):
    print(f"\nTest: All-Core Saturation ({num_cores} cores, {timesteps} ts)")
    net = nc.Network()

    pops = []
    for c in range(num_cores):
        pop = net.population(
            NEURONS_PER_CORE,
            params={"threshold": 100, "leak": 0, "refrac": 0},
            label=f"core_{c}",
        )
        pops.append(pop)

    sim = Simulator(num_cores=num_cores)
    sim.deploy(net)

    total_neurons = num_cores * NEURONS_PER_CORE
    total_spikes = 0
    t_start = time.perf_counter()

    for t in range(timesteps):
        for pop in pops:
            sim.inject(pop, current=200)
        result = sim.run(1)
        total_spikes += result.total_spikes

    elapsed = time.perf_counter() - t_start
    ts_per_sec = timesteps / elapsed

    expected_min = total_neurons * timesteps * 0.9
    print(f"  Neurons: {total_neurons}")
    print(f"  Total spikes: {total_spikes:,} (expected ~{total_neurons * timesteps:,})")
    print(f"  Throughput: {ts_per_sec:.0f} ts/sec")
    print(f"  Elapsed: {elapsed:.1f}s")

    assert total_spikes >= expected_min, \
        f"Expected at least {expected_min:,} spikes, got {total_spikes:,}"
    print("  PASSED")
    return True

def test_long_running_stability(timesteps=10000):
    print(f"\nTest: Long-Running Stability ({timesteps} ts)")
    net = nc.Network()
    exc = net.population(64, params={"threshold": 500, "leak": 3, "refrac": 2})
    inh = net.population(16, params={"threshold": 300, "leak": 5, "refrac": 1})
    net.connect(exc, exc, topology="random_sparse", weight=100, p=0.1, seed=42)
    net.connect(exc, inh, topology="all_to_all", weight=200)
    net.connect(inh, exc, topology="all_to_all", weight=-150)

    sim = Simulator()
    sim.deploy(net)

    total_spikes = 0
    spike_history = []
    t_start = time.perf_counter()

    for t in range(timesteps):
        if t < 100:
            sim.inject(exc[:8], current=600)
        result = sim.run(1)
        total_spikes += result.total_spikes
        if t % 1000 == 0:
            spike_history.append(total_spikes)

    elapsed = time.perf_counter() - t_start
    print(f"  Total spikes: {total_spikes:,}")
    print(f"  Throughput: {timesteps / elapsed:.0f} ts/sec")

    for i in range(sim._n):
        assert 0 <= sim._potential[i] <= 65535, \
            f"Neuron {i} potential {sim._potential[i]} out of range"

    assert not np.any(np.isnan(sim._potential.astype(float))), "NaN in potentials"
    assert not np.any(np.isnan(sim._trace.astype(float))), "NaN in traces"

    print(f"  Elapsed: {elapsed:.1f}s")
    print("  PASSED")
    return True

def test_max_fan_out():
    print("\nTest: Max Fan-Out (1 -> 1023)")
    net = nc.Network()
    src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    tgt = net.population(1023, params={"threshold": 100, "leak": 0, "refrac": 0})
    net.connect(src, tgt, topology="all_to_all", weight=200)

    sim = Simulator()
    sim.deploy(net)

    sim.inject(src, current=200)
    sim.run(1)
    result = sim.run(1)

    print(f"  Connections: 1 -> 1023")
    print(f"  Spikes on delivery timestep: {result.total_spikes}")

    assert result.total_spikes >= 1023, \
        f"Expected >= 1023 spikes, got {result.total_spikes}"
    print("  PASSED")
    return True

def test_weight_extremes():
    print("\nTest: Weight Extremes")

    net = nc.Network()
    src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    tgt = net.population(1, params={"threshold": 30000, "leak": 0, "refrac": 0})
    net.connect(src, tgt, weight=WEIGHT_MAX)

    sim = Simulator()
    sim.deploy(net)
    sim.inject(src, current=200)
    sim.run(1)
    result = sim.run(1)
    assert result.total_spikes >= 1, f"Max positive weight should cause spike, got {result.total_spikes}"
    print(f"  Max positive weight ({WEIGHT_MAX}): PASS")

    net2 = nc.Network()
    src2 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    tgt2 = net2.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    net2.connect(src2, tgt2, weight=WEIGHT_MIN)

    sim2 = Simulator()
    sim2.deploy(net2)
    sim2.inject(tgt2, current=50)
    sim2.run(1)
    sim2.inject(src2, current=200)
    sim2.run(1)
    sim2.run(1)
    tgt_core, tgt_neuron = sim2._compiled.placement.neuron_map[(tgt2.id, 0)]
    tgt_gid = tgt_core * 1024 + tgt_neuron
    assert sim2._potential[tgt_gid] == 0, \
        f"Negative weight should clamp to 0, got {sim2._potential[tgt_gid]}"
    print(f"  Max negative weight ({WEIGHT_MIN}): PASS")

    net3 = nc.Network()
    src3 = net3.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    tgt3 = net3.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
    net3.connect(src3, tgt3, weight=0)

    sim3 = Simulator()
    sim3.deploy(net3)
    sim3.inject(src3, current=200)
    sim3.run(1)
    result3 = sim3.run(5)
    tgt_core3, tgt_neuron3 = sim3._compiled.placement.neuron_map[(tgt3.id, 0)]
    tgt_gid3 = tgt_core3 * 1024 + tgt_neuron3
    assert sim3._potential[tgt_gid3] == 0, \
        f"Zero weight should not charge target, got {sim3._potential[tgt_gid3]}"
    print(f"  Zero weight: PASS")

    print("  PASSED")
    return True

def test_pool_depth_fill():
    print("\nTest: Pool Depth Fill")
    net = nc.Network()
    src = net.population(64, params={"threshold": 100, "leak": 0, "refrac": 0})
    tgt = net.population(500, params={"threshold": 100, "leak": 0, "refrac": 0})
    net.connect(src, tgt, topology="all_to_all", weight=200)

    sim = Simulator()
    sim.deploy(net)

    total_pool_entries = sum(len(v) for v in sim._compiled.adjacency.values())
    print(f"  Pool entries used: {total_pool_entries:,}")
    print(f"  Neurons: {sim._compiled.placement.total_neurons}")

    sim.inject(src[:4], current=200)
    result = sim.run(2)
    print(f"  Spikes in 2 ts: {result.total_spikes}")
    assert result.total_spikes > 0, "Should produce spikes"
    print("  PASSED")
    return True

def test_cross_core_chain(num_cores=16):
    print(f"\nTest: Cross-Core Chain ({num_cores} cores)")
    net = nc.Network()

    relays = []
    for c in range(num_cores):
        relay = net.population(
            1,
            params={"threshold": 100, "leak": 0, "refrac": 2},
            label=f"relay_{c}",
        )
        relays.append(relay)
        if c < num_cores - 1:
            net.population(NEURONS_PER_CORE - 1, label=f"filler_{c}")

    for i in range(num_cores - 1):
        net.connect(relays[i], relays[i + 1], topology="all_to_all", weight=200)

    sim = Simulator(num_cores=num_cores)
    sim.deploy(net)

    sim.inject(relays[0], current=200)

    total_spikes = 0
    for t in range(num_cores * 2 + 5):
        result = sim.run(1)
        total_spikes += result.total_spikes

    print(f"  Total spikes through {num_cores}-core chain: {total_spikes}")
    assert total_spikes >= num_cores, \
        f"Expected >= {num_cores} spikes, got {total_spikes}"
    print("  PASSED")
    return True

TESTS = {
    "saturation": test_all_core_saturation,
    "stability": test_long_running_stability,
    "fanout": test_max_fan_out,
    "weights": test_weight_extremes,
    "pool": test_pool_depth_fill,
    "chain": test_cross_core_chain,
}

def main():
    parser = argparse.ArgumentParser(description="SDK Stress Tests")
    parser.add_argument("--test", choices=list(TESTS.keys()),
                        help="Run specific test (default: all)")
    parser.add_argument("--cores", type=int, default=16)
    args = parser.parse_args()

    if args.test:
        tests = {args.test: TESTS[args.test]}
    else:
        tests = TESTS

    passed = 0
    failed = 0
    for name, func in tests.items():
        try:
            func()
            passed += 1
        except (RuntimeError, ValueError, AssertionError, MemoryError) as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\nStress Tests: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL STRESS TESTS PASSED")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

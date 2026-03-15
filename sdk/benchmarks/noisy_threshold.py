"""Noisy Threshold Benchmark
=============================
Demonstrates P14 stochastic noise injection and its effect on neural dynamics.

A population of identical neurons receives the same sub-threshold input.
With noise enabled, some neurons fire stochastically due to threshold fluctuation.

Features demonstrated: P14 noise, statistical analysis, noise_config parameter.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE


def run_trial(noise_config, noise_enable, num_neurons=32, timesteps=100, current=980):
    """Run a trial with given noise configuration."""
    net = nc.Network()
    pop = net.population(num_neurons, params={
        "threshold": 1000, "leak": 3, "refrac": 3,
        "noise_config": noise_config,
    })

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(noise=noise_enable)

    total_spikes = 0
    for _ in range(timesteps):
        sim.inject(pop, current=current)
        result = sim.run(1)
        total_spikes += result.total_spikes

    return total_spikes


def main():
    print("=" * 60)
    print("  Noisy Threshold Benchmark (P14 Stochastic Noise)")
    print("=" * 60)

    num_neurons = 32
    timesteps = 100

    # Test 1: No noise (deterministic)
    print(f"\nSetup: {num_neurons} neurons, threshold=1000, current=980 (sub-threshold)")
    print(f"Running {timesteps} timesteps per trial\n")

    spikes_no_noise = run_trial(noise_config=0, noise_enable=False)
    print(f"1. No noise:           {spikes_no_noise:4d} spikes (deterministic)")

    # Test 2: Small noise
    # noise_config = 0x21: mantissa=1, exponent=2 -> mask = 1 << 2 = 4
    spikes_small = run_trial(noise_config=0x21, noise_enable=True)
    print(f"2. Small noise (0x21): {spikes_small:4d} spikes (mask=4, +/-2)")

    # Test 3: Medium noise
    # noise_config = 0x34: mantissa=4, exponent=3 -> mask = 4 << 3 = 32
    spikes_medium = run_trial(noise_config=0x34, noise_enable=True)
    print(f"3. Medium noise (0x34):{spikes_medium:4d} spikes (mask=32, +/-16)")

    # Test 4: Large noise
    # noise_config = 0x48: mantissa=8, exponent=4 -> mask = 8 << 4 = 128
    spikes_large = run_trial(noise_config=0x48, noise_enable=True)
    print(f"4. Large noise (0x48): {spikes_large:4d} spikes (mask=128, +/-64)")

    # Test 5: Very large noise
    # noise_config = 0x5F: mantissa=15, exponent=5 -> mask = 15 << 5 = 480
    spikes_vlarge = run_trial(noise_config=0x5F, noise_enable=True)
    print(f"5. V.Large noise(0x5F):{spikes_vlarge:4d} spikes (mask=480, +/-240)")

    # Test 6: Noise enabled but config=0 (should be deterministic)
    spikes_zero_cfg = run_trial(noise_config=0, noise_enable=True)
    print(f"6. Noise on, cfg=0:   {spikes_zero_cfg:4d} spikes (should match #1)")

    # Analysis
    print("\n--- Analysis ---")
    print(f"Sub-threshold gap: 1000 - 980 + 3(leak) = 23")
    print(f"Noise must exceed gap for stochastic firing.")
    print(f"Noise escalation: {spikes_no_noise} -> {spikes_small} -> "
          f"{spikes_medium} -> {spikes_large} -> {spikes_vlarge}")

    if spikes_vlarge > spikes_no_noise:
        print("Result: Noise successfully enables stochastic firing!")
    else:
        print("Result: Noise range too small to overcome threshold gap.")

    print("\nDone!")


if __name__ == "__main__":
    main()

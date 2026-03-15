import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import NEURONS_PER_CORE

def run_trial(noise_config, noise_enable, num_neurons=32, timesteps=100, current=980):
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
    print("Noisy Threshold Benchmark (P14 Stochastic Noise)")

    num_neurons = 32
    timesteps = 100

    print(f"\nSetup: {num_neurons} neurons, threshold=1000, current=980 (sub-threshold)")
    print(f"Running {timesteps} timesteps per trial\n")

    spikes_no_noise = run_trial(noise_config=0, noise_enable=False)
    print(f"1. No noise:           {spikes_no_noise:4d} spikes (deterministic)")

    spikes_small = run_trial(noise_config=0x21, noise_enable=True)
    print(f"2. Small noise (0x21): {spikes_small:4d} spikes (mask=4, +/-2)")

    spikes_medium = run_trial(noise_config=0x34, noise_enable=True)
    print(f"3. Medium noise (0x34):{spikes_medium:4d} spikes (mask=32, +/-16)")

    spikes_large = run_trial(noise_config=0x48, noise_enable=True)
    print(f"4. Large noise (0x48): {spikes_large:4d} spikes (mask=128, +/-64)")

    spikes_vlarge = run_trial(noise_config=0x5F, noise_enable=True)
    print(f"5. V.Large noise(0x5F):{spikes_vlarge:4d} spikes (mask=480, +/-240)")

    spikes_zero_cfg = run_trial(noise_config=0, noise_enable=True)
    print(f"6. Noise on, cfg=0:   {spikes_zero_cfg:4d} spikes (should match #1)")

    print("\nAnalysis:")
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

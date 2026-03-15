import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

def build_network(n_neurons, fan_out=4, weight=200, seed=42):
    net = nc.Network()
    pop = net.population(n_neurons, params={"threshold": 500, "leak": 3})
    net.connect(pop, pop, topology="fixed_fan_out", fan_out=fan_out,
                weight=weight, seed=seed)
    return net, pop

def time_cpu(net, pop, timesteps=50, stim_neurons=16, stim_steps=5):
    sim = nc.Simulator()
    sim.deploy(net)

    start = time.perf_counter()
    for t in range(stim_steps):
        sim.inject(pop[:stim_neurons], current=1200)
        sim.run(1)
    result = sim.run(timesteps - stim_steps)
    elapsed = time.perf_counter() - start
    return elapsed, result.total_spikes

def time_gpu(net, pop, timesteps=50, stim_neurons=16, stim_steps=5, device=None):
    sim = nc.GpuSimulator(device=device)
    sim.deploy(net)

    sim.run(1)
    torch.cuda.synchronize(sim.device)
    sim.close()

    sim = nc.GpuSimulator(device=device)
    sim.deploy(net)

    start = time.perf_counter()
    for t in range(stim_steps):
        sim.inject(pop[:stim_neurons], current=1200)
        sim.run(1)
    result = sim.run(timesteps - stim_steps)
    torch.cuda.synchronize(sim.device)
    elapsed = time.perf_counter() - start
    sim.close()
    return elapsed, result.total_spikes

def main():
    if not HAS_CUDA:
        print("CUDA not available. Cannot run GPU benchmark.")
        return

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    vram = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")
    print()

    print("Part 1: CPU vs GPU Wall-Clock (50 timesteps, fan_out=4)")
    print(f"{'Neurons':>8}  {'Synapses':>10}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>8}")

    configs = [
        (64, 4),
        (256, 4),
        (1024, 4),
        (4096, 4),
        (8192, 4),
        (16384, 4),
        (32768, 4),
    ]

    for n_neurons, fan_out in configs:
        try:
            net, pop = build_network(n_neurons, fan_out=fan_out)
            synapses = n_neurons * fan_out

            if n_neurons <= 8192:
                cpu_time, _ = time_cpu(net, pop)
            else:
                cpu_time = float('inf')

            gpu_time, _ = time_gpu(net, pop, device=device)

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            cpu_str = f"{cpu_time:10.4f}" if cpu_time < float('inf') else "       n/a"

            print(f"{n_neurons:>8}  {synapses:>10}  {cpu_str}  {gpu_time:10.4f}  {speedup:7.1f}x")
        except (RuntimeError, MemoryError, ValueError) as e:
            print(f"{n_neurons:>8}  {'FAILED':>10}  {e}")

    print()
    print("Part 2: Denser Networks (50 timesteps, fan_out=8)")
    print(f"{'Neurons':>8}  {'Synapses':>10}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>8}")

    dense_configs = [
        (256, 8),
        (512, 8),
        (1024, 8),
        (4096, 8),
    ]

    for n_neurons, fan_out in dense_configs:
        try:
            net, pop = build_network(n_neurons, fan_out=fan_out)
            synapses = n_neurons * fan_out

            if n_neurons <= 4096:
                cpu_time, _ = time_cpu(net, pop)
            else:
                cpu_time = float('inf')

            gpu_time, _ = time_gpu(net, pop, device=device)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            cpu_str = f"{cpu_time:10.4f}" if cpu_time < float('inf') else "       n/a"

            print(f"{n_neurons:>8}  {synapses:>10}  {cpu_str}  {gpu_time:10.4f}  {speedup:7.1f}x")
        except (RuntimeError, MemoryError, ValueError) as e:
            print(f"{n_neurons:>8}  {'FAILED':>10}  {e}")

    print()
    print("Part 3: GPU-Only Large Scale (100 timesteps)")
    hdr = f"{'Neurons':>8}  {'Fan-out':>8}  {'Synapses':>10}  {'Time (s)':>10}  {'ts/sec':>8}"
    print(hdr)

    large_configs = [
        (16384, 4),
        (32768, 4),
        (65536, 4),
        (131072, 4),
    ]

    for n_neurons, fan_out in large_configs:
        try:
            net, pop = build_network(n_neurons, fan_out=fan_out)
            gpu_time, _ = time_gpu(net, pop, timesteps=100, device=device)
            ts_per_sec = 100 / gpu_time if gpu_time > 0 else float('inf')
            print(f"{n_neurons:>8}  {fan_out:>8}  {n_neurons * fan_out:>10}  {gpu_time:10.4f}  {ts_per_sec:7.0f}")
        except (RuntimeError, MemoryError, ValueError) as e:
            print(f"{n_neurons:>8}  {fan_out:>8}  {n_neurons * fan_out:>10}  FAILED: {e}")

    print()
    print("Benchmark complete.")

if __name__ == "__main__":
    main()

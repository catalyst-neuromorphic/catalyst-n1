import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.compiler import Compiler
from neurocore.constants import NEURONS_PER_CORE

def benchmark_scale(num_neurons, topology="random_sparse", p=0.05, fmt='sparse',
                    cluster_size=4):
    net = nc.Network()
    pop = net.population(num_neurons, params={"threshold": 500, "leak": 3, "refrac": 3})
    net.connect(pop, pop, topology=topology, p=p, weight=200, seed=42, format=fmt)

    t0 = time.perf_counter()
    compiler = Compiler(cluster_size=cluster_size)
    compiled = compiler.compile(net)
    t_compile = time.perf_counter() - t0

    sim = nc.Simulator()
    sim.deploy(compiled)

    stim_count = max(1, num_neurons // 10)
    for i in range(stim_count):
        sim.inject([(0, i)], current=800)

    t0 = time.perf_counter()
    result = sim.run(50)
    t_sim = time.perf_counter() - t0

    return {
        "neurons": num_neurons,
        "cores": compiled.placement.num_cores_used,
        "pool_cmds": len(compiled.prog_pool_cmds),
        "index_cmds": len(compiled.prog_index_cmds),
        "local_routes": len(compiled.prog_route_cmds),
        "global_routes": len(compiled.prog_global_route_cmds),
        "spikes": result.total_spikes,
        "compile_ms": t_compile * 1000,
        "sim_ms": t_sim * 1000,
        "format": fmt,
    }

def main():
    print("Multi-Core Scaling Benchmark (P18 + P20)")

    print("\nSize Scaling (sparse format, cluster_size=4)")
    print(f"{'Neurons':>8} {'Cores':>5} {'Pool':>6} {'Index':>6} "
          f"{'Local':>6} {'Global':>6} {'Spikes':>7} {'Compile':>8} {'Sim':>8}")
    print()

    for n, p_val in [(64, 0.1), (256, 0.05), (512, 0.03), (1024, 0.015), (2048, 0.001)]:
        stats = benchmark_scale(n, topology="random_sparse", p=p_val, fmt='sparse')
        print(f"{stats['neurons']:>8} {stats['cores']:>5} {stats['pool_cmds']:>6} "
              f"{stats['index_cmds']:>6} {stats['local_routes']:>6} "
              f"{stats['global_routes']:>6} {stats['spikes']:>7} "
              f"{stats['compile_ms']:>7.1f}ms {stats['sim_ms']:>7.1f}ms")

    print("\nSynapse Format Comparison (128 neurons, all_to_all)")
    print(f"{'Format':>8} {'Pool':>6} {'Index':>6} {'Spikes':>7} {'Compile':>8}")
    print()

    for fmt in ['sparse', 'dense', 'pop']:
        stats = benchmark_scale(128, topology="all_to_all", p=1.0, fmt=fmt)
        print(f"{stats['format']:>8} {stats['pool_cmds']:>6} {stats['index_cmds']:>6} "
              f"{stats['spikes']:>7} {stats['compile_ms']:>7.1f}ms")

    print("\nCluster Size Impact (4096 neurons, 4 cores)")
    print(f"{'ClusterSz':>9} {'Local':>6} {'Global':>6} {'Total Routes':>12}")
    print()

    for cs in [2, 4, 8]:
        stats = benchmark_scale(4096, topology="random_sparse", p=0.0002,
                                cluster_size=cs)
        total = stats['local_routes'] + stats['global_routes']
        print(f"{cs:>9} {stats['local_routes']:>6} {stats['global_routes']:>6} "
              f"{total:>12}")

    print("\nDone!")

if __name__ == "__main__":
    main()

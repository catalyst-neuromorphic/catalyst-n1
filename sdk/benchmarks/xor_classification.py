import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc

def main():
    print("XOR Classification Benchmark")

    net = nc.Network()

    inp_a = net.population(8, params={"threshold": 100, "leak": 0, "refrac": 2}, label="input_A")
    inp_b = net.population(8, params={"threshold": 100, "leak": 0, "refrac": 2}, label="input_B")

    hidden = net.population(16, params={"threshold": 400, "leak": 5, "refrac": 3}, label="hidden")

    output = net.population(1, params={"threshold": 600, "leak": 3, "refrac": 5}, label="output")

    net.connect(inp_a, hidden, topology="all_to_all", weight=150)
    net.connect(inp_b, hidden, topology="all_to_all", weight=150)
    net.connect(hidden, output, topology="all_to_all", weight=200)

    net.connect(hidden, hidden, topology="random_sparse", p=0.3, weight=-100, seed=42)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(learn=True)

    xor_patterns = [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ]

    print("\nTraining phase (20 epochs)...")
    for epoch in range(20):
        total_spikes = 0
        for a_active, b_active, expected in xor_patterns:
            if a_active:
                sim.inject(inp_a, current=300)
            if b_active:
                sim.inject(inp_b, current=300)
            result = sim.run(10)
            total_spikes += result.total_spikes

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: {total_spikes} total spikes")

    print("\nTest phase:")
    for a_active, b_active, expected in xor_patterns:
        if a_active:
            sim.inject(inp_a, current=300)
        if b_active:
            sim.inject(inp_b, current=300)
        result = sim.run(10)
        out_gid = result.placement.neuron_map[(output.id, 0)]
        out_gid_flat = out_gid[0] * 1024 + out_gid[1]
        out_spikes = len(result.spike_trains.get(out_gid_flat, []))
        label = "1" if expected else "0"
        print(f"  A={int(a_active)}, B={int(b_active)} -> "
              f"Output spikes: {out_spikes} (expected: {label})")

    print(f"\nCompiled: {sim._compiled.placement.num_cores_used} cores, "
          f"{sim._compiled.placement.total_neurons} neurons")
    print("Done!")

if __name__ == "__main__":
    main()

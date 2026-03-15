"""Custom Learning Rule Benchmark
==================================
Demonstrates P19 microcode learning engine with custom learning rules.

Compares default STDP, anti-STDP, and a custom reward-modulated rule
assembled from microcode text mnemonics.

Features demonstrated: P19 microcode ISA, assembler, LearningRule, custom rules.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.microcode import LearningRule


def build_network():
    """Create a simple pre->post network for learning experiments."""
    net = nc.Network()
    pre = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0}, label="pre")
    post = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0}, label="post")
    net.connect(pre, post, topology="all_to_all", weight=500)
    return net, pre, post


def get_final_weight(sim):
    """Extract the weight from the simulator's adjacency table."""
    for targets in sim._adjacency.values():
        for entry in targets:
            return entry[1]
    return None


def run_stdp(rule, rule_name, three_factor=False):
    """Run a learning trial with the given rule."""
    net, pre, post = build_network()
    net.set_learning_rule(rule)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(learn=True, three_factor=three_factor)

    # Generate pre-before-post spike pattern (normally LTP)
    for _ in range(5):
        sim.inject(pre, current=200)
        sim.run(1)  # pre spikes
        sim.run(1)  # post receives input, spikes -> LTP correlation

    if three_factor:
        sim.reward(500)
        sim.run(1)

    final_w = get_final_weight(sim)
    print(f"  {rule_name}: initial=500, final={final_w}")
    return final_w


def main():
    print("=" * 60)
    print("  Custom Learning Rule Benchmark (P19 Microcode)")
    print("=" * 60)

    # 1. Default STDP (weight directly modified)
    print("\n1. Default STDP (pre-before-post = LTP):")
    rule_stdp = LearningRule.stdp()
    w_stdp = run_stdp(rule_stdp, "Default STDP")
    assert w_stdp > 500, "STDP LTP should increase weight"

    # 2. Anti-STDP (inverted: pre-before-post = LTD)
    print("\n2. Anti-STDP (inverted correlation):")
    rule_anti = LearningRule()
    rule_anti.assemble_ltd("""
        SHR R5, R0, 3       ; delta = trace >> 3
        SKIP_Z R5            ; skip if zero
        ADD R2, R2, R5       ; weight += delta (anti-LTD = potentiate)
        STORE_W R2
        HALT
    """)
    rule_anti.assemble_ltp("""
        SHR R5, R0, 3       ; delta = trace >> 3
        SKIP_Z R5            ; skip if zero
        SUB R2, R2, R5       ; weight -= delta (anti-LTP = depress)
        STORE_W R2
        HALT
    """)
    w_anti = run_stdp(rule_anti, "Anti-STDP")
    assert w_anti < 500, "Anti-STDP should decrease weight for pre-before-post"

    # 3. Scaled STDP (2x learning rate via SHL)
    print("\n3. Scaled STDP (2x learning rate):")
    rule_fast = LearningRule()
    rule_fast.assemble_ltd("""
        SHR R5, R0, 3       ; delta = trace >> 3
        SHL R5, R5, 1       ; delta *= 2 (double rate)
        SKIP_Z R5
        SUB R2, R2, R5
        STORE_W R2
        HALT
    """)
    rule_fast.assemble_ltp("""
        SHR R5, R0, 3       ; delta = trace >> 3
        SHL R5, R5, 1       ; delta *= 2
        SKIP_Z R5
        ADD R2, R2, R5
        STORE_W R2
        HALT
    """)
    w_fast = run_stdp(rule_fast, "2x STDP")
    assert w_fast > w_stdp, f"2x STDP ({w_fast}) should be > default ({w_stdp})"

    # 4. 3-factor eligibility learning (default program)
    print("\n4. 3-factor eligibility + reward:")
    rule_3f = LearningRule.three_factor()
    w_3f = run_stdp(rule_3f, "3-factor STDP", three_factor=True)
    print(f"     (Reward applied: weight change reflects eligibility * reward)")

    # 5. Custom capped rule (weight bounded to [400, 600])
    print("\n5. Capped STDP (weight bounded [400, 600]):")
    rule_capped = LearningRule()
    rule_capped.assemble_ltp("""
        SHR R5, R0, 3       ; delta = trace >> 3
        SKIP_Z R5
        ADD R2, R2, R5       ; weight += delta
        LOADI R4, 600        ; max weight
        MIN R2, R2, R4       ; clamp to max
        STORE_W R2
        HALT
    """)
    rule_capped.assemble_ltd("""
        SHR R5, R0, 3
        SKIP_Z R5
        SUB R2, R2, R5       ; weight -= delta
        LOADI R4, 400        ; min weight
        MAX R2, R2, R4       ; clamp to min
        STORE_W R2
        HALT
    """)
    w_capped = run_stdp(rule_capped, "Capped STDP")
    assert 400 <= w_capped <= 600, f"Capped weight should be in [400,600], got {w_capped}"

    # Summary
    print("\n--- Summary ---")
    print(f"Default STDP:  {w_stdp:>6d} (LTP: weight increased)")
    print(f"Anti-STDP:     {w_anti:>6d} (inverted: weight decreased)")
    print(f"2x STDP:       {w_fast:>6d} (double learning rate)")
    print(f"3-Factor:      {w_3f:>6d} (eligibility + reward)")
    print(f"Capped [400,600]: {w_capped:>4d} (bounded)")
    print("\nAll custom learning rules verified!")
    print("Done!")


if __name__ == "__main__":
    main()

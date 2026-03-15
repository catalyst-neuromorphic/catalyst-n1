import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.microcode import LearningRule

def build_network():
    net = nc.Network()
    pre = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0}, label="pre")
    post = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0}, label="post")
    net.connect(pre, post, topology="all_to_all", weight=500)
    return net, pre, post

def get_final_weight(sim):
    for targets in sim._adjacency.values():
        for entry in targets:
            return entry[1]
    return None

def run_stdp(rule, rule_name, three_factor=False):
    net, pre, post = build_network()
    net.set_learning_rule(rule)

    sim = nc.Simulator()
    sim.deploy(net)
    sim.set_learning(learn=True, three_factor=three_factor)

    for _ in range(5):
        sim.inject(pre, current=200)
        sim.run(1)
        sim.run(1)

    if three_factor:
        sim.reward(500)
        sim.run(1)

    final_w = get_final_weight(sim)
    print(f"  {rule_name}: initial=500, final={final_w}")
    return final_w

def main():
    print("Custom Learning Rule Benchmark (P19 Microcode)")

    print("\nDefault STDP (pre-before-post = LTP):")
    rule_stdp = LearningRule.stdp()
    w_stdp = run_stdp(rule_stdp, "Default STDP")
    assert w_stdp > 500, "STDP LTP should increase weight"

    print("\nAnti-STDP (inverted correlation):")
    rule_anti = LearningRule()
    rule_anti.assemble_ltd("""
        SHR R5, R0, 3
        SKIP_Z R5
        ADD R2, R2, R5
        STORE_W R2
        HALT
    """)
    rule_anti.assemble_ltp("""
        SHR R5, R0, 3
        SKIP_Z R5
        SUB R2, R2, R5
        STORE_W R2
        HALT
    """)
    w_anti = run_stdp(rule_anti, "Anti-STDP")
    assert w_anti < 500, "Anti-STDP should decrease weight for pre-before-post"

    print("\nScaled STDP (2x learning rate):")
    rule_fast = LearningRule()
    rule_fast.assemble_ltd("""
        SHR R5, R0, 3
        SHL R5, R5, 1
        SKIP_Z R5
        SUB R2, R2, R5
        STORE_W R2
        HALT
    """)
    rule_fast.assemble_ltp("""
        SHR R5, R0, 3
        SHL R5, R5, 1
        SKIP_Z R5
        ADD R2, R2, R5
        STORE_W R2
        HALT
    """)
    w_fast = run_stdp(rule_fast, "2x STDP")
    assert w_fast > w_stdp, f"2x STDP ({w_fast}) should be > default ({w_stdp})"

    print("\n3-factor eligibility + reward:")
    rule_3f = LearningRule.three_factor()
    w_3f = run_stdp(rule_3f, "3-factor STDP", three_factor=True)

    print("\nCapped STDP (weight bounded [400, 600]):")
    rule_capped = LearningRule()
    rule_capped.assemble_ltp("""
        SHR R5, R0, 3
        SKIP_Z R5
        ADD R2, R2, R5
        LOADI R4, 600
        MIN R2, R2, R4
        STORE_W R2
        HALT
    """)
    rule_capped.assemble_ltd("""
        SHR R5, R0, 3
        SKIP_Z R5
        SUB R2, R2, R5
        LOADI R4, 400
        MAX R2, R2, R4
        STORE_W R2
        HALT
    """)
    w_capped = run_stdp(rule_capped, "Capped STDP")
    assert 400 <= w_capped <= 600, f"Capped weight should be in [400,600], got {w_capped}"

    print(f"\nDefault STDP:     {w_stdp}")
    print(f"Anti-STDP:        {w_anti}")
    print(f"2x STDP:          {w_fast}")
    print(f"3-Factor:         {w_3f}")
    print(f"Capped [400,600]: {w_capped}")

if __name__ == "__main__":
    main()

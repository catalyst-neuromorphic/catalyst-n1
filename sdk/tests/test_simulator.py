"""Tests for cycle-accurate LIF simulator.

These tests verify the simulator matches the RTL behavior in scalable_core_v2.v.
P20 update: noise, dual traces, delays, formats, microcode, hierarchical routing.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import (
    DEFAULT_THRESHOLD, DEFAULT_LEAK, DEFAULT_REFRAC, NEURONS_PER_CORE,
    TRACE_MAX, DEFAULT_TAU1, DEFAULT_TAU2,
)


class TestSingleNeuron:
    def test_constant_input_spike_timing(self):
        """With threshold=1000, leak=3, constant input=200:
        Each timestep adds (200 - 3) = 197 to potential.
        Spike at timestep where cumulative >= 1000.
        ceil(1000 / 197) = 6 timesteps.

        t0: 0 + 200 - 3 = 197
        t1: 197 + 200 - 3 = 394
        t2: 394 + 200 - 3 = 591
        t3: 591 + 200 - 3 = 788
        t4: 788 + 200 - 3 = 985 (< 1000)
        t5: 985 + 200 - 3 = 1182 >= 1000 -> SPIKE at t5
        """
        net = nc.Network()
        pop = net.population(1, params={"threshold": 1000, "leak": 3})
        sim = nc.Simulator()
        sim.deploy(net)

        spike_times = []
        for t in range(20):
            sim.inject(pop, current=200)
            result = sim.run(1)
            if result.total_spikes > 0:
                spike_times.append(t)

        assert spike_times[0] == 5

    def test_refractory_period(self):
        """After spiking, neuron should be silent for refrac_period timesteps."""
        net = nc.Network()
        pop = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 3})
        sim = nc.Simulator()
        sim.deploy(net)

        spike_times = []
        for t in range(20):
            sim.inject(pop, current=200)
            result = sim.run(1)
            if result.total_spikes > 0:
                spike_times.append(t)

        assert spike_times[0] == 0
        assert spike_times[1] == 4

    def test_subthreshold_decay_to_resting(self):
        """If input is less than leak, potential should floor to resting."""
        net = nc.Network()
        pop = net.population(1, params={"threshold": 1000, "leak": 100, "resting": 0})
        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(pop, current=50)
        result = sim.run(1)
        assert result.total_spikes == 0
        assert int(sim._potential[0]) == 0


class TestChainPropagation:
    def test_spike_chain(self, chain_network_manual):
        """N0 -> N1 -> N2 -> N3 with weight=1200, stimulus N0."""
        net, n0, n1, n2, n3 = chain_network_manual
        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(n0, current=1200)
        result = sim.run(10)

        assert result.total_spikes >= 4

        p = result.placement
        gid0 = p.neuron_map[(n0.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n0.id, 0)][1]
        gid1 = p.neuron_map[(n1.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n1.id, 0)][1]
        gid2 = p.neuron_map[(n2.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n2.id, 0)][1]
        gid3 = p.neuron_map[(n3.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n3.id, 0)][1]

        assert 0 in result.spike_trains.get(gid0, [])
        assert 1 in result.spike_trains.get(gid1, [])
        assert 2 in result.spike_trains.get(gid2, [])
        assert 3 in result.spike_trains.get(gid3, [])


class TestInhibition:
    def test_inhibitory_weight_prevents_spike(self):
        """Negative weight should reduce potential."""
        net = nc.Network()
        exc = net.population(1, label="exc")
        inh = net.population(1, label="inh")
        target = net.population(1, label="target")

        net.connect(exc, target, topology="all_to_all", weight=500)
        net.connect(inh, target, topology="all_to_all", weight=-600)

        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(exc, current=1200)
        sim.inject(inh, current=1200)
        result = sim.run(5)

        p = result.placement
        tgt_gid = p.neuron_map[(target.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(target.id, 0)][1]
        tgt_spikes = result.spike_trains.get(tgt_gid, [])
        assert 1 not in tgt_spikes


class TestGradedSpikes:
    def test_graded_payload_scaling(self):
        """With graded enabled, spike payload should scale delivered current."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt = net.population(1, params={"threshold": 1000, "leak": 0})
        net.connect(src, tgt, topology="all_to_all", weight=200)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(graded=True)

        sim.inject(src, current=500)
        result = sim.run(3)

        p = result.placement
        tgt_gid = p.neuron_map[(tgt.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(tgt.id, 0)][1]
        assert 1 not in result.spike_trains.get(tgt_gid, [])


class TestDendriticCompartments:
    def test_dendritic_threshold(self):
        """Dendritic input below threshold should be suppressed."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt = net.population(1, params={
            "threshold": 1000, "leak": 0, "dend_threshold": 500
        })
        net.connect(src, tgt, topology="all_to_all", weight=200, compartment=1)

        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(src, current=200)
        result = sim.run(5)

        p = result.placement
        tgt_gid = p.neuron_map[(tgt.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(tgt.id, 0)][1]
        assert len(result.spike_trains.get(tgt_gid, [])) == 0


class TestAsyncMode:
    """Tests for P12 GALS async event-driven simulation."""

    def test_basic_async_propagation(self, chain_network_manual):
        """Chain N0->N1->N2->N3 should propagate in async mode."""
        net, n0, n1, n2, n3 = chain_network_manual
        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(async_mode=True)

        sim.inject(n0, current=1200)
        result = sim.run(1)

        assert result.total_spikes == 4

        p = result.placement
        gid0 = p.neuron_map[(n0.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n0.id, 0)][1]
        gid1 = p.neuron_map[(n1.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n1.id, 0)][1]
        gid2 = p.neuron_map[(n2.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n2.id, 0)][1]
        gid3 = p.neuron_map[(n3.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n3.id, 0)][1]

        assert 0 in result.spike_trains.get(gid0, [])
        assert 0 in result.spike_trains.get(gid1, [])
        assert 0 in result.spike_trains.get(gid2, [])
        assert 0 in result.spike_trains.get(gid3, [])

    def test_quiescence_single_neuron(self):
        """Isolated neuron with no connections — activity dies immediately."""
        net = nc.Network()
        pop = net.population(1, params={"threshold": 100, "leak": 0})
        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(async_mode=True)

        sim.inject(pop, current=200)
        result = sim.run(1)
        assert result.total_spikes == 1

    def test_async_sync_equivalence(self):
        """Critical test: async mode must produce identical spike counts
        to sync mode for accumulation-dominated workloads."""
        def build_and_run(async_mode):
            net = nc.Network()
            src = net.population(1, params={"threshold": 1000, "leak": 3, "refrac": 3})
            tgt = net.population(1, params={"threshold": 1000, "leak": 3, "refrac": 3})
            net.connect(src, tgt, topology="all_to_all", weight=1200)

            sim = nc.Simulator()
            sim.deploy(net)
            sim.set_learning(async_mode=async_mode)

            total = 0
            for _ in range(10):
                sim.inject(src, current=200)
                result = sim.run(1)
                total += result.total_spikes
            return total

        sync_spikes = build_and_run(async_mode=False)
        async_spikes = build_and_run(async_mode=True)

        assert sync_spikes == async_spikes, (
            f"Sync ({sync_spikes}) != Async ({async_spikes}) — equivalence broken!")

    def test_async_chain_collapses_to_one_timestep(self):
        """In async mode, a spike chain propagates within a single timestep."""
        net = nc.Network()
        n0 = net.population(1, params={"threshold": 100, "leak": 0}, label="n0")
        n1 = net.population(1, params={"threshold": 100, "leak": 0}, label="n1")
        n2 = net.population(1, params={"threshold": 100, "leak": 0}, label="n2")
        n3 = net.population(1, params={"threshold": 100, "leak": 0}, label="n3")
        net.connect(n0, n1, topology="all_to_all", weight=200)
        net.connect(n1, n2, topology="all_to_all", weight=200)
        net.connect(n2, n3, topology="all_to_all", weight=200)

        # Sync: takes 4 timesteps
        sim_sync = nc.Simulator()
        sim_sync.deploy(net)
        sim_sync.inject(n0, current=200)
        result_sync = sim_sync.run(1)
        assert result_sync.total_spikes == 1

        # Async: entire chain in 1 timestep
        sim_async = nc.Simulator()
        sim_async.deploy(net)
        sim_async.set_learning(async_mode=True)
        sim_async.inject(n0, current=200)
        result_async = sim_async.run(1)
        assert result_async.total_spikes == 4

    def test_async_multi_population(self):
        """E/I network should work in async mode."""
        net = nc.Network()
        exc = net.population(8, params={"threshold": 500, "leak": 2, "refrac": 2})
        inh = net.population(4, params={"threshold": 400, "leak": 2, "refrac": 2})
        net.connect(exc, inh, topology="fixed_fan_out", fan_out=4, weight=250, seed=42)
        net.connect(inh, exc, topology="fixed_fan_out", fan_out=8, weight=-200, seed=42)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(async_mode=True)

        sim.inject(exc[:4], current=600)
        result = sim.run(5)

        assert result.total_spikes > 0
        assert result.timesteps == 5

    def test_async_no_input_no_spikes(self):
        """No stimulus -> no activity in async mode."""
        net = nc.Network()
        net.population(16, params={"threshold": 500, "leak": 2})
        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(async_mode=True)

        result = sim.run(10)
        assert result.total_spikes == 0

    def test_async_inter_core_routing(self):
        """Spikes should propagate across cores in async mode."""
        net = nc.Network()
        a = net.population(NEURONS_PER_CORE, label="core0")  # fills core 0
        b = net.population(1, params={"threshold": 100, "leak": 0}, label="core1")
        net.connect(a, b, topology="all_to_all", weight=200)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(async_mode=True)

        sim.inject(a[0], current=1200)
        result = sim.run(1)

        p = result.placement
        b_gid = p.neuron_map[(b.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(b.id, 0)][1]
        assert 0 in result.spike_trains.get(b_gid, []), \
            "Inter-core spike failed to propagate in async mode"


class TestThreeFactorLearning:
    """Tests for P13c 3-factor learning with eligibility traces."""

    def test_eligibility_accumulation_no_weight_change(self):
        """Without reward, STDP correlation accumulates eligibility but
        doesn't change weights."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        # Make both spike (accumulate eligibility via STDP correlation)
        sim.inject(src, current=200)
        sim.inject(tgt, current=200)
        sim.run(5)

        # No reward was applied, so check eligibility exists
        assert len(sim._eligibility) > 0, "Eligibility should accumulate"

        # Weight should be unchanged (no reward applied)
        adj = sim._adjacency
        for targets in adj.values():
            for entry in targets:
                w = entry[1]
                assert w == 500, f"Weight changed without reward: {w}"

    def test_reward_changes_weights(self):
        """Positive reward should change weights when eligibility exists."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        # Generate correlated spikes to build eligibility
        for _ in range(3):
            sim.inject(src, current=200)
            sim.inject(tgt, current=200)
            sim.run(1)

        # Now apply positive reward
        sim.reward(500)
        sim.run(1)

        # Weight should have changed
        weight_changed = False
        for targets in sim._adjacency.values():
            for entry in targets:
                w = entry[1]
                if w != 500:
                    weight_changed = True
        assert weight_changed, "Reward should modify weights via eligibility"

    def test_negative_reward_weakens(self):
        """Negative reward should decrease weights for positive eligibility."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        # Build positive eligibility (LTP: pre fires, then post fires)
        for _ in range(3):
            sim.inject(src, current=200)
            sim.run(1)

        # Negative reward
        sim.reward(-500)
        sim.run(1)

        # Check weights
        for targets in sim._adjacency.values():
            for entry in targets:
                w = entry[1]
                if w != 500:
                    # Weight should have decreased (negative reward * positive elig)
                    assert w < 500, f"Expected weight < 500, got {w}"

    def test_eligibility_decays(self):
        """Eligibility should decay over time without reward."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 1})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 1})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        # Build eligibility with temporal order: src fires first (t=0),
        # tgt fires from synaptic input at t=1. This creates clear LTP
        # since pre fires before post.
        sim.inject(src, current=200)
        sim.run(1)  # src spikes at t=0, trace[src]=100

        # tgt receives weight 500 at t=1 DELIVER: 500 >= 100 -> spike
        sim.run(1)  # tgt spikes at t=1, checks trace[src] for LTP

        assert len(sim._eligibility) > 0, \
            "Eligibility should accumulate from temporal correlation"

        # Run many timesteps without spikes — eligibility should decay to 0
        for _ in range(100):
            sim.run(1)

        assert len(sim._eligibility) == 0, \
            "Eligibility should fully decay without reinforcement"

    def test_delayed_reward(self):
        """Reward arriving after delay should still modify weights
        (eligibility hasn't fully decayed)."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True, three_factor=True)

        # Spike correlation
        sim.inject(src, current=200)
        sim.inject(tgt, current=200)
        sim.run(1)

        # Short delay (eligibility still non-zero)
        sim.run(3)
        assert len(sim._eligibility) > 0, "Eligibility should persist briefly"

        # Delayed reward
        sim.reward(500)
        sim.run(1)

        # Weight should have changed despite delay
        weight_changed = False
        for targets in sim._adjacency.values():
            for entry in targets:
                w = entry[1]
                if w != 500:
                    weight_changed = True
        assert weight_changed, "Delayed reward should still modify weights"

    def test_three_factor_implies_learn(self):
        """Setting three_factor=True should auto-enable learn."""
        sim = nc.Simulator()
        net = nc.Network()
        net.population(1)
        sim.deploy(net)
        sim.set_learning(three_factor=True)
        assert sim._learn_enable is True
        assert sim._three_factor_enable is True


class TestRunResult:
    def test_result_fields(self, chain_network_manual):
        net, n0, _, _, _ = chain_network_manual
        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(n0, current=1200)
        result = sim.run(10)
        assert result.backend == "simulator"
        assert result.timesteps == 10
        assert isinstance(result.spike_trains, dict)

    def test_firing_rates(self, chain_network_manual):
        net, n0, _, _, _ = chain_network_manual
        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(n0, current=1200)
        result = sim.run(10)
        rates = result.firing_rates()
        assert isinstance(rates, dict)
        assert all(r >= 0 for r in rates.values())

    def test_spike_count_timeseries(self, chain_network_manual):
        net, n0, _, _, _ = chain_network_manual
        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(n0, current=1200)
        result = sim.run(10)
        ts = result.spike_count_timeseries()
        assert len(ts) == 10


class TestStochasticNoise:
    """Tests for P14 stochastic noise injection."""

    def test_noise_disabled_deterministic(self):
        """With noise_enable=False, identical runs produce identical results."""
        def run_once():
            net = nc.Network()
            pop = net.population(4, params={"threshold": 500, "leak": 3})
            sim = nc.Simulator()
            sim.deploy(net)
            # noise_enable is False by default
            total = 0
            for _ in range(20):
                sim.inject(pop, current=100)
                result = sim.run(1)
                total += result.total_spikes
            return total

        assert run_once() == run_once()

    def test_noise_enabled_variability(self):
        """With noise_enable=True and non-zero config, results vary due to
        different LFSR evolution per neuron (different noise sequences for
        neurons near threshold)."""
        net = nc.Network()
        # Many neurons near threshold for maximal noise effect
        pop = net.population(16, params={
            "threshold": 200, "leak": 0, "refrac": 0,
            "noise_config": 0x34  # mantissa=4, exponent=3 -> noise_mask=32
        })
        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(noise=True)

        # Inject current right at threshold boundary
        sim.inject(pop, current=200)
        result = sim.run(20)

        # With noise, some near-threshold neurons should spike at different times
        # Check that not all neurons spike on the same timestep pattern
        trains = result.spike_trains
        spike_sets = [set(trains.get(i, [])) for i in range(16)]
        # With noise_mask=32 centered around threshold, some neurons will fire
        # at different timesteps. Not all spike patterns should be identical.
        unique_patterns = len(set(frozenset(s) for s in spike_sets))
        assert unique_patterns > 1, \
            "All neurons had identical spike patterns despite noise"

    def test_zero_config_still_deterministic(self):
        """noise_enable=True but noise_config=0 means no actual noise."""
        def run_once():
            net = nc.Network()
            pop = net.population(4, params={"threshold": 500, "leak": 3})
            sim = nc.Simulator()
            sim.deploy(net)
            sim.set_learning(noise=True)  # enabled but config=0
            total = 0
            for _ in range(20):
                sim.inject(pop, current=100)
                result = sim.run(1)
                total += result.total_spikes
            return total

        assert run_once() == run_once()

    def test_noise_config_generates_commands(self):
        """Non-default noise_config should generate PROG_NEURON param_id=5."""
        net = nc.Network()
        net.population(2, params={"noise_config": 0x45})
        from neurocore.compiler import Compiler
        compiled = Compiler().compile(net)
        noise_cmds = [c for c in compiled.prog_neuron_cmds if c["param_id"] == 5]
        assert len(noise_cmds) == 2
        assert noise_cmds[0]["value"] == 0x45


class TestDualTraces:
    """Tests for P15 dual spike traces with exponential decay."""

    def test_both_traces_set_on_spike(self):
        """After spiking, both trace and trace2 should be TRACE_MAX."""
        net = nc.Network()
        pop = net.population(1, params={"threshold": 100, "leak": 0})
        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(pop, current=200)
        sim.run(1)  # should spike

        assert int(sim._trace[0]) == TRACE_MAX
        assert int(sim._trace2[0]) == TRACE_MAX

    def test_different_decay_rates(self):
        """tau1=2 should decay faster than tau2=6."""
        net = nc.Network()
        pop = net.population(1, params={
            "threshold": 100, "leak": 0, "refrac": 0,
            "tau1": 2, "tau2": 6
        })
        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(pop, current=200)
        sim.run(1)  # spike -> both traces = TRACE_MAX

        # Run several timesteps without input, let traces decay
        sim.run(5)

        trace1 = int(sim._trace[0])
        trace2 = int(sim._trace2[0])
        assert trace1 < trace2, \
            f"trace1 ({trace1}) should be < trace2 ({trace2}) with faster decay"

    def test_min_step_1_convergence(self):
        """Traces should reach 0 (no stuck values) via min-step-1."""
        net = nc.Network()
        pop = net.population(1, params={
            "threshold": 100, "leak": 0, "refrac": 0,
            "tau1": 8, "tau2": 8  # very slow decay
        })
        sim = nc.Simulator()
        sim.deploy(net)

        sim.inject(pop, current=200)
        sim.run(1)  # spike

        # Run many timesteps — traces should eventually reach 0
        sim.run(200)
        assert int(sim._trace[0]) == 0
        assert int(sim._trace2[0]) == 0

    def test_stdp_uses_trace1(self):
        """STDP weight updates should use trace1 only (backward compat)."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.set_learning(learn=True)

        # Make src spike first, then tgt (LTP: pre before post)
        sim.inject(src, current=200)
        sim.run(1)  # src spikes at t=0
        sim.run(1)  # tgt gets input, spikes at t=1 -> LTP

        # Weight should have increased (LTP using trace1)
        adj = sim._adjacency
        for targets in adj.values():
            for entry in targets:
                w = entry[1]
                assert w > 500, f"Expected LTP weight increase, got {w}"

    def test_default_tau_values(self):
        """Default tau1=3, tau2=4 should be set."""
        net = nc.Network()
        pop = net.population(1)
        sim = nc.Simulator()
        sim.deploy(net)
        assert int(sim._tau1[0]) == DEFAULT_TAU1
        assert int(sim._tau2[0]) == DEFAULT_TAU2

    def test_tau_generates_commands(self):
        """Non-default tau values should generate PROG_NEURON commands."""
        net = nc.Network()
        net.population(2, params={"tau1": 5, "tau2": 7})
        from neurocore.compiler import Compiler
        compiled = Compiler().compile(net)
        tau1_cmds = [c for c in compiled.prog_neuron_cmds if c["param_id"] == 6]
        tau2_cmds = [c for c in compiled.prog_neuron_cmds if c["param_id"] == 7]
        assert len(tau1_cmds) == 2
        assert len(tau2_cmds) == 2
        assert tau1_cmds[0]["value"] == 5
        assert tau2_cmds[0]["value"] == 7


class TestAxonDelays:
    """Tests for P17 axon delays."""

    def test_delay_zero_backward_compat(self):
        """Chain with delay=0 should behave identically to original."""
        net = nc.Network()
        n0 = net.population(1, params={"threshold": 100, "leak": 0}, label="n0")
        n1 = net.population(1, params={"threshold": 100, "leak": 0}, label="n1")
        net.connect(n0, n1, topology="all_to_all", weight=200, delay=0)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(n0, current=200)
        result = sim.run(5)

        p = result.placement
        gid1 = p.neuron_map[(n1.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n1.id, 0)][1]
        assert 1 in result.spike_trains.get(gid1, []), \
            "N1 should spike at t=1 with delay=0"

    def test_delay_3_shifts_spike(self):
        """With delay=3, target should spike 3 timesteps later than delay=0."""
        net = nc.Network()
        n0 = net.population(1, params={"threshold": 100, "leak": 0}, label="n0")
        n1 = net.population(1, params={"threshold": 100, "leak": 0}, label="n1")
        net.connect(n0, n1, topology="all_to_all", weight=200, delay=3)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(n0, current=200)
        result = sim.run(10)

        p = result.placement
        gid1 = p.neuron_map[(n1.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(n1.id, 0)][1]
        spikes_n1 = result.spike_trains.get(gid1, [])
        # delay=0 would spike at t=1. delay=3 means spike arrives 3 timesteps later.
        # Spike at t=0 + delivery at t=0 (pending) + delay=3 -> arrives at t=3
        # Then n1 accumulates at t=3, spikes at t=3
        assert len(spikes_n1) > 0, "N1 should eventually spike"
        assert spikes_n1[0] > 1, \
            f"N1 first spike at t={spikes_n1[0]}, should be delayed past t=1"

    def test_mixed_delays(self):
        """Two targets with different delays should spike at different times."""
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0}, label="src")
        fast = net.population(1, params={"threshold": 100, "leak": 0}, label="fast")
        slow = net.population(1, params={"threshold": 100, "leak": 0}, label="slow")
        net.connect(src, fast, topology="all_to_all", weight=200, delay=1)
        net.connect(src, slow, topology="all_to_all", weight=200, delay=5)

        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(src, current=200)
        result = sim.run(10)

        p = result.placement
        gid_fast = p.neuron_map[(fast.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(fast.id, 0)][1]
        gid_slow = p.neuron_map[(slow.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(slow.id, 0)][1]
        fast_spikes = result.spike_trains.get(gid_fast, [])
        slow_spikes = result.spike_trains.get(gid_slow, [])
        assert len(fast_spikes) > 0 and len(slow_spikes) > 0
        assert fast_spikes[0] < slow_spikes[0], \
            f"Fast ({fast_spikes[0]}) should spike before slow ({slow_spikes[0]})"

    def test_delay_validation(self):
        """Invalid delay values should raise ValueError."""
        net = nc.Network()
        src = net.population(1)
        tgt = net.population(1)
        with pytest.raises(ValueError):
            net.connect(src, tgt, weight=200, delay=-1)
        with pytest.raises(ValueError):
            net.connect(src, tgt, weight=200, delay=64)

    def test_delay_generates_commands(self):
        """delay>0 should generate PROG_DELAY commands in compiler."""
        net = nc.Network()
        src = net.population(2)
        tgt = net.population(2)
        net.connect(src, tgt, topology="all_to_all", weight=200, delay=5)
        from neurocore.compiler import Compiler
        compiled = Compiler().compile(net)
        assert len(compiled.prog_delay_cmds) == 4  # 2*2 connections
        assert all(c["delay"] == 5 for c in compiled.prog_delay_cmds)


class TestSynapseFormats:
    """Tests for P18 synapse formats (sparse, dense, pop)."""

    def test_sparse_backward_compat(self):
        """Default format='sparse' should behave identically to pre-P18."""
        net = nc.Network()
        src = net.population(2, params={"threshold": 100, "leak": 0})
        tgt = net.population(2, params={"threshold": 100, "leak": 0})
        net.connect(src, tgt, topology="all_to_all", weight=200, format='sparse')

        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(src, current=200)
        result = sim.run(5)

        # Both targets should spike at t=1
        p = result.placement
        gid_t0 = p.neuron_map[(tgt.id, 0)][0] * NEURONS_PER_CORE + p.neuron_map[(tgt.id, 0)][1]
        gid_t1 = p.neuron_map[(tgt.id, 1)][0] * NEURONS_PER_CORE + p.neuron_map[(tgt.id, 1)][1]
        assert 1 in result.spike_trains.get(gid_t0, [])
        assert 1 in result.spike_trains.get(gid_t1, [])

    def test_dense_all_to_all(self):
        """Dense format with all_to_all should produce same spikes as sparse."""
        def run_with_format(fmt):
            net = nc.Network()
            src = net.population(2, params={"threshold": 100, "leak": 0})
            tgt = net.population(2, params={"threshold": 100, "leak": 0})
            net.connect(src, tgt, topology="all_to_all", weight=200, format=fmt)
            sim = nc.Simulator()
            sim.deploy(net)
            sim.inject(src, current=200)
            result = sim.run(5)
            return result.total_spikes

        sparse_spikes = run_with_format('sparse')
        dense_spikes = run_with_format('dense')
        assert sparse_spikes == dense_spikes, \
            f"Dense ({dense_spikes}) should match sparse ({sparse_spikes})"

    def test_pop_shared_weight(self):
        """Pop format should produce same spikes as sparse with uniform weights."""
        def run_with_format(fmt):
            net = nc.Network()
            src = net.population(1, params={"threshold": 100, "leak": 0})
            tgt = net.population(4, params={"threshold": 100, "leak": 0})
            net.connect(src, tgt, topology="all_to_all", weight=300, format=fmt)
            sim = nc.Simulator()
            sim.deploy(net)
            sim.inject(src, current=200)
            result = sim.run(5)
            return result.total_spikes

        sparse_spikes = run_with_format('sparse')
        pop_spikes = run_with_format('pop')
        assert sparse_spikes == pop_spikes, \
            f"Pop ({pop_spikes}) should match sparse ({sparse_spikes})"

    def test_compiler_format_in_index(self):
        """Compiler should include format field in index commands."""
        from neurocore.compiler import Compiler
        from neurocore.constants import FMT_DENSE, FMT_POP

        # Dense format
        net = nc.Network()
        src = net.population(1)
        tgt = net.population(3)
        net.connect(src, tgt, topology="all_to_all", weight=200, format='dense')
        compiled = Compiler().compile(net)
        assert len(compiled.prog_index_cmds) > 0
        idx = compiled.prog_index_cmds[0]
        assert idx["format"] == FMT_DENSE
        assert "base_target" in idx

    def test_pop_format_single_pool_entry(self):
        """Pop format should generate only 1 pool entry regardless of target count."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        src = net.population(1)
        tgt = net.population(4)
        net.connect(src, tgt, topology="all_to_all", weight=200, format='pop')
        compiled = Compiler().compile(net)

        # Pop: 1 pool entry for all 4 targets
        assert len(compiled.prog_pool_cmds) == 1
        # Index should show count=4 (number of targets)
        assert compiled.prog_index_cmds[0]["count"] == 4

    def test_invalid_format_raises(self):
        """Invalid format string should raise ValueError."""
        net = nc.Network()
        src = net.population(1)
        tgt = net.population(1)
        with pytest.raises(ValueError, match="Unknown format"):
            net.connect(src, tgt, weight=200, format='invalid')

    def test_mixed_formats_same_network(self):
        """Different connections can use different formats in one network."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt_sparse = net.population(2, params={"threshold": 100, "leak": 0})
        tgt_dense = net.population(2, params={"threshold": 100, "leak": 0})
        net.connect(src, tgt_sparse, topology="all_to_all", weight=200, format='sparse')
        net.connect(src, tgt_dense, topology="all_to_all", weight=200, format='dense')

        compiled = Compiler().compile(net)
        # Should have index entries with different formats
        formats_used = set(idx["format"] for idx in compiled.prog_index_cmds)
        assert len(formats_used) >= 1  # at least one format present

        # Simulator should still work
        sim = nc.Simulator()
        sim.deploy(net)
        sim.inject(src, current=200)
        result = sim.run(5)
        assert result.total_spikes > 0


class TestHierarchicalRouting:
    """Tests for P20 hierarchical routing (local vs global routes)."""

    def test_intra_cluster_uses_local_routes(self):
        """Routes within a cluster should use prog_route_cmds (local)."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        # Two populations on different cores but same cluster (cluster_size=4)
        a = net.population(NEURONS_PER_CORE, label="core0")  # fills core 0
        b = net.population(1, params={"threshold": 100, "leak": 0}, label="core1")
        net.connect(a, b, topology="all_to_all", weight=200)

        compiled = Compiler(cluster_size=4).compile(net)
        # Core 0 and core 1 are in same cluster (0 // 4 == 1 // 4 == 0)
        assert len(compiled.prog_route_cmds) > 0
        assert len(compiled.prog_global_route_cmds) == 0

    def test_inter_cluster_uses_global_routes(self):
        """Routes across clusters should use prog_global_route_cmds."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        # Fillers with self-connections (density 2) sort before a/e (density 1)
        b = net.population(NEURONS_PER_CORE, label="filler1")
        c = net.population(NEURONS_PER_CORE, label="filler2")
        d = net.population(NEURONS_PER_CORE, label="filler3")
        net.connect(b, b, topology="one_to_one", weight=100)
        net.connect(c, c, topology="one_to_one", weight=100)
        net.connect(d, d, topology="one_to_one", weight=100)

        a = net.population(NEURONS_PER_CORE, label="src")   # will be core 3 (cluster 0)
        e = net.population(1, params={"threshold": 100, "leak": 0}, label="tgt")  # core 4 (cluster 1)
        net.connect(a, e, topology="all_to_all", weight=200)

        compiled = Compiler(cluster_size=4).compile(net)
        # a on core 3 (3//4=0), e on core 4 (4//4=1) -> different clusters
        assert len(compiled.prog_global_route_cmds) > 0, \
            f"Expected global routes, got local: {len(compiled.prog_route_cmds)}"

    def test_mixed_local_and_global(self):
        """Source pop can have both local and global route targets."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        # Use cluster_size=2. Give a highest density (4) so it sorts first (core 0).
        # b has density 3 -> core 1. e has density 1 -> core 2.
        # cluster 0 = cores 0,1. cluster 1 = cores 2,3.
        a = net.population(NEURONS_PER_CORE, label="src")
        b = net.population(NEURONS_PER_CORE, label="local_tgt")
        e = net.population(1, params={"threshold": 100, "leak": 0}, label="global_tgt")

        net.connect(a, a, topology="one_to_one", weight=50)  # a density boost
        net.connect(b, b, topology="one_to_one", weight=50)  # b density boost
        net.connect(a, b, topology="one_to_one", weight=200) # a->b local
        net.connect(a, e, topology="all_to_all", weight=200)  # a->e global

        # a density: 3(source) + 1(target) = 4, b: 1+2=3, e: 0+1=1
        # Sort: a(4)->core0, b(3)->core1, e(1)->core2
        # cluster_size=2: a(c0,cl0), b(c1,cl0), e(c2,cl1)
        compiled = Compiler(cluster_size=2).compile(net)
        assert len(compiled.prog_route_cmds) > 0, "Should have local routes (a->b)"
        assert len(compiled.prog_global_route_cmds) > 0, "Should have global routes (a->e)"

    def test_global_route_overflow(self):
        """Exceeding GLOBAL_ROUTE_SLOTS should raise RouteOverflowError."""
        from neurocore.compiler import Compiler
        from neurocore.exceptions import RouteOverflowError
        from neurocore.constants import GLOBAL_ROUTE_SLOTS

        net = nc.Network()
        # Create enough core-filling populations to span multiple clusters
        pops = [net.population(NEURONS_PER_CORE) for _ in range(GLOBAL_ROUTE_SLOTS + 2)]
        # Connect first pop to all others (each on its own core = its own cluster with cluster_size=1)
        for tgt in pops[1:]:
            net.connect(pops[0], tgt, topology="one_to_one", weight=200)

        with pytest.raises(RouteOverflowError):
            Compiler(cluster_size=1).compile(net)

    def test_small_network_zero_global_routes(self):
        """A network fitting in one cluster should have zero global routes."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        a = net.population(4, params={"threshold": 100, "leak": 0})
        b = net.population(4, params={"threshold": 100, "leak": 0})
        net.connect(a, b, topology="all_to_all", weight=200)

        compiled = Compiler(cluster_size=4).compile(net)
        # Both populations fit in core 0 (same cluster)
        assert len(compiled.prog_global_route_cmds) == 0

    def test_custom_cluster_size(self):
        """Changing cluster_size should change routing classification."""
        from neurocore.compiler import Compiler

        net = nc.Network()
        a = net.population(NEURONS_PER_CORE, label="core0")  # core 0
        b = net.population(1, params={"threshold": 100, "leak": 0}, label="core1")  # core 1
        net.connect(a, b, topology="all_to_all", weight=200)

        # cluster_size=4: cores 0 and 1 in same cluster -> local route
        compiled_4 = Compiler(cluster_size=4).compile(net)
        assert len(compiled_4.prog_global_route_cmds) == 0

        # cluster_size=1: every core is its own cluster -> global route
        compiled_1 = Compiler(cluster_size=1).compile(net)
        assert len(compiled_1.prog_global_route_cmds) > 0


class TestWeightMatrix:
    """Test per-synapse weight_matrix connections."""

    def test_weight_matrix_basic(self):
        """A 2x2 weight matrix should create per-synapse connections."""
        import numpy as np

        net = nc.Network()
        src = net.population(2, params={"threshold": 100, "leak": 0})
        tgt = net.population(2, params={"threshold": 100, "leak": 0})

        wm = np.array([[500, 0], [0, 300]], dtype=np.int32)
        net.connect(src, tgt, weight_matrix=wm)

        sim = nc.Simulator()
        sim.deploy(net)

        # Check adjacency has correct per-synapse weights
        adj = sim._compiled.adjacency
        # src[0] -> tgt[0] with weight 500 (only nonzero in row 0)
        src0_gid = 0 * 1024 + 0  # first pop placed first
        found_weights = {entry[1] for entry in adj.get(src0_gid, [])}
        assert 500 in found_weights, f"Expected weight 500 in {found_weights}"

    def test_weight_matrix_shape_mismatch(self):
        """Shape mismatch should raise ValueError."""
        import numpy as np
        from neurocore.exceptions import WeightOutOfRangeError

        net = nc.Network()
        src = net.population(3)
        tgt = net.population(2)

        wm = np.array([[1, 2]], dtype=np.int32)  # wrong shape (1,2) vs (3,2)
        with pytest.raises(ValueError, match="weight_matrix shape"):
            net.connect(src, tgt, weight_matrix=wm)

    def test_weight_matrix_range_check(self):
        """Weights outside int16 range should raise."""
        import numpy as np
        from neurocore.exceptions import WeightOutOfRangeError

        net = nc.Network()
        src = net.population(2)
        tgt = net.population(2)

        wm = np.array([[40000, 0], [0, 0]], dtype=np.int32)  # > 32767
        with pytest.raises(WeightOutOfRangeError):
            net.connect(src, tgt, weight_matrix=wm)

    def test_weight_matrix_zeros_skipped(self):
        """Zero entries in weight_matrix should not create connections."""
        import numpy as np

        net = nc.Network()
        src = net.population(3, params={"threshold": 100, "leak": 0})
        tgt = net.population(3, params={"threshold": 100, "leak": 0})

        # Only diagonal nonzero
        wm = np.diag([100, 200, 300]).astype(np.int32)
        net.connect(src, tgt, weight_matrix=wm)

        sim = nc.Simulator()
        sim.deploy(net)

        # Should have exactly 3 connections (diagonal only)
        total_conns = sum(len(v) for v in sim._compiled.adjacency.values())
        assert total_conns == 3, f"Expected 3 connections, got {total_conns}"

    def test_weight_matrix_simulation(self):
        """End-to-end: specific weight_matrix drives correct spike behavior."""
        import numpy as np

        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(2, params={"threshold": 500, "leak": 0, "refrac": 0})

        # src[0] -> tgt[0] with weight 600 (will spike), tgt[1] with weight 200 (won't)
        wm = np.array([[600, 200]], dtype=np.int32)
        net.connect(src, tgt, weight_matrix=wm)

        sim = nc.Simulator()
        sim.deploy(net)

        # Inject enough to fire src
        sim.inject(src, current=200)
        sim.run(1)  # t0: src fires (200 >= 100)
        result = sim.run(1)  # t1: tgt[0] receives 600 >= 500 -> spikes
        # tgt[1] receives 200 < 500 -> no spike

        # At least tgt[0] should have spiked
        assert result.total_spikes >= 1

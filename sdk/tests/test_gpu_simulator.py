import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.constants import (
    DEFAULT_THRESHOLD, DEFAULT_LEAK, DEFAULT_REFRAC, NEURONS_PER_CORE,
    TRACE_MAX, DEFAULT_TAU1, DEFAULT_TAU2,
)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

def _get_gpu_device():
    if torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    return torch.device("cuda:0")

def _gid(placement, pop, neuron_idx=0):
    core, nid = placement.neuron_map[(pop.id, neuron_idx)]
    return core * NEURONS_PER_CORE + nid

def _run_cpu(net, stimulus_fn, timesteps, learn_cfg=None):
    sim = nc.Simulator()
    sim.deploy(net)
    if learn_cfg:
        sim.set_learning(**learn_cfg)
    return _run_sim(sim, stimulus_fn, timesteps)

def _run_gpu(net, stimulus_fn, timesteps, learn_cfg=None):
    sim = nc.GpuSimulator(device=_get_gpu_device())
    sim.deploy(net)
    if learn_cfg:
        sim.set_learning(**learn_cfg)
    return _run_sim(sim, stimulus_fn, timesteps)

def _run_sim(sim, stimulus_fn, timesteps):
    if stimulus_fn is None:
        return sim.run(timesteps)

    all_trains = {}
    total = 0
    for t in range(timesteps):
        stimulus_fn(sim, t)
        result = sim.run(1)
        total += result.total_spikes
        for gid, times in result.spike_trains.items():
            if gid not in all_trains:
                all_trains[gid] = []
            all_trains[gid].extend([t_ + t for t_ in times])
    return _CombinedResult(total, timesteps, all_trains, result.placement)

class _CombinedResult:
    def __init__(self, total_spikes, timesteps, spike_trains, placement):
        self.total_spikes = total_spikes
        self.timesteps = timesteps
        self.spike_trains = spike_trains
        self.placement = placement

def _assert_trains_match(cpu_result, gpu_result, msg=""):
    cpu_trains = cpu_result.spike_trains
    gpu_trains = gpu_result.spike_trains
    all_gids = set(cpu_trains.keys()) | set(gpu_trains.keys())
    for gid in sorted(all_gids):
        cpu_times = cpu_trains.get(gid, [])
        gpu_times = gpu_trains.get(gid, [])
        assert cpu_times == gpu_times, (
            f"{msg}GID {gid}: CPU spikes={cpu_times}, GPU spikes={gpu_times}"
        )
    assert cpu_result.total_spikes == gpu_result.total_spikes, (
        f"{msg}Total: CPU={cpu_result.total_spikes}, GPU={gpu_result.total_spikes}"
    )

class TestSingleNeuronGPU:
    def test_constant_input_spike_timing(self):
        net = nc.Network()
        pop = net.population(1, params={"threshold": 1000, "leak": 3})

        def stim(sim, t):
            sim.inject(pop, current=200)

        cpu = _run_cpu(net, stim, 20)
        gpu = _run_gpu(net, stim, 20)
        _assert_trains_match(cpu, gpu, "SingleNeuron constant input: ")

    def test_refractory_period(self):
        net = nc.Network()
        pop = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 3})

        def stim(sim, t):
            sim.inject(pop, current=200)

        cpu = _run_cpu(net, stim, 20)
        gpu = _run_gpu(net, stim, 20)
        _assert_trains_match(cpu, gpu, "Refractory: ")

    def test_subthreshold_no_spikes(self):
        net = nc.Network()
        pop = net.population(1, params={"threshold": 1000, "leak": 100, "resting": 0})

        def stim(sim, t):
            sim.inject(pop, current=50)

        cpu = _run_cpu(net, stim, 10)
        gpu = _run_gpu(net, stim, 10)
        assert cpu.total_spikes == 0
        assert gpu.total_spikes == 0

class TestChainPropagationGPU:
    def test_spike_chain_4_neurons(self):
        net = nc.Network()
        n0 = net.population(1, label="n0")
        n1 = net.population(1, label="n1")
        n2 = net.population(1, label="n2")
        n3 = net.population(1, label="n3")
        net.connect(n0, n1, topology="all_to_all", weight=1200)
        net.connect(n1, n2, topology="all_to_all", weight=1200)
        net.connect(n2, n3, topology="all_to_all", weight=1200)

        def stim(sim, t):
            if t == 0:
                sim.inject(n0, current=1200)

        cpu = _run_cpu(net, stim, 10)
        gpu = _run_gpu(net, stim, 10)
        _assert_trains_match(cpu, gpu, "Chain: ")

        p = cpu.placement
        assert 0 in cpu.spike_trains.get(_gid(p, n0), [])
        assert 1 in cpu.spike_trains.get(_gid(p, n1), [])
        assert 2 in cpu.spike_trains.get(_gid(p, n2), [])
        assert 3 in cpu.spike_trains.get(_gid(p, n3), [])

class TestInhibitionGPU:
    def test_inhibitory_weight_prevents_spike(self):
        net = nc.Network()
        exc = net.population(1, label="exc")
        inh = net.population(1, label="inh")
        target = net.population(1, label="target")
        net.connect(exc, target, topology="all_to_all", weight=500)
        net.connect(inh, target, topology="all_to_all", weight=-600)

        def stim(sim, t):
            if t == 0:
                sim.inject(exc, current=1200)
                sim.inject(inh, current=1200)

        cpu = _run_cpu(net, stim, 5)
        gpu = _run_gpu(net, stim, 5)
        _assert_trains_match(cpu, gpu, "Inhibition: ")

        p = cpu.placement
        tgt_gid = _gid(p, target)
        assert 1 not in cpu.spike_trains.get(tgt_gid, [])
        assert 1 not in gpu.spike_trains.get(tgt_gid, [])

class TestGradedSpikesGPU:
    def test_graded_payload_scaling(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt = net.population(1, params={"threshold": 1000, "leak": 0})
        net.connect(src, tgt, topology="all_to_all", weight=200)

        def stim(sim, t):
            if t == 0:
                sim.inject(src, current=500)

        cfg = {"graded": True}
        cpu = _run_cpu(net, stim, 5, learn_cfg=cfg)
        gpu = _run_gpu(net, stim, 5, learn_cfg=cfg)
        _assert_trains_match(cpu, gpu, "Graded: ")

class TestDendriticCompartmentsGPU:
    def test_dendritic_threshold_suppression(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt = net.population(1, params={
            "threshold": 1000, "leak": 0, "dend_threshold": 500
        })
        net.connect(src, tgt, topology="all_to_all", weight=200, compartment=1)

        def stim(sim, t):
            if t == 0:
                sim.inject(src, current=200)

        cfg = {"dendritic": True}
        cpu = _run_cpu(net, stim, 5, learn_cfg=cfg)
        gpu = _run_gpu(net, stim, 5, learn_cfg=cfg)
        _assert_trains_match(cpu, gpu, "Dendritic: ")

        assert cpu.total_spikes == 1
        assert gpu.total_spikes == 1

class TestNoiseGPU:
    def test_noise_disabled_deterministic(self):
        net = nc.Network()
        pop = net.population(4, params={"threshold": 500, "leak": 3})

        def stim(sim, t):
            sim.inject(pop, current=100)

        cpu = _run_cpu(net, stim, 20)
        gpu = _run_gpu(net, stim, 20)
        _assert_trains_match(cpu, gpu, "NoNoise: ")

    def test_noise_enabled_matches_cpu(self):
        net = nc.Network()
        pop = net.population(4, params={
            "threshold": 500, "leak": 3,
            "noise_config": 0x34,
        })

        def stim(sim, t):
            sim.inject(pop, current=100)

        cfg = {"noise": True}
        cpu = _run_cpu(net, stim, 20, learn_cfg=cfg)
        gpu = _run_gpu(net, stim, 20, learn_cfg=cfg)
        _assert_trains_match(cpu, gpu, "Noise: ")

class TestDualTracesGPU:
    def test_both_traces_set_on_spike(self):
        net = nc.Network()
        pop = net.population(1, params={"threshold": 100, "leak": 0})

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.inject(pop, current=200)
        sim_gpu.run(1)

        assert int(sim_gpu._trace[0].item()) == TRACE_MAX
        assert int(sim_gpu._trace2[0].item()) == TRACE_MAX

    def test_different_decay_rates(self):
        net = nc.Network()
        pop = net.population(1, params={
            "threshold": 100, "leak": 0, "refrac": 0,
            "tau1": 2, "tau2": 6,
        })

        sim_cpu = nc.Simulator()
        sim_cpu.deploy(net)
        sim_cpu.inject(pop, current=200)
        sim_cpu.run(1)
        sim_cpu.run(5)
        cpu_t1 = int(sim_cpu._trace[0])
        cpu_t2 = int(sim_cpu._trace2[0])

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.inject(pop, current=200)
        sim_gpu.run(1)
        sim_gpu.run(5)
        gpu_t1 = int(sim_gpu._trace[0].item())
        gpu_t2 = int(sim_gpu._trace2[0].item())

        assert cpu_t1 == gpu_t1, f"trace1: CPU={cpu_t1}, GPU={gpu_t1}"
        assert cpu_t2 == gpu_t2, f"trace2: CPU={cpu_t2}, GPU={gpu_t2}"
        assert cpu_t1 < cpu_t2

    def test_min_step_1_convergence(self):
        net = nc.Network()
        pop = net.population(1, params={
            "threshold": 100, "leak": 0, "refrac": 0,
            "tau1": 8, "tau2": 8,
        })

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.inject(pop, current=200)
        sim_gpu.run(1)
        sim_gpu.run(200)

        assert int(sim_gpu._trace[0].item()) == 0
        assert int(sim_gpu._trace2[0].item()) == 0

class TestAxonDelaysGPU:
    def test_delay_zero_backward_compat(self):
        net = nc.Network()
        n0 = net.population(1, params={"threshold": 100, "leak": 0}, label="n0")
        n1 = net.population(1, params={"threshold": 100, "leak": 0}, label="n1")
        net.connect(n0, n1, topology="all_to_all", weight=200, delay=0)

        def stim(sim, t):
            if t == 0:
                sim.inject(n0, current=200)

        cpu = _run_cpu(net, stim, 5)
        gpu = _run_gpu(net, stim, 5)
        _assert_trains_match(cpu, gpu, "Delay0: ")

    def test_delay_3_shifts_spike(self):
        net = nc.Network()
        n0 = net.population(1, params={"threshold": 100, "leak": 0}, label="n0")
        n1 = net.population(1, params={"threshold": 100, "leak": 0}, label="n1")
        net.connect(n0, n1, topology="all_to_all", weight=200, delay=3)

        def stim(sim, t):
            if t == 0:
                sim.inject(n0, current=200)

        cpu = _run_cpu(net, stim, 10)
        gpu = _run_gpu(net, stim, 10)
        _assert_trains_match(cpu, gpu, "Delay3: ")

        p = cpu.placement
        n1_spikes = cpu.spike_trains.get(_gid(p, n1), [])
        assert len(n1_spikes) > 0
        assert n1_spikes[0] > 1

    def test_mixed_delays(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0}, label="src")
        fast = net.population(1, params={"threshold": 100, "leak": 0}, label="fast")
        slow = net.population(1, params={"threshold": 100, "leak": 0}, label="slow")
        net.connect(src, fast, topology="all_to_all", weight=200, delay=1)
        net.connect(src, slow, topology="all_to_all", weight=200, delay=5)

        def stim(sim, t):
            if t == 0:
                sim.inject(src, current=200)

        cpu = _run_cpu(net, stim, 10)
        gpu = _run_gpu(net, stim, 10)
        _assert_trains_match(cpu, gpu, "MixedDelay: ")

class TestSynapseFormatsGPU:
    def test_dense_matches_cpu(self):
        net = nc.Network()
        src = net.population(2, params={"threshold": 100, "leak": 0})
        tgt = net.population(2, params={"threshold": 100, "leak": 0})
        net.connect(src, tgt, topology="all_to_all", weight=200, format='dense')

        def stim(sim, t):
            if t == 0:
                sim.inject(src, current=200)

        cpu = _run_cpu(net, stim, 5)
        gpu = _run_gpu(net, stim, 5)
        _assert_trains_match(cpu, gpu, "Dense: ")

    def test_pop_matches_cpu(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0})
        tgt = net.population(4, params={"threshold": 100, "leak": 0})
        net.connect(src, tgt, topology="all_to_all", weight=300, format='pop')

        def stim(sim, t):
            if t == 0:
                sim.inject(src, current=200)

        cpu = _run_cpu(net, stim, 5)
        gpu = _run_gpu(net, stim, 5)
        _assert_trains_match(cpu, gpu, "Pop: ")

class TestSTDPGPU:
    def test_ltp_weight_increase(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        cfg = {"learn": True}

        sim_cpu = nc.Simulator()
        sim_cpu.deploy(net)
        sim_cpu.set_learning(**cfg)
        sim_cpu.inject(src, current=200)
        sim_cpu.run(1)
        sim_cpu.run(1)

        cpu_w = None
        for targets in sim_cpu._adjacency.values():
            for entry in targets:
                cpu_w = entry[1]

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.set_learning(**cfg)
        sim_gpu.inject(src, current=200)
        sim_gpu.run(1)
        sim_gpu.run(1)
        gpu_adj = sim_gpu.get_weights()
        gpu_w = None
        for targets in gpu_adj.values():
            for entry in targets:
                gpu_w = entry[1]

        assert cpu_w is not None and cpu_w > 500, f"CPU LTP failed: w={cpu_w}"
        assert gpu_w is not None and gpu_w > 500, f"GPU LTP failed: w={gpu_w}"
        assert cpu_w == gpu_w, f"Weight mismatch: CPU={cpu_w}, GPU={gpu_w}"

    def test_stdp_weight_evolution_100_steps(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 1})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 1})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        cfg = {"learn": True}

        def stim(sim, t):
            sim.inject(src, current=200)

        sim_cpu = nc.Simulator()
        sim_cpu.deploy(net)
        sim_cpu.set_learning(**cfg)
        for t in range(100):
            sim_cpu.inject(src, current=200)
            sim_cpu.run(1)
        cpu_w = None
        for targets in sim_cpu._adjacency.values():
            for entry in targets:
                cpu_w = entry[1]

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.set_learning(**cfg)
        for t in range(100):
            sim_gpu.inject(src, current=200)
            sim_gpu.run(1)
        gpu_adj = sim_gpu.get_weights()
        gpu_w = None
        for targets in gpu_adj.values():
            for entry in targets:
                gpu_w = entry[1]

        assert cpu_w == gpu_w, f"100-step STDP: CPU={cpu_w}, GPU={gpu_w}"

class TestThreeFactorGPU:
    def test_no_reward_no_weight_change(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        cfg = {"learn": True, "three_factor": True}

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.set_learning(**cfg)
        sim_gpu.inject(src, current=200)
        sim_gpu.inject(tgt, current=200)
        sim_gpu.run(5)

        gpu_adj = sim_gpu.get_weights()
        for targets in gpu_adj.values():
            for entry in targets:
                assert entry[1] == 500, f"Weight changed without reward: {entry[1]}"

    def test_reward_changes_weight(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        cfg = {"learn": True, "three_factor": True}

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.set_learning(**cfg)

        for _ in range(3):
            sim_gpu.inject(src, current=200)
            sim_gpu.inject(tgt, current=200)
            sim_gpu.run(1)

        sim_gpu.reward(500)
        sim_gpu.run(1)

        gpu_adj = sim_gpu.get_weights()
        weight_changed = False
        for targets in gpu_adj.values():
            for entry in targets:
                if entry[1] != 500:
                    weight_changed = True
        assert weight_changed, "Reward should modify weights via eligibility"

    def test_three_factor_cpu_gpu_match(self):
        net = nc.Network()
        src = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        tgt = net.population(1, params={"threshold": 100, "leak": 0, "refrac": 0})
        net.connect(src, tgt, topology="all_to_all", weight=500)

        cfg = {"learn": True, "three_factor": True}

        sim_cpu = nc.Simulator()
        sim_cpu.deploy(net)
        sim_cpu.set_learning(**cfg)
        for _ in range(3):
            sim_cpu.inject(src, current=200)
            sim_cpu.inject(tgt, current=200)
            sim_cpu.run(1)
        sim_cpu.reward(500)
        sim_cpu.run(1)
        cpu_w = None
        for targets in sim_cpu._adjacency.values():
            for entry in targets:
                cpu_w = entry[1]

        sim_gpu = nc.GpuSimulator(device=_get_gpu_device())
        sim_gpu.deploy(net)
        sim_gpu.set_learning(**cfg)
        for _ in range(3):
            sim_gpu.inject(src, current=200)
            sim_gpu.inject(tgt, current=200)
            sim_gpu.run(1)
        sim_gpu.reward(500)
        sim_gpu.run(1)
        gpu_adj = sim_gpu.get_weights()
        gpu_w = None
        for targets in gpu_adj.values():
            for entry in targets:
                gpu_w = entry[1]

        assert cpu_w == gpu_w, f"3-factor: CPU={cpu_w}, GPU={gpu_w}"

class TestScalingGPU:
    @pytest.mark.parametrize("n_neurons,p", [(64, 0.1), (256, 0.05), (1024, 0.015)])
    def test_multi_neuron_match(self, n_neurons, p):
        net = nc.Network()
        pop = net.population(n_neurons, params={"threshold": 500, "leak": 3})
        net.connect(pop, pop, topology="random_sparse", p=p, weight=200, seed=42)

        def stim(sim, t):
            if t < 5:
                sim.inject(pop[:8], current=1200)

        cpu = _run_cpu(net, stim, 20)
        gpu = _run_gpu(net, stim, 20)
        _assert_trains_match(cpu, gpu, f"Scale {n_neurons}: ")

    def test_4096_neurons_runs(self):
        net = nc.Network()
        pop = net.population(4096, params={"threshold": 500, "leak": 3})
        net.connect(pop, pop, topology="fixed_fan_out", fan_out=4, weight=200, seed=42)

        sim = nc.GpuSimulator(device=_get_gpu_device())
        sim.deploy(net)
        sim.inject(pop[:16], current=1200)
        result = sim.run(10)
        assert result.total_spikes > 0
        assert result.timesteps == 10
        sim.close()

class TestRunResultGPU:
    def test_backend_tag(self):
        net = nc.Network()
        pop = net.population(4)
        sim = nc.GpuSimulator(device=_get_gpu_device())
        sim.deploy(net)
        result = sim.run(1)
        assert result.backend == "gpu_simulator"

    def test_status(self):
        net = nc.Network()
        pop = net.population(4)
        sim = nc.GpuSimulator(device=_get_gpu_device())
        sim.deploy(net)
        sim.run(5)
        s = sim.status()
        assert s["timestep_count"] == 5

    def test_async_raises(self):
        net = nc.Network()
        pop = net.population(4)
        sim = nc.GpuSimulator(device=_get_gpu_device())
        sim.deploy(net)
        with pytest.raises(nc.NeurocoreError):
            sim.set_learning(async_mode=True)

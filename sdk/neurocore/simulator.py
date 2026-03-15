import numpy as np
from collections import defaultdict

from .backend import Backend
from .compiler import Compiler, CompiledNetwork
from .network import Network, Population, PopulationSlice
from .constants import (
    MAX_CORES, NEURONS_PER_CORE, GRADE_SHIFT,
    TRACE_MAX, TRACE_DECAY, LEARN_SHIFT,
    WEIGHT_MAX_STDP, WEIGHT_MIN_STDP,
    REWARD_SHIFT, ELIG_DECAY_SHIFT, ELIG_MAX,
    DEFAULT_THRESHOLD, DEFAULT_LEAK, DEFAULT_RESTING, DEFAULT_REFRAC,
    DEFAULT_DEND_THRESHOLD, DEFAULT_NOISE_CONFIG, DEFAULT_TAU1, DEFAULT_TAU2,
    NOISE_LFSR_SEED, NOISE_LFSR_TAPS,
    DELAY_QUEUE_BUCKETS,
)
from .microcode import (
    execute_program, R_TRACE1, R_TRACE2, R_WEIGHT, R_ELIG, R_CONST,
    R_TEMP0, R_TEMP1, R_REWARD, LTD_START, LTD_END, LTP_START, LTP_END,
)
from .exceptions import NeurocoreError

ASYNC_MAX_MICRO_STEPS = 10000

class Simulator(Backend):

    def __init__(self, num_cores=MAX_CORES):
        self.max_cores = num_cores
        self._compiled = None
        self._compiler = Compiler(max_cores=num_cores, pool_depth=2**20)
        self._n = 0

        self._potential = None
        self._refrac = None
        self._trace = None

        self._threshold = None
        self._leak = None
        self._resting = None
        self._refrac_period = None
        self._dend_threshold = None

        self._adjacency = None
        self._intra_core_adj = None
        self._inter_core_adj = None

        self._noise_config = None
        self._noise_enable = False
        self._lfsr = None

        self._trace2 = None
        self._tau1 = None
        self._tau2 = None

        self._learning_rule = None

        self._learn_enable = False
        self._graded_enable = False
        self._dendritic_enable = False
        self._async_enable = False
        self._three_factor_enable = False
        self._noise_enable = False

        self._ext_current = None

        self._pending_spikes = []

        self._delay_queue = None

        self._timestep_count = 0

        self._eligibility = None
        self._reward_value = 0
        self._reward_pending = False

    def deploy(self, network_or_compiled):
        if isinstance(network_or_compiled, Network):
            self._compiled = self._compiler.compile(network_or_compiled)
        elif isinstance(network_or_compiled, CompiledNetwork):
            self._compiled = network_or_compiled
        else:
            raise TypeError(f"Expected Network or CompiledNetwork, got {type(network_or_compiled)}")

        n = self._compiled.placement.total_neurons
        self._n = n

        self._potential = np.zeros(n, dtype=np.int32)
        self._refrac = np.zeros(n, dtype=np.int32)
        self._trace = np.zeros(n, dtype=np.int32)
        self._ext_current = np.zeros(n, dtype=np.int32)

        self._threshold = np.full(n, DEFAULT_THRESHOLD, dtype=np.int32)
        self._leak = np.full(n, DEFAULT_LEAK, dtype=np.int32)
        self._resting = np.full(n, DEFAULT_RESTING, dtype=np.int32)
        self._refrac_period = np.full(n, DEFAULT_REFRAC, dtype=np.int32)
        self._dend_threshold = np.full(n, DEFAULT_DEND_THRESHOLD, dtype=np.int32)
        self._noise_config = np.full(n, DEFAULT_NOISE_CONFIG, dtype=np.uint8)
        self._tau1 = np.full(n, DEFAULT_TAU1, dtype=np.int32)
        self._tau2 = np.full(n, DEFAULT_TAU2, dtype=np.int32)
        self._trace2 = np.zeros(n, dtype=np.int32)
        self._lfsr = np.zeros(n, dtype=np.uint16)
        lfsr = NOISE_LFSR_SEED
        for gid in range(n):
            self._lfsr[gid] = lfsr
            bit = lfsr & 1
            lfsr >>= 1
            if bit:
                lfsr ^= NOISE_LFSR_TAPS

        for gid, params in self._compiled.neuron_params.items():
            if gid < n:
                self._threshold[gid] = params.threshold
                self._leak[gid] = params.leak
                self._resting[gid] = params.resting
                self._refrac_period[gid] = params.refrac
                self._dend_threshold[gid] = params.dend_threshold
                self._noise_config[gid] = params.noise_config
                self._tau1[gid] = params.tau1
                self._tau2[gid] = params.tau2

        self._adjacency = dict(self._compiled.adjacency)

        self._intra_core_adj = defaultdict(list)
        self._inter_core_adj = defaultdict(list)
        for src_gid, targets in self._adjacency.items():
            src_core = src_gid // NEURONS_PER_CORE
            for entry in targets:
                tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                delay = entry[3] if len(entry) > 3 else 0
                tgt_core = tgt_gid // NEURONS_PER_CORE
                if src_core == tgt_core:
                    self._intra_core_adj[src_gid].append((tgt_gid, weight, comp, delay))
                else:
                    self._inter_core_adj[src_gid].append((tgt_gid, weight, comp, delay))

        cfg = self._compiled.learn_config
        self._learn_enable = cfg.get("learn_enable", False)
        self._graded_enable = cfg.get("graded_enable", False)
        self._dendritic_enable = cfg.get("dendritic_enable", False)
        self._async_enable = cfg.get("async_enable", False)
        self._noise_enable = cfg.get("noise_enable", False)

        self._learning_rule = self._compiled.learning_rule

        self._eligibility = defaultdict(int)
        self._reward_value = 0
        self._reward_pending = False

        self._pending_spikes = []
        self._delay_queue = defaultdict(list)
        self._timestep_count = 0

    def inject(self, target, current):
        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")
        resolved = self._resolve_targets(target)
        for core, neuron in resolved:
            gid = core * NEURONS_PER_CORE + neuron
            if gid < self._n:
                self._ext_current[gid] = current

    def reward(self, value):
        self._reward_value = int(value)
        self._reward_pending = True

    def run(self, timesteps):
        from .result import RunResult

        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")

        if self._async_enable:
            return self._run_async(timesteps)

        return self._run_sync(timesteps)

    def _run_sync(self, timesteps):
        from .result import RunResult

        n = self._n
        spike_trains = defaultdict(list)
        total_spikes = 0

        weights = {}
        if self._learn_enable:
            for src, targets in self._adjacency.items():
                weights[src] = list(targets)

        for t in range(timesteps):
            acc_soma = np.zeros(n, dtype=np.int32)
            acc_dend = [np.zeros(n, dtype=np.int32) for _ in range(3)]

            bucket = self._timestep_count % DELAY_QUEUE_BUCKETS
            for tgt_gid, delivered, comp in self._delay_queue.pop(bucket, []):
                if comp == 0:
                    acc_soma[tgt_gid] += delivered
                elif 1 <= comp <= 3:
                    acc_dend[comp - 1][tgt_gid] += delivered

            for spike_gid, payload in self._pending_spikes:
                adj = (weights if self._learn_enable else self._adjacency)
                targets = adj.get(spike_gid, [])
                for entry in targets:
                    tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                    delay = entry[3] if len(entry) > 3 else 0
                    if tgt_gid >= n:
                        continue
                    if self._graded_enable:
                        delivered = (weight * payload) >> GRADE_SHIFT
                    else:
                        delivered = weight
                    if delay > 0:
                        future = (self._timestep_count + delay) % DELAY_QUEUE_BUCKETS
                        self._delay_queue[future].append((tgt_gid, delivered, comp))
                    elif comp == 0:
                        acc_soma[tgt_gid] += delivered
                    elif 1 <= comp <= 3:
                        acc_dend[comp - 1][tgt_gid] += delivered

            acc_soma += self._ext_current

            new_spikes = self._update_neurons(range(n), acc_soma, acc_dend)

            total_spikes += len(new_spikes)
            for gid, payload in new_spikes:
                spike_trains[gid].append(t)

            if self._learn_enable:
                if self._three_factor_enable:
                    self._elig_update(weights, new_spikes)
                    if self._reward_pending:
                        self._reward_apply(weights)
                        self._reward_pending = False
                    self._elig_decay()
                else:
                    self._stdp_update(weights, new_spikes)

            self._pending_spikes = new_spikes
            self._ext_current[:] = 0
            self._timestep_count += 1

        if self._learn_enable:
            self._adjacency = weights

        return RunResult(
            total_spikes=total_spikes,
            timesteps=timesteps,
            spike_trains=dict(spike_trains),
            placement=self._compiled.placement,
            backend="simulator",
        )

    def _run_async(self, timesteps):
        from .result import RunResult

        n = self._n
        num_cores = self._compiled.placement.num_cores_used
        spike_trains = defaultdict(list)
        total_spikes = 0

        for t in range(timesteps):
            pcif = defaultdict(list)

            for gid in range(n):
                if self._ext_current[gid] != 0:
                    core = gid // NEURONS_PER_CORE
                    pcif[core].append((gid, int(self._ext_current[gid])))

            for spike_gid, payload in self._pending_spikes:
                targets = self._inter_core_adj.get(spike_gid, [])
                for entry in targets:
                    tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                    if tgt_gid >= n:
                        continue
                    tgt_core = tgt_gid // NEURONS_PER_CORE
                    if self._graded_enable:
                        delivered = (weight * payload) >> GRADE_SHIFT
                    else:
                        delivered = weight
                    pcif[tgt_core].append((tgt_gid, delivered, comp))

            core_internal_spikes = defaultdict(list)
            for spike_gid, payload in self._pending_spikes:
                src_core = spike_gid // NEURONS_PER_CORE
                intra_targets = self._intra_core_adj.get(spike_gid, [])
                for entry in intra_targets:
                    tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                    if self._graded_enable:
                        delivered = (weight * payload) >> GRADE_SHIFT
                    else:
                        delivered = weight
                    core_internal_spikes[src_core].append((tgt_gid, delivered, comp))

            core_needs_restart = set()
            all_new_spikes = []
            micro_step = 0

            while micro_step < ASYNC_MAX_MICRO_STEPS:
                micro_step += 1

                active_cores = set()
                for c in range(num_cores):
                    if pcif[c] or core_internal_spikes[c] or c in core_needs_restart:
                        active_cores.add(c)

                if not active_cores:
                    break

                new_inter_core = []
                core_needs_restart_next = set()

                for core_id in sorted(active_cores):
                    core_start = core_id * NEURONS_PER_CORE
                    core_end = min(core_start + NEURONS_PER_CORE, n)
                    acc_soma = np.zeros(n, dtype=np.int32)
                    acc_dend = [np.zeros(n, dtype=np.int32) for _ in range(3)]

                    for entry in pcif[core_id]:
                        if len(entry) == 2:
                            gid, current = entry
                            acc_soma[gid] += current
                        else:
                            gid, current, comp = entry
                            if comp == 0:
                                acc_soma[gid] += current
                            elif 1 <= comp <= 3:
                                acc_dend[comp - 1][gid] += current
                    pcif[core_id] = []

                    for entry in core_internal_spikes[core_id]:
                        tgt_gid, delivered, comp = entry
                        if comp == 0:
                            acc_soma[tgt_gid] += delivered
                        elif 1 <= comp <= 3:
                            acc_dend[comp - 1][tgt_gid] += delivered
                    core_internal_spikes[core_id] = []
                    core_needs_restart.discard(core_id)

                    neuron_range = range(core_start, core_end)
                    core_spikes = self._update_neurons(neuron_range, acc_soma, acc_dend)

                    if core_spikes:
                        core_needs_restart_next.add(core_id)

                    for spike_gid, payload in core_spikes:
                        all_new_spikes.append((spike_gid, payload))
                        spike_trains[spike_gid].append(t)

                        intra_targets = self._intra_core_adj.get(spike_gid, [])
                        for entry in intra_targets:
                            tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                            if self._graded_enable:
                                delivered = (weight * payload) >> GRADE_SHIFT
                            else:
                                delivered = weight
                            core_internal_spikes[core_id].append(
                                (tgt_gid, delivered, comp))

                        inter_targets = self._inter_core_adj.get(spike_gid, [])
                        for entry in inter_targets:
                            tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                            if tgt_gid >= n:
                                continue
                            tgt_core = tgt_gid // NEURONS_PER_CORE
                            if self._graded_enable:
                                delivered = (weight * payload) >> GRADE_SHIFT
                            else:
                                delivered = weight
                            pcif[tgt_core].append((tgt_gid, delivered, comp))

                core_needs_restart = core_needs_restart_next

            total_spikes += len(all_new_spikes)
            self._pending_spikes = []
            self._ext_current[:] = 0
            self._timestep_count += 1

        return RunResult(
            total_spikes=total_spikes,
            timesteps=timesteps,
            spike_trains=dict(spike_trains),
            placement=self._compiled.placement,
            backend="simulator",
        )

    def _decay_trace(self, trace_val, tau):
        if trace_val <= 0:
            return 0
        decay = trace_val >> tau
        if decay == 0:
            decay = 1
        return max(0, trace_val - decay)

    def _advance_lfsr(self, i):
        lfsr = int(self._lfsr[i])
        bit = lfsr & 1
        lfsr >>= 1
        if bit:
            lfsr ^= NOISE_LFSR_TAPS
        self._lfsr[i] = lfsr
        return lfsr

    def _update_neurons(self, neuron_range, acc_soma, acc_dend):
        new_spikes = []
        for i in neuron_range:
            total_input = int(acc_soma[i])
            if self._dendritic_enable:
                dthr = int(self._dend_threshold[i])
                for d in range(3):
                    dval = int(acc_dend[d][i])
                    if dval > dthr:
                        total_input += dval - dthr

            potential = int(self._potential[i])
            refrac = int(self._refrac[i])
            leak = int(self._leak[i])
            threshold = int(self._threshold[i])
            resting = int(self._resting[i])
            trace = int(self._trace[i])
            trace2 = int(self._trace2[i])
            tau1 = int(self._tau1[i])
            tau2 = int(self._tau2[i])

            if self._noise_enable:
                cfg = int(self._noise_config[i])
                mantissa = cfg & 0x0F
                exponent = (cfg >> 4) & 0x0F
                if mantissa > 0:
                    lfsr = self._advance_lfsr(i)
                    noise_mask = mantissa << exponent
                    noise_val = (lfsr & noise_mask) - (noise_mask >> 1)
                    threshold = threshold + noise_val

            if refrac > 0:
                self._potential[i] = resting
                self._refrac[i] = refrac - 1
                self._trace[i] = self._decay_trace(trace, tau1)
                self._trace2[i] = self._decay_trace(trace2, tau2)
            elif potential + total_input - leak >= threshold:
                excess = potential + total_input - leak - threshold
                payload = max(1, min(255, excess))
                self._potential[i] = resting
                self._refrac[i] = int(self._refrac_period[i])
                self._trace[i] = TRACE_MAX
                self._trace2[i] = TRACE_MAX
                new_spikes.append((i, payload if self._graded_enable else 128))
            elif potential + total_input > leak:
                self._potential[i] = potential + total_input - leak
                self._trace[i] = self._decay_trace(trace, tau1)
                self._trace2[i] = self._decay_trace(trace2, tau2)
            else:
                self._potential[i] = resting
                self._trace[i] = self._decay_trace(trace, tau1)
                self._trace2[i] = self._decay_trace(trace2, tau2)

        return new_spikes

    def _stdp_update(self, weights, new_spikes):
        if self._learning_rule is not None:
            self._microcode_learn(weights, new_spikes, three_factor=False)
            return

        for spike_gid, _ in new_spikes:
            if spike_gid in weights:
                updated = []
                for entry in weights[spike_gid]:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt < self._n:
                        post_trace = int(self._trace[tgt])
                        if post_trace > 0:
                            delta = post_trace >> LEARN_SHIFT
                            w = max(WEIGHT_MIN_STDP, w - delta)
                    updated.append((tgt, w, c, *rest))
                weights[spike_gid] = updated

            for src, targets in weights.items():
                if src == spike_gid:
                    continue
                updated = []
                for entry in targets:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt == spike_gid:
                        pre_trace = int(self._trace[src])
                        if pre_trace > 0:
                            delta = pre_trace >> LEARN_SHIFT
                            w = min(WEIGHT_MAX_STDP, w + delta)
                    updated.append((tgt, w, c, *rest))
                weights[src] = updated

    def _elig_update(self, weights, new_spikes):
        if self._learning_rule is not None:
            self._microcode_learn(weights, new_spikes, three_factor=True)
            return

        for spike_gid, _ in new_spikes:
            if spike_gid in weights:
                for entry in weights[spike_gid]:
                    tgt = entry[0]
                    if tgt < self._n:
                        post_trace = int(self._trace[tgt])
                        if post_trace > 0:
                            delta = post_trace >> LEARN_SHIFT
                            key = (spike_gid, tgt)
                            self._eligibility[key] = max(
                                -ELIG_MAX,
                                self._eligibility[key] - delta)

            for src, targets in weights.items():
                if src == spike_gid:
                    continue
                for entry in targets:
                    tgt = entry[0]
                    if tgt == spike_gid:
                        pre_trace = int(self._trace[src])
                        if pre_trace > 0:
                            delta = pre_trace >> LEARN_SHIFT
                            key = (src, spike_gid)
                            self._eligibility[key] = min(
                                ELIG_MAX,
                                self._eligibility[key] + delta)

    def _reward_apply(self, weights):
        reward = self._reward_value
        if reward == 0:
            return

        for src in list(weights.keys()):
            updated = []
            for entry in weights[src]:
                tgt, w, c = entry[0], entry[1], entry[2]
                rest = entry[3:]
                key = (src, tgt)
                elig = self._eligibility.get(key, 0)
                if elig != 0:
                    delta = (elig * reward) >> REWARD_SHIFT
                    w = max(WEIGHT_MIN_STDP, min(WEIGHT_MAX_STDP, w + delta))
                updated.append((tgt, w, c, *rest))
            weights[src] = updated

        self._reward_value = 0

    def _elig_decay(self):
        to_delete = []
        for key in self._eligibility:
            val = self._eligibility[key]
            if val > 0:
                val -= max(1, val >> ELIG_DECAY_SHIFT)
            elif val < 0:
                val += max(1, (-val) >> ELIG_DECAY_SHIFT)
            if val == 0:
                to_delete.append(key)
            else:
                self._eligibility[key] = val
        for key in to_delete:
            del self._eligibility[key]

    def _microcode_learn(self, weights, new_spikes, three_factor=False):
        program = self._learning_rule.get_program()

        for spike_gid, _ in new_spikes:
            if spike_gid in weights:
                updated = []
                for entry in weights[spike_gid]:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt < self._n:
                        post_trace1 = int(self._trace[tgt])
                        post_trace2 = int(self._trace2[tgt])
                        elig = self._eligibility.get((spike_gid, tgt), 0)
                        regs = [post_trace1, post_trace2, w, elig,
                                0, 0, 0, self._reward_value]
                        result = execute_program(
                            program, LTD_START, LTD_END + 1, regs)
                        if three_factor:
                            if result["elig_written"]:
                                new_elig = max(-ELIG_MAX, min(ELIG_MAX, result["elig"]))
                                self._eligibility[(spike_gid, tgt)] = new_elig
                        else:
                            if result["weight_written"]:
                                w = max(WEIGHT_MIN_STDP, min(WEIGHT_MAX_STDP, result["weight"]))
                    updated.append((tgt, w, c, *rest))
                weights[spike_gid] = updated

            for src, targets in weights.items():
                if src == spike_gid:
                    continue
                updated = []
                for entry in targets:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt == spike_gid:
                        pre_trace1 = int(self._trace[src])
                        pre_trace2 = int(self._trace2[src])
                        elig = self._eligibility.get((src, tgt), 0)
                        regs = [pre_trace1, pre_trace2, w, elig,
                                0, 0, 0, self._reward_value]
                        result = execute_program(
                            program, LTP_START, LTP_END + 1, regs)
                        if three_factor:
                            if result["elig_written"]:
                                new_elig = max(-ELIG_MAX, min(ELIG_MAX, result["elig"]))
                                self._eligibility[(src, tgt)] = new_elig
                        else:
                            if result["weight_written"]:
                                w = max(WEIGHT_MIN_STDP, min(WEIGHT_MAX_STDP, result["weight"]))
                    updated.append((tgt, w, c, *rest))
                weights[src] = updated

    def set_learning(self, learn=False, graded=False, dendritic=False,
                     async_mode=False, three_factor=False, noise=False):
        self._learn_enable = learn
        self._graded_enable = graded
        self._dendritic_enable = dendritic
        self._async_enable = async_mode
        self._three_factor_enable = three_factor
        self._noise_enable = noise
        if three_factor and not learn:
            self._learn_enable = True

    def status(self):
        return {
            "state": 0,
            "timestep_count": self._timestep_count,
        }

    def close(self):
        pass

    def _resolve_targets(self, target):
        if isinstance(target, list):
            return target
        placement = self._compiled.placement
        if isinstance(target, PopulationSlice):
            return [
                placement.neuron_map[(target.population.id, i)]
                for i in target.indices
            ]
        if isinstance(target, Population):
            return [
                placement.neuron_map[(target.id, i)]
                for i in range(target.size)
            ]
        raise TypeError(f"Cannot resolve target of type {type(target)}")

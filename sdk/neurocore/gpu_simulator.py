import torch
import numpy as np
from collections import defaultdict

from .backend import Backend
from .compiler import Compiler, CompiledNetwork
from .network import Network, Population, PopulationSlice
from .constants import (
    MAX_CORES, NEURONS_PER_CORE, GRADE_SHIFT,
    TRACE_MAX, LEARN_SHIFT,
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

class GpuSimulator(Backend):

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device
        self._compiler = Compiler()
        self._compiled = None
        self._n = 0
        self._timestep_count = 0

        self._potential = None
        self._refrac = None
        self._trace = None
        self._trace2 = None
        self._ext_current = None

        self._threshold = None
        self._leak = None
        self._resting = None
        self._refrac_period = None
        self._dend_threshold = None
        self._noise_config = None
        self._tau1 = None
        self._tau2 = None
        self._lfsr = None

        self._W_soma = None
        self._W_dend = [None] * 3

        self._has_delays = False
        self._delay_buf_soma = None
        self._delay_buf_dend = None
        self._delay_src_ids = None
        self._delay_tgt_ids = None
        self._delay_weights = None
        self._delay_comps = None
        self._delay_values = None

        self._prev_spike_vec = None
        self._spike_mask = None

        self._learn_enable = False
        self._graded_enable = False
        self._dendritic_enable = False
        self._three_factor_enable = False
        self._noise_enable = False

        self._learning_rule = None
        self._elig_crow = None
        self._elig_col = None
        self._elig_vals = None
        self._reward_value = 0
        self._reward_pending = False

        self._stdp_mask = None

        self._soma_crow = None
        self._soma_col = None
        self._soma_row_idx = None

        self._adjacency = None

    def deploy(self, network_or_compiled):
        if isinstance(network_or_compiled, Network):
            self._compiled = self._compiler.compile(network_or_compiled)
        elif isinstance(network_or_compiled, CompiledNetwork):
            self._compiled = network_or_compiled
        else:
            raise TypeError(f"Expected Network or CompiledNetwork, got {type(network_or_compiled)}")

        n = self._compiled.placement.total_neurons
        self._n = n
        dev = self.device

        self._potential = torch.zeros(n, dtype=torch.int32, device=dev)
        self._refrac = torch.zeros(n, dtype=torch.int32, device=dev)
        self._trace = torch.zeros(n, dtype=torch.int32, device=dev)
        self._trace2 = torch.zeros(n, dtype=torch.int32, device=dev)
        self._ext_current = torch.zeros(n, dtype=torch.int32, device=dev)

        self._threshold = torch.full((n,), DEFAULT_THRESHOLD, dtype=torch.int32, device=dev)
        self._leak = torch.full((n,), DEFAULT_LEAK, dtype=torch.int32, device=dev)
        self._resting = torch.full((n,), DEFAULT_RESTING, dtype=torch.int32, device=dev)
        self._refrac_period = torch.full((n,), DEFAULT_REFRAC, dtype=torch.int32, device=dev)
        self._dend_threshold = torch.full((n,), DEFAULT_DEND_THRESHOLD, dtype=torch.int32, device=dev)
        self._noise_config = torch.full((n,), DEFAULT_NOISE_CONFIG, dtype=torch.int32, device=dev)
        self._tau1 = torch.full((n,), DEFAULT_TAU1, dtype=torch.int32, device=dev)
        self._tau2 = torch.full((n,), DEFAULT_TAU2, dtype=torch.int32, device=dev)

        lfsr_seeds = np.zeros(n, dtype=np.int32)
        lfsr = NOISE_LFSR_SEED
        for gid in range(n):
            lfsr_seeds[gid] = lfsr
            bit = lfsr & 1
            lfsr >>= 1
            if bit:
                lfsr ^= NOISE_LFSR_TAPS
        self._lfsr = torch.from_numpy(lfsr_seeds).to(dev)

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
        self._build_weight_matrices(n)

        cfg = self._compiled.learn_config
        self._learn_enable = cfg.get("learn_enable", False)
        self._graded_enable = cfg.get("graded_enable", False)
        self._dendritic_enable = cfg.get("dendritic_enable", False)
        self._noise_enable = cfg.get("noise_enable", False)

        self._learning_rule = self._compiled.learning_rule

        self._prev_spike_vec = torch.zeros(n, dtype=torch.float32, device=dev)

        self._reward_value = 0
        self._reward_pending = False

        if self._W_soma is not None and self._W_soma._nnz() > 0:
            self._elig_crow = self._soma_crow
            self._elig_col = self._soma_col
            self._elig_vals = torch.zeros(self._W_soma._nnz(), dtype=torch.float32, device=dev)
        else:
            self._elig_vals = None

        self._timestep_count = 0

    def _build_weight_matrices(self, n):
        dev = self.device

        rows_imm = [[] for _ in range(4)]
        cols_imm = [[] for _ in range(4)]
        vals_imm = [[] for _ in range(4)]

        delay_srcs, delay_tgts, delay_wts, delay_comps, delay_vals = [], [], [], [], []

        for src_gid, targets in self._adjacency.items():
            for entry in targets:
                tgt_gid, weight, comp = entry[0], entry[1], entry[2]
                delay = entry[3] if len(entry) > 3 else 0
                if tgt_gid >= n:
                    continue
                if delay > 0:
                    delay_srcs.append(src_gid)
                    delay_tgts.append(tgt_gid)
                    delay_wts.append(float(weight))
                    delay_comps.append(comp)
                    delay_vals.append(delay)
                else:
                    rows_imm[comp].append(tgt_gid)
                    cols_imm[comp].append(src_gid)
                    vals_imm[comp].append(float(weight))

        def _build_csr(rows, cols, vals):
            if not rows:
                return torch.sparse_csr_tensor(
                    torch.zeros(n + 1, dtype=torch.int32),
                    torch.tensor([], dtype=torch.int32),
                    torch.tensor([], dtype=torch.float32),
                    size=(n, n),
                ).to(dev)
            indices = torch.tensor([rows, cols], dtype=torch.int64)
            values = torch.tensor(vals, dtype=torch.float32)
            coo = torch.sparse_coo_tensor(indices, values, (n, n))
            coo = coo.coalesce()
            return coo.to_sparse_csr().to(dev)

        self._W_soma = _build_csr(rows_imm[0], cols_imm[0], vals_imm[0])
        for d in range(3):
            self._W_dend[d] = _build_csr(rows_imm[d + 1], cols_imm[d + 1], vals_imm[d + 1])

        self._soma_crow = self._W_soma.crow_indices()
        self._soma_col = self._W_soma.col_indices()
        if self._W_soma._nnz() > 0:
            self._soma_row_idx = torch.repeat_interleave(
                torch.arange(n, device=dev),
                self._soma_crow[1:] - self._soma_crow[:-1]
            )
        else:
            self._soma_row_idx = torch.tensor([], dtype=torch.int64, device=dev)

        if delay_srcs:
            self._has_delays = True
            self._delay_src_ids = torch.tensor(delay_srcs, dtype=torch.int64, device=dev)
            self._delay_tgt_ids = torch.tensor(delay_tgts, dtype=torch.int64, device=dev)
            self._delay_weights = torch.tensor(delay_wts, dtype=torch.float32, device=dev)
            self._delay_comps = torch.tensor(delay_comps, dtype=torch.int64, device=dev)
            self._delay_values = torch.tensor(delay_vals, dtype=torch.int64, device=dev)
            self._delay_buf_soma = torch.zeros(DELAY_QUEUE_BUCKETS, n, dtype=torch.float32, device=dev)
            self._delay_buf_dend = torch.zeros(3, DELAY_QUEUE_BUCKETS, n, dtype=torch.float32, device=dev)
        else:
            self._has_delays = False

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

        if getattr(self, '_async_enable', False):
            raise NeurocoreError("Async mode not supported on GPU simulator. Use sync mode.")

        return self._run_sync(timesteps)

    @torch.no_grad()
    def _run_sync(self, timesteps):
        from .result import RunResult

        n = self._n
        dev = self.device
        spike_trains = defaultdict(list)
        total_spikes = 0

        acc_soma = torch.zeros(n, dtype=torch.float32, device=dev)
        acc_dend = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in range(3)]
        zero_f = torch.zeros(n, dtype=torch.float32, device=dev)

        for t in range(timesteps):
            acc_soma.zero_()
            for d in range(3):
                acc_dend[d].zero_()

            if self._has_delays:
                bucket = self._timestep_count % DELAY_QUEUE_BUCKETS
                acc_soma.add_(self._delay_buf_soma[bucket])
                self._delay_buf_soma[bucket].zero_()
                for d in range(3):
                    acc_dend[d].add_(self._delay_buf_dend[d, bucket])
                    self._delay_buf_dend[d, bucket].zero_()

            if self._prev_spike_vec.any():
                spike_col = self._prev_spike_vec.unsqueeze(1)

                if self._graded_enable:
                    raw = torch.sparse.mm(self._W_soma, spike_col).squeeze(1)
                    acc_soma.add_(torch.div(raw, 128, rounding_mode='trunc'))
                    if self._dendritic_enable:
                        for d in range(3):
                            raw_d = torch.sparse.mm(self._W_dend[d], spike_col).squeeze(1)
                            acc_dend[d].add_(torch.div(raw_d, 128, rounding_mode='trunc'))
                else:
                    binary_vec = (self._prev_spike_vec > 0).float().unsqueeze(1)
                    acc_soma.add_(torch.sparse.mm(self._W_soma, binary_vec).squeeze(1))
                    if self._dendritic_enable:
                        for d in range(3):
                            acc_dend[d].add_(torch.sparse.mm(self._W_dend[d], binary_vec).squeeze(1))

                if self._has_delays:
                    self._deliver_delayed()

            acc_soma.add_(self._ext_current.float())

            spike_vec, spike_mask = self._update_neurons_gpu(acc_soma, acc_dend)

            if spike_mask.any():
                spiking_ids = spike_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                total_spikes += len(spiking_ids)
                for gid in spiking_ids:
                    spike_trains[int(gid)].append(t)

            if self._learn_enable:
                if self._three_factor_enable:
                    self._elig_update_gpu(spike_mask)
                    if self._reward_pending:
                        self._reward_apply_gpu()
                        self._reward_pending = False
                    self._elig_decay_gpu()
                else:
                    self._stdp_update_gpu(spike_mask)

            self._prev_spike_vec = spike_vec.clone()
            self._ext_current.zero_()
            self._timestep_count += 1

        if self._learn_enable:
            self._sync_weights_to_adjacency()

        return RunResult(
            total_spikes=total_spikes,
            timesteps=timesteps,
            spike_trains=dict(spike_trains),
            placement=self._compiled.placement,
            backend="gpu_simulator",
        )

    @torch.no_grad()
    def run_with_schedule(self, schedule, rest_steps=0, sync_weights=True):
        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")

        n = self._n
        dev = self.device
        total_timesteps = schedule.shape[0] + rest_steps

        spike_counts = torch.zeros(n, dtype=torch.int32, device=dev)
        total_spikes = 0

        acc_soma = torch.zeros(n, dtype=torch.float32, device=dev)
        acc_dend = [torch.zeros(n, dtype=torch.float32, device=dev) for _ in range(3)]

        for t in range(total_timesteps):
            acc_soma.zero_()
            for d in range(3):
                acc_dend[d].zero_()

            if self._has_delays:
                bucket = self._timestep_count % DELAY_QUEUE_BUCKETS
                acc_soma.add_(self._delay_buf_soma[bucket])
                self._delay_buf_soma[bucket].zero_()
                for d in range(3):
                    acc_dend[d].add_(self._delay_buf_dend[d, bucket])
                    self._delay_buf_dend[d, bucket].zero_()

            if self._prev_spike_vec.any():
                spike_col = self._prev_spike_vec.unsqueeze(1)
                if self._graded_enable:
                    raw = torch.sparse.mm(self._W_soma, spike_col).squeeze(1)
                    acc_soma.add_(torch.div(raw, 128, rounding_mode='trunc'))
                    if self._dendritic_enable:
                        for d in range(3):
                            raw_d = torch.sparse.mm(self._W_dend[d], spike_col).squeeze(1)
                            acc_dend[d].add_(torch.div(raw_d, 128, rounding_mode='trunc'))
                else:
                    binary_vec = (self._prev_spike_vec > 0).float().unsqueeze(1)
                    acc_soma.add_(torch.sparse.mm(self._W_soma, binary_vec).squeeze(1))
                    if self._dendritic_enable:
                        for d in range(3):
                            acc_dend[d].add_(torch.sparse.mm(self._W_dend[d], binary_vec).squeeze(1))

                if self._has_delays:
                    self._deliver_delayed()

            if t < schedule.shape[0]:
                acc_soma.add_(schedule[t].float())

            spike_vec, spike_mask = self._update_neurons_gpu(acc_soma, acc_dend)

            spike_counts.add_(spike_mask.int())

            if self._learn_enable:
                if self._three_factor_enable:
                    self._elig_update_gpu(spike_mask)
                    if self._reward_pending:
                        self._reward_apply_gpu()
                        self._reward_pending = False
                    self._elig_decay_gpu()
                else:
                    self._stdp_update_gpu(spike_mask)

            self._prev_spike_vec = spike_vec.clone()
            self._timestep_count += 1

        if self._learn_enable and sync_weights:
            self._sync_weights_to_adjacency()

        counts_np = spike_counts.cpu().numpy()
        return counts_np, int(spike_counts.sum().item())

    def _deliver_delayed(self):
        if self._graded_enable:
            src_payloads = self._prev_spike_vec[self._delay_src_ids]
        else:
            src_payloads = (self._prev_spike_vec > 0).float()
            src_payloads = src_payloads[self._delay_src_ids]

        active = src_payloads > 0
        if not active.any():
            return

        tgts = self._delay_tgt_ids[active]
        weights = self._delay_weights[active]
        comps = self._delay_comps[active]
        delays = self._delay_values[active]

        if self._graded_enable:
            payloads = src_payloads[active]
            delivered = torch.div(weights * payloads, 128, rounding_mode='trunc')
        else:
            delivered = weights

        buckets = (self._timestep_count + delays) % DELAY_QUEUE_BUCKETS

        soma_mask = comps == 0
        if soma_mask.any():
            self._delay_buf_soma.index_put_(
                (buckets[soma_mask], tgts[soma_mask]),
                delivered[soma_mask], accumulate=True)
        for d in range(3):
            d_mask = comps == (d + 1)
            if d_mask.any():
                self._delay_buf_dend[d].index_put_(
                    (buckets[d_mask], tgts[d_mask]),
                    delivered[d_mask], accumulate=True)

    def _update_neurons_gpu(self, acc_soma, acc_dend):
        n = self._n
        dev = self.device

        total_input = acc_soma.int()
        if self._dendritic_enable:
            dthr = self._dend_threshold
            for d in range(3):
                dval = acc_dend[d].int()
                excess = dval - dthr
                total_input = total_input + torch.where(excess > 0, excess, torch.zeros_like(excess))

        threshold = self._threshold.clone()
        if self._noise_enable:
            threshold = self._apply_noise(threshold)

        potential = self._potential
        refrac = self._refrac
        leak = self._leak
        resting = self._resting

        in_refrac = refrac > 0
        v_plus_input = potential + total_input
        v_minus_leak = v_plus_input - leak
        above_thresh = (~in_refrac) & (v_minus_leak >= threshold)
        above_leak = (~in_refrac) & (~above_thresh) & (v_plus_input > leak)
        below_leak = (~in_refrac) & (~above_thresh) & (~above_leak)

        self._potential = torch.where(in_refrac, resting, self._potential)
        self._refrac = torch.where(in_refrac, refrac - 1, self._refrac)

        excess = v_minus_leak - threshold
        payload = torch.clamp(excess, min=1, max=255)
        self._potential = torch.where(above_thresh, resting, self._potential)
        self._refrac = torch.where(above_thresh, self._refrac_period, self._refrac)
        trace_max_t = torch.full_like(self._trace, TRACE_MAX)
        self._trace = torch.where(above_thresh, trace_max_t, self._trace)
        self._trace2 = torch.where(above_thresh, trace_max_t, self._trace2)

        self._potential = torch.where(above_leak, v_minus_leak, self._potential)

        self._potential = torch.where(below_leak, resting, self._potential)

        non_spiking = ~above_thresh
        self._trace = torch.where(non_spiking,
                                   self._decay_trace_vec(self._trace, self._tau1),
                                   self._trace)
        self._trace2 = torch.where(non_spiking,
                                    self._decay_trace_vec(self._trace2, self._tau2),
                                    self._trace2)

        if self._graded_enable:
            spike_vec = torch.where(above_thresh, payload.float(),
                                    torch.zeros(n, dtype=torch.float32, device=dev))
        else:
            spike_vec = torch.where(above_thresh,
                                    torch.full((n,), 128.0, dtype=torch.float32, device=dev),
                                    torch.zeros(n, dtype=torch.float32, device=dev))

        return spike_vec, above_thresh

    def _decay_trace_vec(self, trace, tau):
        positive = trace > 0
        decay = torch.max(torch.ones_like(trace), trace >> tau)
        new_trace = torch.clamp(trace - decay, min=0)
        return torch.where(positive, new_trace, trace)

    def _apply_noise(self, threshold):
        lfsr = self._lfsr
        bit = lfsr & 1
        lfsr_shifted = lfsr >> 1
        lfsr_xored = lfsr_shifted ^ NOISE_LFSR_TAPS
        self._lfsr = torch.where(bit.bool(), lfsr_xored, lfsr_shifted)

        mantissa = self._noise_config & 0x0F
        exponent = (self._noise_config >> 4) & 0x0F
        has_noise = mantissa > 0

        noise_mask = mantissa << exponent
        noise_val = (self._lfsr & noise_mask) - (noise_mask >> 1)
        return torch.where(has_noise, threshold + noise_val, threshold)

    def _stdp_update_gpu(self, spike_mask):
        if self._learning_rule is not None:
            self._microcode_learn_gpu(spike_mask, three_factor=False)
            return

        if not spike_mask.any() or self._W_soma._nnz() == 0:
            return

        spike_f = spike_mask.float()
        crow = self._soma_crow
        col = self._soma_col
        row_idx = self._soma_row_idx
        val = self._W_soma.values().clone()

        trace_shifted = (self._trace >> LEARN_SHIFT).float()
        zero = torch.zeros_like(val)

        ltd_active = spike_f[col] > 0
        ltd_delta = trace_shifted[row_idx]
        delta_ltd = torch.where(ltd_active, ltd_delta, zero)

        ltp_active = spike_f[row_idx] > 0
        ltp_delta = trace_shifted[col]
        delta_ltp = torch.where(ltp_active, ltp_delta, zero)

        if self._stdp_mask is not None:
            delta_ltd = delta_ltd * self._stdp_mask.float()
            delta_ltp = delta_ltp * self._stdp_mask.float()

        val_new = val - delta_ltd + delta_ltp

        clamped = torch.clamp(val_new, min=WEIGHT_MIN_STDP, max=WEIGHT_MAX_STDP)
        if self._stdp_mask is not None:
            val_new = torch.where(self._stdp_mask, clamped, val)
        else:
            val_new = clamped

        self._W_soma = torch.sparse_csr_tensor(crow, col, val_new, (self._n, self._n))

    def _elig_update_gpu(self, spike_mask):
        if self._learning_rule is not None:
            self._microcode_learn_gpu(spike_mask, three_factor=True)
            return

        if not spike_mask.any() or self._elig_vals is None:
            return

        spike_f = spike_mask.float()
        col = self._soma_col
        row_idx = self._soma_row_idx

        trace_shifted = (self._trace >> LEARN_SHIFT).float()

        ltd_active = spike_f[col] > 0
        ltd_delta = trace_shifted[row_idx]
        self._elig_vals = self._elig_vals - torch.where(ltd_active, ltd_delta,
                                                         torch.zeros_like(self._elig_vals))

        ltp_active = spike_f[row_idx] > 0
        ltp_delta = trace_shifted[col]
        self._elig_vals = self._elig_vals + torch.where(ltp_active, ltp_delta,
                                                         torch.zeros_like(self._elig_vals))

        self._elig_vals = torch.clamp(self._elig_vals, min=-ELIG_MAX, max=ELIG_MAX)

    def _reward_apply_gpu(self):
        if self._reward_value == 0 or self._elig_vals is None:
            return

        delta = torch.div(self._elig_vals * self._reward_value, 1 << REWARD_SHIFT,
                          rounding_mode='trunc')
        val = self._W_soma.values() + delta
        val = torch.clamp(val, min=WEIGHT_MIN_STDP, max=WEIGHT_MAX_STDP)

        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val, (self._n, self._n))
        self._reward_value = 0

    def _elig_decay_gpu(self):
        if self._elig_vals is None:
            return

        abs_vals = self._elig_vals.abs()
        nonzero = abs_vals > 0
        decay = torch.max(torch.ones_like(self._elig_vals),
                          torch.div(abs_vals, 1 << ELIG_DECAY_SHIFT, rounding_mode='trunc'))
        sign = self._elig_vals.sign()

        new_vals = self._elig_vals - sign * decay
        crossed_zero = (self._elig_vals * new_vals) < 0
        new_vals = torch.where(crossed_zero, torch.zeros_like(new_vals), new_vals)
        new_vals = torch.where(nonzero, new_vals, self._elig_vals)

        self._elig_vals = new_vals

    def _microcode_learn_gpu(self, spike_mask, three_factor=False):
        if not spike_mask.any() or self._W_soma._nnz() == 0:
            return

        program = self._learning_rule.get_program()
        spiking_ids = spike_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        trace_cpu = self._trace.cpu().numpy()
        trace2_cpu = self._trace2.cpu().numpy()

        crow_cpu = self._soma_crow.cpu().numpy()
        col_cpu = self._soma_col.cpu().numpy()
        val_cpu = self._W_soma.values().cpu().numpy().copy()

        elig_cpu = self._elig_vals.cpu().numpy().copy() if self._elig_vals is not None else None

        for spike_gid in spiking_ids:
            row_start = crow_cpu[spike_gid]
            row_end = crow_cpu[spike_gid + 1]
            for idx in range(row_start, row_end):
                pass

        adj = self._adjacency
        weights_dict = {}
        for src, targets in adj.items():
            weights_dict[src] = list(targets)

        for spike_gid in spiking_ids:
            spike_gid = int(spike_gid)
            if spike_gid in weights_dict:
                updated = []
                for entry in weights_dict[spike_gid]:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt < self._n:
                        post_t1 = int(trace_cpu[tgt])
                        post_t2 = int(trace2_cpu[tgt])
                        elig_key = self._get_elig_index(spike_gid, tgt)
                        elig = int(elig_cpu[elig_key]) if elig_cpu is not None and elig_key is not None else 0
                        regs = [post_t1, post_t2, w, elig, 0, 0, 0, self._reward_value]
                        result = execute_program(program, LTD_START, LTD_END + 1, regs)
                        if three_factor:
                            if result["elig_written"] and elig_key is not None:
                                elig_cpu[elig_key] = max(-ELIG_MAX, min(ELIG_MAX, result["elig"]))
                        else:
                            if result["weight_written"]:
                                w = max(WEIGHT_MIN_STDP, min(WEIGHT_MAX_STDP, result["weight"]))
                    updated.append((tgt, w, c, *rest))
                weights_dict[spike_gid] = updated

            for src, targets in weights_dict.items():
                if src == spike_gid:
                    continue
                updated = []
                for entry in targets:
                    tgt, w, c = entry[0], entry[1], entry[2]
                    rest = entry[3:]
                    if tgt == spike_gid:
                        pre_t1 = int(trace_cpu[src])
                        pre_t2 = int(trace2_cpu[src])
                        elig_key = self._get_elig_index(src, tgt)
                        elig = int(elig_cpu[elig_key]) if elig_cpu is not None and elig_key is not None else 0
                        regs = [pre_t1, pre_t2, w, elig, 0, 0, 0, self._reward_value]
                        result = execute_program(program, LTP_START, LTP_END + 1, regs)
                        if three_factor:
                            if result["elig_written"] and elig_key is not None:
                                elig_cpu[elig_key] = max(-ELIG_MAX, min(ELIG_MAX, result["elig"]))
                        else:
                            if result["weight_written"]:
                                w = max(WEIGHT_MIN_STDP, min(WEIGHT_MAX_STDP, result["weight"]))
                    updated.append((tgt, w, c, *rest))
                weights_dict[src] = updated

        self._adjacency = weights_dict
        self._rebuild_weight_matrices_from_adjacency()
        if elig_cpu is not None and self._elig_vals is not None:
            self._elig_vals = torch.from_numpy(elig_cpu).to(self.device)

    def _get_elig_index(self, src_gid, tgt_gid):
        if self._soma_crow is None:
            return None
        crow_cpu = self._soma_crow.cpu()
        col_cpu = self._soma_col.cpu()
        row_start = int(crow_cpu[tgt_gid])
        row_end = int(crow_cpu[tgt_gid + 1])
        for idx in range(row_start, row_end):
            if int(col_cpu[idx]) == src_gid:
                return idx
        return None

    def _rebuild_weight_matrices_from_adjacency(self):
        self._build_weight_matrices(self._n)

    def _sync_weights_to_adjacency(self):
        if self._W_soma is None or self._W_soma._nnz() == 0:
            return

        val_cpu = self._W_soma.values().cpu().numpy()
        crow_cpu = self._soma_crow.cpu().numpy()
        col_cpu = self._soma_col.cpu().numpy()

        weight_updates = {}
        for tgt in range(self._n):
            start = int(crow_cpu[tgt])
            end = int(crow_cpu[tgt + 1])
            for idx in range(start, end):
                src = int(col_cpu[idx])
                weight_updates[(src, tgt)] = int(round(val_cpu[idx]))

        for src, targets in self._adjacency.items():
            updated = []
            for entry in targets:
                tgt, w, c = entry[0], entry[1], entry[2]
                rest = entry[3:]
                delay = rest[0] if rest else 0
                if delay == 0 and c == 0:
                    key = (src, tgt)
                    if key in weight_updates:
                        w = weight_updates[key]
                updated.append((tgt, w, c, *rest))
            self._adjacency[src] = updated

    def set_learning(self, learn=False, graded=False, dendritic=False,
                     async_mode=False, three_factor=False, noise=False):
        self._learn_enable = learn
        self._graded_enable = graded
        self._dendritic_enable = dendritic
        self._three_factor_enable = three_factor
        self._noise_enable = noise
        if async_mode:
            raise NeurocoreError("Async mode not supported on GPU simulator.")
        if three_factor and not learn:
            self._learn_enable = True

    def set_stdp_mask(self, learnable_source_gids):
        if self._W_soma is None or self._W_soma._nnz() == 0:
            return
        src_set = set(learnable_source_gids)
        col = self._soma_col.cpu().numpy()
        mask = torch.tensor([int(c) in src_set for c in col],
                            dtype=torch.bool, device=self.device)
        self._stdp_mask = mask

    def reset_state(self):
        self._potential.zero_()
        self._refrac.zero_()
        self._trace.zero_()
        self._trace2.zero_()
        self._ext_current.zero_()
        self._prev_spike_vec.zero_()
        if self._has_delays and self._delay_buf_soma is not None:
            self._delay_buf_soma.zero_()
            self._delay_buf_dend.zero_()

    @torch.no_grad()
    def randomize_learnable_weights(self, low=10.0, high=400.0, seed=42):
        if self._stdp_mask is None or self._W_soma._nnz() == 0:
            return
        nnz = int(self._W_soma._nnz())
        rng = np.random.RandomState(seed)
        rand_vals = torch.from_numpy(
            rng.uniform(low, high, size=nnz).astype(np.float32)
        ).to(self.device)
        val = self._W_soma.values().clone()
        val_new = torch.where(self._stdp_mask, rand_vals, val)
        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val_new, (self._n, self._n))

    @torch.no_grad()
    def competitive_update(self, winner_gids, pixel_intensity, pixel_gids,
                           eta_ltp=0.05, eta_ltd=0.01, w_max=2000.0):
        if self._stdp_mask is None or self._W_soma._nnz() == 0:
            return

        dev = self.device
        val = self._W_soma.values()
        col = self._soma_col
        row_idx = self._soma_row_idx.long()
        learnable = self._stdp_mask

        pixel_lookup = torch.zeros(self._n, dtype=torch.float32, device=dev)
        pixel_lookup[pixel_gids] = pixel_intensity
        x_pre = pixel_lookup[col]

        winner_full = torch.zeros(self._n, dtype=torch.bool, device=dev)
        winner_full[winner_gids] = True
        is_winner = winner_full[row_idx]
        winner_mask = learnable & is_winner

        w_per_tgt = torch.zeros(self._n, dtype=torch.float32, device=dev)
        w_per_tgt.scatter_add_(0, row_idx,
                               torch.where(winner_mask, val.clamp(min=0), torch.zeros_like(val)))
        x_per_tgt = torch.zeros(self._n, dtype=torch.float32, device=dev)
        x_per_tgt.scatter_add_(0, row_idx,
                               torch.where(winner_mask, x_pre, torch.zeros_like(x_pre)))
        scale = torch.where(x_per_tgt > 1e-6, w_per_tgt / x_per_tgt,
                            torch.ones(self._n, dtype=torch.float32, device=dev))
        entry_scale = scale[row_idx]

        target = x_pre * entry_scale
        dw_winner = eta_ltp * (target - val)

        active = x_pre > 0.01
        loser_mask = learnable & (~is_winner) & active
        dw_loser = eta_ltd * val * x_pre

        val_new = val.clone()
        val_new = torch.where(winner_mask, val + dw_winner, val_new)
        val_new = torch.where(loser_mask, val - dw_loser, val_new)

        val_clamped = torch.clamp(val_new, min=0.0, max=w_max)
        val_final = torch.where(learnable, val_clamped, val)

        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val_final, (self._n, self._n))

    @torch.no_grad()
    def normalize_learnable_weights(self, target_sum, target_gids=None):
        if self._stdp_mask is None or self._W_soma._nnz() == 0:
            return

        dev = self.device
        val = self._W_soma.values().clone()
        row_idx = self._soma_row_idx.long()
        learnable = self._stdp_mask

        if target_gids is not None:
            tgt_mask = torch.zeros(self._n, dtype=torch.bool, device=dev)
            tgt_mask[target_gids] = True
            entry_mask = tgt_mask[row_idx] & learnable
        else:
            entry_mask = learnable

        masked_vals = torch.where(entry_mask, val.clamp(min=0), torch.zeros_like(val))
        per_tgt_sum = torch.zeros(self._n, dtype=torch.float32, device=dev)
        per_tgt_sum.scatter_add_(0, row_idx, masked_vals)

        scale = torch.where(per_tgt_sum > 0,
                            float(target_sum) / per_tgt_sum,
                            torch.ones(self._n, dtype=torch.float32, device=dev))
        entry_scale = scale[row_idx]

        val_scaled = torch.where(entry_mask, val * entry_scale, val)
        val_final = torch.where(learnable,
                                val_scaled.clamp(min=0, max=float(WEIGHT_MAX_STDP)),
                                val)

        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val_final, (self._n, self._n))

    def status(self):
        return {"state": 0, "timestep_count": self._timestep_count}

    def close(self):
        self._W_soma = None
        self._W_dend = [None] * 3
        self._potential = None
        self._delay_buf_soma = None
        self._delay_buf_dend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def get_weights(self):
        if self._learn_enable:
            self._sync_weights_to_adjacency()
        return dict(self._adjacency) if self._adjacency else {}

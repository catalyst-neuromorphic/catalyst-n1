"""GPU-accelerated LIF simulator using PyTorch sparse tensors.

Matches the cycle-accurate behavior of simulator.py but runs on CUDA GPU,
achieving 100-1000x speedup for large networks (4K-32K neurons).

All neuron state stored as dense int32 tensors on GPU.
Connectivity stored as sparse CSR float32 matrices: W @ spike_vec = current.
"""

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
    """GPU-accelerated LIF simulator using PyTorch CUDA tensors."""

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                # Prefer GPU 1 (20GB 3080) if available, else GPU 0
                device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device
        self._compiler = Compiler()
        self._compiled = None
        self._n = 0
        self._timestep_count = 0

        # Neuron state tensors (set by deploy)
        self._potential = None
        self._refrac = None
        self._trace = None
        self._trace2 = None
        self._ext_current = None

        # Per-neuron parameter tensors
        self._threshold = None
        self._leak = None
        self._resting = None
        self._refrac_period = None
        self._dend_threshold = None
        self._noise_config = None
        self._tau1 = None
        self._tau2 = None
        self._lfsr = None

        # Sparse weight matrices (CSR, float32, shape (N, N))
        # Convention: W[target, source] so W @ spike_vec = accumulated current
        self._W_soma = None        # compartment 0, delay=0
        self._W_dend = [None] * 3  # compartments 1-3, delay=0

        # Delay structures
        self._has_delays = False
        self._delay_buf_soma = None   # (64, N) ring buffer
        self._delay_buf_dend = None   # (3, 64, N) ring buffer
        self._delay_src_ids = None    # (num_delayed,) source neuron indices
        self._delay_tgt_ids = None    # (num_delayed,) target neuron indices
        self._delay_weights = None    # (num_delayed,) weight values
        self._delay_comps = None      # (num_delayed,) compartment IDs
        self._delay_values = None     # (num_delayed,) delay tick values

        # Spike vectors
        self._prev_spike_vec = None   # (N,) float32 - payload from previous timestep
        self._spike_mask = None       # (N,) bool - who spiked this timestep

        # Config flags
        self._learn_enable = False
        self._graded_enable = False
        self._dendritic_enable = False
        self._three_factor_enable = False
        self._noise_enable = False

        # Learning state
        self._learning_rule = None
        self._elig_crow = None    # CSR row pointers for eligibility
        self._elig_col = None     # CSR column indices
        self._elig_vals = None    # eligibility values (same sparsity as W_soma)
        self._reward_value = 0
        self._reward_pending = False

        # STDP mask: bool tensor over CSR values (True = learnable)
        self._stdp_mask = None    # None means all connections learnable

        # CSR structure cache for STDP (avoids recomputing each timestep)
        self._soma_crow = None
        self._soma_col = None
        self._soma_row_idx = None  # expanded row indices (nnz,)

        # CPU-side adjacency for microcode fallback and weight export
        self._adjacency = None

    def deploy(self, network_or_compiled):
        """Compile (if needed) and initialize GPU state."""
        if isinstance(network_or_compiled, Network):
            self._compiled = self._compiler.compile(network_or_compiled)
        elif isinstance(network_or_compiled, CompiledNetwork):
            self._compiled = network_or_compiled
        else:
            raise TypeError(f"Expected Network or CompiledNetwork, got {type(network_or_compiled)}")

        n = self._compiled.placement.total_neurons
        self._n = n
        dev = self.device

        # Initialize neuron state tensors
        self._potential = torch.zeros(n, dtype=torch.int32, device=dev)
        self._refrac = torch.zeros(n, dtype=torch.int32, device=dev)
        self._trace = torch.zeros(n, dtype=torch.int32, device=dev)
        self._trace2 = torch.zeros(n, dtype=torch.int32, device=dev)
        self._ext_current = torch.zeros(n, dtype=torch.int32, device=dev)

        # Per-neuron parameters
        self._threshold = torch.full((n,), DEFAULT_THRESHOLD, dtype=torch.int32, device=dev)
        self._leak = torch.full((n,), DEFAULT_LEAK, dtype=torch.int32, device=dev)
        self._resting = torch.full((n,), DEFAULT_RESTING, dtype=torch.int32, device=dev)
        self._refrac_period = torch.full((n,), DEFAULT_REFRAC, dtype=torch.int32, device=dev)
        self._dend_threshold = torch.full((n,), DEFAULT_DEND_THRESHOLD, dtype=torch.int32, device=dev)
        self._noise_config = torch.full((n,), DEFAULT_NOISE_CONFIG, dtype=torch.int32, device=dev)
        self._tau1 = torch.full((n,), DEFAULT_TAU1, dtype=torch.int32, device=dev)
        self._tau2 = torch.full((n,), DEFAULT_TAU2, dtype=torch.int32, device=dev)

        # LFSR seeds: advance per-neuron for unique starting states
        lfsr_seeds = np.zeros(n, dtype=np.int32)
        lfsr = NOISE_LFSR_SEED
        for gid in range(n):
            lfsr_seeds[gid] = lfsr
            bit = lfsr & 1
            lfsr >>= 1
            if bit:
                lfsr ^= NOISE_LFSR_TAPS
        self._lfsr = torch.from_numpy(lfsr_seeds).to(dev)

        # Apply per-neuron parameter overrides
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

        # Build sparse weight matrices from adjacency
        self._adjacency = dict(self._compiled.adjacency)
        self._build_weight_matrices(n)

        # Apply learn config
        cfg = self._compiled.learn_config
        self._learn_enable = cfg.get("learn_enable", False)
        self._graded_enable = cfg.get("graded_enable", False)
        self._dendritic_enable = cfg.get("dendritic_enable", False)
        self._noise_enable = cfg.get("noise_enable", False)

        # P19 learning rule
        self._learning_rule = self._compiled.learning_rule

        # Spike vectors
        self._prev_spike_vec = torch.zeros(n, dtype=torch.float32, device=dev)

        # Learning state
        self._reward_value = 0
        self._reward_pending = False

        # Initialize eligibility with same sparsity as W_soma
        if self._W_soma is not None and self._W_soma._nnz() > 0:
            self._elig_crow = self._soma_crow
            self._elig_col = self._soma_col
            self._elig_vals = torch.zeros(self._W_soma._nnz(), dtype=torch.float32, device=dev)
        else:
            self._elig_vals = None

        self._timestep_count = 0

    def _build_weight_matrices(self, n):
        """Build sparse CSR weight matrices from adjacency dict."""
        dev = self.device

        # Collect COO triplets per compartment, split by delay
        rows_imm = [[] for _ in range(4)]   # immediate (delay=0)
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

        # Build CSR for each compartment (immediate delivery)
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
            # Coalesce to sum duplicates (same src->tgt with different entries)
            coo = coo.coalesce()
            return coo.to_sparse_csr().to(dev)

        self._W_soma = _build_csr(rows_imm[0], cols_imm[0], vals_imm[0])
        for d in range(3):
            self._W_dend[d] = _build_csr(rows_imm[d + 1], cols_imm[d + 1], vals_imm[d + 1])

        # Cache CSR structure for STDP
        self._soma_crow = self._W_soma.crow_indices()
        self._soma_col = self._W_soma.col_indices()
        if self._W_soma._nnz() > 0:
            self._soma_row_idx = torch.repeat_interleave(
                torch.arange(n, device=dev),
                self._soma_crow[1:] - self._soma_crow[:-1]
            )
        else:
            self._soma_row_idx = torch.tensor([], dtype=torch.int64, device=dev)

        # Build delay structures
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
        """Set external stimulus current for specified neurons."""
        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")
        resolved = self._resolve_targets(target)
        for core, neuron in resolved:
            gid = core * NEURONS_PER_CORE + neuron
            if gid < self._n:
                self._ext_current[gid] = current

    def reward(self, value):
        """Set reward signal for 3-factor learning."""
        self._reward_value = int(value)
        self._reward_pending = True

    def run(self, timesteps):
        """Execute timesteps on GPU and return RunResult."""
        from .result import RunResult

        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")

        if getattr(self, '_async_enable', False):
            raise NeurocoreError("Async mode not supported on GPU simulator. Use sync mode.")

        return self._run_sync(timesteps)

    @torch.no_grad()
    def _run_sync(self, timesteps):
        """Synchronous GPU execution: all neurons updated every timestep."""
        from .result import RunResult

        n = self._n
        dev = self.device
        spike_trains = defaultdict(list)
        total_spikes = 0

        # Pre-allocate accumulators
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
                spike_col = self._prev_spike_vec.unsqueeze(1)  # (N, 1)

                if self._graded_enable:
                    # Graded: result = (W @ payload_vec) / 128
                    raw = torch.sparse.mm(self._W_soma, spike_col).squeeze(1)
                    acc_soma.add_(torch.div(raw, 128, rounding_mode='trunc'))
                    if self._dendritic_enable:
                        for d in range(3):
                            raw_d = torch.sparse.mm(self._W_dend[d], spike_col).squeeze(1)
                            acc_dend[d].add_(torch.div(raw_d, 128, rounding_mode='trunc'))
                else:
                    # Binary: result = W @ spike_binary (spike_vec has value 128 for binary)
                    # But we stored actual weights in W, not weight*128.
                    # CPU sim uses: delivered = weight (when not graded)
                    # Our spike_vec has payload=128 for non-graded. We need:
                    # delivered = weight, so we need W @ binary_spike_vec
                    binary_vec = (self._prev_spike_vec > 0).float().unsqueeze(1)
                    acc_soma.add_(torch.sparse.mm(self._W_soma, binary_vec).squeeze(1))
                    if self._dendritic_enable:
                        for d in range(3):
                            acc_dend[d].add_(torch.sparse.mm(self._W_dend[d], binary_vec).squeeze(1))

                # Delayed connections
                if self._has_delays:
                    self._deliver_delayed()

            # Add external current
            acc_soma.add_(self._ext_current.float())

            spike_vec, spike_mask = self._update_neurons_gpu(acc_soma, acc_dend)

            # Record spikes (small GPU->CPU transfer)
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

        # Update adjacency from GPU weights (for weight export / subsequent runs)
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
        """Run timesteps with pre-computed per-timestep stimulus, returning spike counts.

        This is much faster than calling inject()+run(1) in a Python loop because:
        - No Python→GPU per-timestep injection overhead
        - Spike counts accumulated on GPU (no per-timestep CPU transfer)

        Args:
            schedule: torch.Tensor of shape (T, N), int32, on self.device.
                schedule[t, gid] = external current for neuron gid at timestep t.
            rest_steps: additional timesteps to run after schedule with no stimulus.
            sync_weights: if True (default), sync GPU weights back to adjacency dict
                after run. Set False during training loops for performance, then
                call _sync_weights_to_adjacency() manually when needed.

        Returns:
            (spike_counts, total_spikes) where spike_counts is a (N,) int32 numpy
            array of per-neuron spike counts across all timesteps.
        """
        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")

        n = self._n
        dev = self.device
        total_timesteps = schedule.shape[0] + rest_steps

        # Accumulate spike counts on GPU — no per-timestep CPU transfer
        spike_counts = torch.zeros(n, dtype=torch.int32, device=dev)
        total_spikes = 0

        # Pre-allocate accumulators
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

            # Spike delivery
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

            # Add scheduled stimulus (or zero during rest)
            if t < schedule.shape[0]:
                acc_soma.add_(schedule[t].float())

            # Neuron update
            spike_vec, spike_mask = self._update_neurons_gpu(acc_soma, acc_dend)

            # Accumulate counts on GPU (no CPU transfer!)
            spike_counts.add_(spike_mask.int())

            # STDP learning
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

        # Sync weights after learning (can be deferred for performance)
        if self._learn_enable and sync_weights:
            self._sync_weights_to_adjacency()

        counts_np = spike_counts.cpu().numpy()
        return counts_np, int(spike_counts.sum().item())

    def _deliver_delayed(self):
        """Scatter delayed spike currents into future ring buffer buckets."""
        # Find which delayed synapses have spiking sources
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

        # Scatter by compartment
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
        """Vectorized LIF update for all neurons simultaneously.

        Returns:
            spike_vec: (N,) float32 - payload values for spiking neurons, 0 elsewhere
            spike_mask: (N,) bool - which neurons spiked
        """
        n = self._n
        dev = self.device

        # Dendritic compartment thresholding
        total_input = acc_soma.int()
        if self._dendritic_enable:
            dthr = self._dend_threshold
            for d in range(3):
                dval = acc_dend[d].int()
                excess = dval - dthr
                total_input = total_input + torch.where(excess > 0, excess, torch.zeros_like(excess))

        # P14 Noise: vectorized LFSR advance + threshold perturbation
        threshold = self._threshold.clone()
        if self._noise_enable:
            threshold = self._apply_noise(threshold)

        potential = self._potential
        refrac = self._refrac
        leak = self._leak
        resting = self._resting

        # Compute conditions for all neurons simultaneously
        in_refrac = refrac > 0
        v_plus_input = potential + total_input
        v_minus_leak = v_plus_input - leak
        above_thresh = (~in_refrac) & (v_minus_leak >= threshold)
        above_leak = (~in_refrac) & (~above_thresh) & (v_plus_input > leak)
        below_leak = (~in_refrac) & (~above_thresh) & (~above_leak)

        # Branch 1: Refractory — reset potential, decrement counter, decay traces
        self._potential = torch.where(in_refrac, resting, self._potential)
        self._refrac = torch.where(in_refrac, refrac - 1, self._refrac)

        # Spike: reset, enter refractory, set traces to max
        excess = v_minus_leak - threshold
        payload = torch.clamp(excess, min=1, max=255)
        self._potential = torch.where(above_thresh, resting, self._potential)
        self._refrac = torch.where(above_thresh, self._refrac_period, self._refrac)
        trace_max_t = torch.full_like(self._trace, TRACE_MAX)
        self._trace = torch.where(above_thresh, trace_max_t, self._trace)
        self._trace2 = torch.where(above_thresh, trace_max_t, self._trace2)

        # Branch 3: Integrate — accumulate input
        self._potential = torch.where(above_leak, v_minus_leak, self._potential)

        # Branch 4: Below leak — reset to resting
        self._potential = torch.where(below_leak, resting, self._potential)

        # Trace decay for non-spiking neurons (P15 dual traces)
        non_spiking = ~above_thresh
        self._trace = torch.where(non_spiking,
                                   self._decay_trace_vec(self._trace, self._tau1),
                                   self._trace)
        self._trace2 = torch.where(non_spiking,
                                    self._decay_trace_vec(self._trace2, self._tau2),
                                    self._trace2)

        # Build spike vector
        if self._graded_enable:
            spike_vec = torch.where(above_thresh, payload.float(),
                                    torch.zeros(n, dtype=torch.float32, device=dev))
        else:
            spike_vec = torch.where(above_thresh,
                                    torch.full((n,), 128.0, dtype=torch.float32, device=dev),
                                    torch.zeros(n, dtype=torch.float32, device=dev))

        return spike_vec, above_thresh

    def _decay_trace_vec(self, trace, tau):
        """Vectorized P15 exponential trace decay with min-step-1 guarantee."""
        positive = trace > 0
        decay = torch.max(torch.ones_like(trace), trace >> tau)
        new_trace = torch.clamp(trace - decay, min=0)
        return torch.where(positive, new_trace, trace)

    def _apply_noise(self, threshold):
        """Vectorized P14 LFSR advance and threshold perturbation."""
        # Advance Galois LFSR: bit = lfsr & 1; lfsr >>= 1; if bit: lfsr ^= taps
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
        """Vectorized 2-factor STDP using CSR structure."""
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

        # LTD: source spiked → weight -= post_trace[target] >> 3
        ltd_active = spike_f[col] > 0
        ltd_delta = trace_shifted[row_idx]
        delta_ltd = torch.where(ltd_active, ltd_delta, zero)

        # LTP: target spiked → weight += pre_trace[source] >> 3
        ltp_active = spike_f[row_idx] > 0
        ltp_delta = trace_shifted[col]
        delta_ltp = torch.where(ltp_active, ltp_delta, zero)

        # Apply mask: only update learnable connections
        if self._stdp_mask is not None:
            delta_ltd = delta_ltd * self._stdp_mask.float()
            delta_ltp = delta_ltp * self._stdp_mask.float()

        val_new = val - delta_ltd + delta_ltp

        # Clamp only learnable connections (preserve fixed inhibitory weights)
        clamped = torch.clamp(val_new, min=WEIGHT_MIN_STDP, max=WEIGHT_MAX_STDP)
        if self._stdp_mask is not None:
            val_new = torch.where(self._stdp_mask, clamped, val)
        else:
            val_new = clamped

        # Rebuild CSR (structure unchanged, only values updated)
        self._W_soma = torch.sparse_csr_tensor(crow, col, val_new, (self._n, self._n))

    def _elig_update_gpu(self, spike_mask):
        """3-factor: STDP correlation → eligibility accumulation."""
        if self._learning_rule is not None:
            self._microcode_learn_gpu(spike_mask, three_factor=True)
            return

        if not spike_mask.any() or self._elig_vals is None:
            return

        spike_f = spike_mask.float()
        col = self._soma_col
        row_idx = self._soma_row_idx

        trace_shifted = (self._trace >> LEARN_SHIFT).float()

        # LTD: source spiked → elig -= post_trace[target] >> 3
        ltd_active = spike_f[col] > 0
        ltd_delta = trace_shifted[row_idx]
        self._elig_vals = self._elig_vals - torch.where(ltd_active, ltd_delta,
                                                         torch.zeros_like(self._elig_vals))

        # LTP: target spiked → elig += pre_trace[source] >> 3
        ltp_active = spike_f[row_idx] > 0
        ltp_delta = trace_shifted[col]
        self._elig_vals = self._elig_vals + torch.where(ltp_active, ltp_delta,
                                                         torch.zeros_like(self._elig_vals))

        # Clamp eligibility
        self._elig_vals = torch.clamp(self._elig_vals, min=-ELIG_MAX, max=ELIG_MAX)

    def _reward_apply_gpu(self):
        """Apply reward to weights via eligibility: W += (elig * reward) >> REWARD_SHIFT."""
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
        """Exponential decay of eligibility: elig -= sign(elig) * max(1, |elig| >> 3)."""
        if self._elig_vals is None:
            return

        abs_vals = self._elig_vals.abs()
        nonzero = abs_vals > 0
        decay = torch.max(torch.ones_like(self._elig_vals),
                          torch.div(abs_vals, 1 << ELIG_DECAY_SHIFT, rounding_mode='trunc'))
        sign = self._elig_vals.sign()

        new_vals = self._elig_vals - sign * decay
        # Zero out values that crossed zero
        crossed_zero = (self._elig_vals * new_vals) < 0
        new_vals = torch.where(crossed_zero, torch.zeros_like(new_vals), new_vals)
        # Also zero out values where decay >= |val|
        new_vals = torch.where(nonzero, new_vals, self._elig_vals)

        self._elig_vals = new_vals

    def _microcode_learn_gpu(self, spike_mask, three_factor=False):
        """P19 microcode learning: CPU fallback for custom rules.

        Transfers spiking neuron data to CPU, runs interpreter, transfers back.
        """
        if not spike_mask.any() or self._W_soma._nnz() == 0:
            return

        program = self._learning_rule.get_program()
        spiking_ids = spike_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        trace_cpu = self._trace.cpu().numpy()
        trace2_cpu = self._trace2.cpu().numpy()

        # Pull weight values to CPU
        crow_cpu = self._soma_crow.cpu().numpy()
        col_cpu = self._soma_col.cpu().numpy()
        val_cpu = self._W_soma.values().cpu().numpy().copy()

        # Pull eligibility if 3-factor
        elig_cpu = self._elig_vals.cpu().numpy().copy() if self._elig_vals is not None else None

        for spike_gid in spiking_ids:
            row_start = crow_cpu[spike_gid]
            row_end = crow_cpu[spike_gid + 1]
            for idx in range(row_start, row_end):
                pass

        # Full adjacency iteration for microcode learning
        adj = self._adjacency
        weights_dict = {}
        # Build mutable weight dict from adjacency
        for src, targets in adj.items():
            weights_dict[src] = list(targets)

        for spike_gid in spiking_ids:
            spike_gid = int(spike_gid)
            # LTD: pre spiked
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

            # LTP: post spiked
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

        # Sync back to GPU
        self._adjacency = weights_dict
        self._rebuild_weight_matrices_from_adjacency()
        if elig_cpu is not None and self._elig_vals is not None:
            self._elig_vals = torch.from_numpy(elig_cpu).to(self.device)

    def _get_elig_index(self, src_gid, tgt_gid):
        """Find the CSR value index for synapse (src_gid, tgt_gid) in W_soma.

        W_soma is (target, source) CSR, so row=tgt_gid, and we search
        for col=src_gid within that row.
        """
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
        """Rebuild GPU weight matrices from CPU adjacency (after microcode update)."""
        self._build_weight_matrices(self._n)

    def _sync_weights_to_adjacency(self):
        """Sync GPU weight matrix values back to CPU adjacency dict.

        Only updates weights for compartment-0 immediate connections (the learnable ones).
        """
        if self._W_soma is None or self._W_soma._nnz() == 0:
            return

        val_cpu = self._W_soma.values().cpu().numpy()
        crow_cpu = self._soma_crow.cpu().numpy()
        col_cpu = self._soma_col.cpu().numpy()

        # Build a lookup: (tgt, src) -> new_weight
        weight_updates = {}
        for tgt in range(self._n):
            start = int(crow_cpu[tgt])
            end = int(crow_cpu[tgt + 1])
            for idx in range(start, end):
                src = int(col_cpu[idx])
                weight_updates[(src, tgt)] = int(round(val_cpu[idx]))

        # Update adjacency
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
        """Configure feature flags."""
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
        """Mark which connections are STDP-learnable by source neuron ID.

        Only connections FROM neurons in learnable_source_gids will be updated
        by STDP. All other connections remain fixed. This is essential for
        networks where only some connections should learn (e.g., input→excitatory
        in Diehl & Cook architecture).

        Args:
            learnable_source_gids: set or list of global neuron IDs whose
                outgoing connections should be STDP-learnable.
        """
        if self._W_soma is None or self._W_soma._nnz() == 0:
            return
        src_set = set(learnable_source_gids)
        col = self._soma_col.cpu().numpy()
        mask = torch.tensor([int(c) in src_set for c in col],
                            dtype=torch.bool, device=self.device)
        self._stdp_mask = mask

    def reset_state(self):
        """Reset all neuron state to initial values. Call between training images."""
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
        """Randomize STDP-masked connection weights on GPU.

        Useful for breaking symmetry before competitive learning.
        Only modifies entries where self._stdp_mask is True.
        """
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
        """GPU-native competitive weight update on W_soma CSR values.

        Uses scale-invariant EMA: the target is scaled to match each winner
        neuron's current weight magnitude, so eta truly represents the
        fractional movement toward the input pattern.

        Winner: w += eta_ltp * (x_pre * scale_i - w)
            where scale_i = sum(w_i) / sum(x_pre_i) for neuron i.
        Loser: w -= eta_ltd * w * x_pre
            Anti-Hebbian for active pixels.

        Args:
            winner_gids: (K,) int64 tensor of winner GIDs on GPU
            pixel_intensity: (n_input,) float32 tensor of pixel values [0,1] on GPU
            pixel_gids: (n_input,) int64 tensor of input neuron GIDs on GPU
            eta_ltp: learning rate for winners (default: 0.05)
            eta_ltd: learning rate for losers (default: 0.01)
            w_max: clamp ceiling for final weights
        """
        if self._stdp_mask is None or self._W_soma._nnz() == 0:
            return

        dev = self.device
        val = self._W_soma.values()
        col = self._soma_col
        row_idx = self._soma_row_idx.long()
        learnable = self._stdp_mask

        # Pixel intensity lookup: only input neuron GIDs have nonzero values
        pixel_lookup = torch.zeros(self._n, dtype=torch.float32, device=dev)
        pixel_lookup[pixel_gids] = pixel_intensity
        x_pre = pixel_lookup[col]  # (nnz,) pixel intensity per source

        # Winner lookup
        winner_full = torch.zeros(self._n, dtype=torch.bool, device=dev)
        winner_full[winner_gids] = True
        is_winner = winner_full[row_idx]  # (nnz,)
        winner_mask = learnable & is_winner

        # Compute per-neuron adaptive scale so target has same magnitude as
        # current weights (scale = w_sum / x_sum per winner neuron)
        w_per_tgt = torch.zeros(self._n, dtype=torch.float32, device=dev)
        w_per_tgt.scatter_add_(0, row_idx,
                               torch.where(winner_mask, val.clamp(min=0), torch.zeros_like(val)))
        x_per_tgt = torch.zeros(self._n, dtype=torch.float32, device=dev)
        x_per_tgt.scatter_add_(0, row_idx,
                               torch.where(winner_mask, x_pre, torch.zeros_like(x_pre)))
        scale = torch.where(x_per_tgt > 1e-6, w_per_tgt / x_per_tgt,
                            torch.ones(self._n, dtype=torch.float32, device=dev))
        entry_scale = scale[row_idx]  # (nnz,) per-entry scale

        # Winner: scale-invariant EMA toward input pattern
        target = x_pre * entry_scale
        dw_winner = eta_ltp * (target - val)

        # Loser: anti-Hebbian for active pixels
        active = x_pre > 0.01
        loser_mask = learnable & (~is_winner) & active
        dw_loser = eta_ltd * val * x_pre

        val_new = val.clone()
        val_new = torch.where(winner_mask, val + dw_winner, val_new)
        val_new = torch.where(loser_mask, val - dw_loser, val_new)

        # Clamp learnable only, preserve fixed weights
        val_clamped = torch.clamp(val_new, min=0.0, max=w_max)
        val_final = torch.where(learnable, val_clamped, val)

        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val_final, (self._n, self._n))

    @torch.no_grad()
    def normalize_learnable_weights(self, target_sum, target_gids=None):
        """GPU-native per-target weight normalization for learnable connections.

        Scales learnable incoming weights for each target neuron so their sum
        equals target_sum. Non-learnable weights are preserved.

        Args:
            target_sum: desired sum of learnable weights per target neuron
            target_gids: (M,) int64 tensor of target GIDs on GPU, or None for all
        """
        if self._stdp_mask is None or self._W_soma._nnz() == 0:
            return

        dev = self.device
        val = self._W_soma.values().clone()
        row_idx = self._soma_row_idx.long()
        learnable = self._stdp_mask

        # Entry mask: learnable connections to specified targets
        if target_gids is not None:
            tgt_mask = torch.zeros(self._n, dtype=torch.bool, device=dev)
            tgt_mask[target_gids] = True
            entry_mask = tgt_mask[row_idx] & learnable
        else:
            entry_mask = learnable

        # Sum positive weights per target (only masked entries)
        masked_vals = torch.where(entry_mask, val.clamp(min=0), torch.zeros_like(val))
        per_tgt_sum = torch.zeros(self._n, dtype=torch.float32, device=dev)
        per_tgt_sum.scatter_add_(0, row_idx, masked_vals)

        # Per-target scale factor
        scale = torch.where(per_tgt_sum > 0,
                            float(target_sum) / per_tgt_sum,
                            torch.ones(self._n, dtype=torch.float32, device=dev))
        entry_scale = scale[row_idx]

        # Apply scale only to masked entries
        val_scaled = torch.where(entry_mask, val * entry_scale, val)
        val_final = torch.where(learnable,
                                val_scaled.clamp(min=0, max=float(WEIGHT_MAX_STDP)),
                                val)

        self._W_soma = torch.sparse_csr_tensor(
            self._soma_crow, self._soma_col, val_final, (self._n, self._n))

    def status(self):
        return {"state": 0, "timestep_count": self._timestep_count}

    def close(self):
        """Release GPU memory."""
        self._W_soma = None
        self._W_dend = [None] * 3
        self._potential = None
        self._delay_buf_soma = None
        self._delay_buf_dend = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_targets(self, target):
        """Convert Population/PopulationSlice to [(core, neuron)] pairs."""
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
        """Export current weights as adjacency dict (CPU)."""
        if self._learn_enable:
            self._sync_weights_to_adjacency()
        return dict(self._adjacency) if self._adjacency else {}

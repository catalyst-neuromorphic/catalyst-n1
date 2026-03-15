"""Hardware constants and default parameters for the neuromorphic chip.

P20 update: Full Loihi parity — noise, dual traces, delays, synapse formats,
microcode learning, hierarchical routing.
"""

# Hardware limits (from neuromorphic_top.v, scalable_core_v2.v)
MAX_CORES = 128
NEURONS_PER_CORE = 1024       # P13a: was 256
NEURON_BITS = 10              # P13a: was 8 (log2(1024))
DATA_WIDTH = 16
WEIGHT_MIN = -32768
WEIGHT_MAX = 32767
COMPARTMENTS = 4              # 0=soma, 1-3=dendrites

# CSR connectivity pool (P13a: replaces fixed 32-slot fanout)
POOL_DEPTH = 32768            # shared connection pool entries per core
POOL_ADDR_BITS = 15           # log2(POOL_DEPTH)
INDEX_WIDTH = 25              # base_addr(15) + count(10)
COUNT_BITS = 10               # max 1024 connections per neuron

# Multicast inter-core routing (P13b: was 1 route per source)
ROUTE_FANOUT = 8              # max inter-core route slots per source neuron
ROUTE_SLOT_BITS = 3           # log2(ROUTE_FANOUT)

# Reverse connection table for STDP (P13a: updated for CSR)
REV_FANIN = 32                # max tracked incoming connections per target
REV_SLOT_BITS = 5

# Legacy constant (kept for backward compat, no longer enforced per-neuron)
MAX_FANOUT = 32

# Default neuron parameters (from scalable_core_v2.v)
DEFAULT_THRESHOLD = 1000
DEFAULT_LEAK = 3
DEFAULT_RESTING = 0
DEFAULT_REFRAC = 3
DEFAULT_DEND_THRESHOLD = 0

# Parameter IDs (from host.py CMD_PROG_NEURON)
PARAM_THRESHOLD = 0
PARAM_LEAK = 1
PARAM_RESTING = 2
PARAM_REFRAC = 3
PARAM_DEND_THRESHOLD = 4
PARAM_NOISE_CFG = 5          # P14: noise config {exponent[7:4], mantissa[3:0]}
PARAM_TAU1 = 6               # P15: trace1 decay shift
PARAM_TAU2 = 7               # P15: trace2 decay shift

# STDP constants (from scalable_core_v2.v)
TRACE_MAX = 100
TRACE_DECAY = 3
LEARN_SHIFT = 3
GRADE_SHIFT = 7
WEIGHT_MAX_STDP = 2000
WEIGHT_MIN_STDP = 0

# P14 Stochastic Noise
DEFAULT_NOISE_CONFIG = 0      # noise disabled (mantissa=0, exponent=0)
NOISE_LFSR_SEED = 0xACE1     # 16-bit Galois LFSR seed (must be non-zero)
NOISE_LFSR_TAPS = 0xB400     # x^16+x^14+x^13+x^11+1

# P15 Dual Spike Traces
DEFAULT_TAU1 = 3              # trace1 decay shift (matches RTL TAU1_DEFAULT)
DEFAULT_TAU2 = 4              # trace2 decay shift (matches RTL TAU2_DEFAULT)

# P17 Axon Delays
MAX_DELAY = 63                # 6-bit delay field
DEFAULT_DELAY = 0             # no delay by default
DELAY_QUEUE_BUCKETS = 64      # mod-64 timestep ring buffer

# P18 Synapse Formats
FMT_SPARSE = 0                # CSR (existing): explicit target per pool entry
FMT_DENSE = 1                 # Dense: implicit targets (base+offset), per-weight
FMT_POP = 2                   # Population: single shared weight, implicit targets
VALID_FORMATS = {'sparse': FMT_SPARSE, 'dense': FMT_DENSE, 'pop': FMT_POP}

# 3-factor learning constants (P13c)
REWARD_SHIFT = 7              # scales reward * eligibility
ELIG_DECAY_SHIFT = 3          # exponential decay: elig -= elig >> 3 (~12.5%/step)
ELIG_MAX = 1000               # clamp eligibility magnitude

# P20 Hierarchical Routing
DEFAULT_CLUSTER_SIZE = 4          # cores per cluster
GLOBAL_ROUTE_SLOTS = 4            # max inter-cluster route slots per source neuron

# P19 Microcode Learning Engine
MICROCODE_DEPTH = 64              # instructions per core
MICROCODE_LTD_START = 0           # LTD program region start
MICROCODE_LTP_START = 16          # LTP program region start

# Host command IDs (synced with RTL host_interface.v v1.0)
CMD_PROG_POOL = 0x01          # P13a: CSR pool entry (8B)
CMD_PROG_ROUTE = 0x02         # P13b: inter-core route with slot (9B)
CMD_STIMULUS = 0x03           # P13a: widened to 5B (10-bit neuron addr)
CMD_RUN = 0x04
CMD_STATUS = 0x05
CMD_LEARN_CFG = 0x06          # bit[0-5]: learn/graded/dendritic/async/3factor/noise
CMD_PROG_NEURON = 0x07        # P9+: param_id 0-7 (threshold..tau2)
CMD_PROG_INDEX = 0x08         # P13a/P18: CSR index entry
CMD_REWARD = 0x09             # P13c: reward signal (2B)
CMD_PROG_DELAY = 0x0A         # P17: axon delay (4B)
CMD_PROG_LEARN = 0x0C         # P19: microcode instruction (6B)
CMD_PROG_GLOBAL_ROUTE = 0x10  # P20: inter-cluster route (9B)
# Legacy aliases
CMD_PROG_CONN = CMD_PROG_POOL

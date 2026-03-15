"""
Neurocore — Python SDK for the custom neuromorphic chip.

Usage:
    import neurocore as nc

    net = nc.Network()
    exc = net.population(64, params={"threshold": 800, "leak": 5})
    inh = net.population(16, params={"threshold": 600, "leak": 2})

    net.connect(exc, exc, topology="random_sparse", p=0.1, weight=200)
    net.connect(exc, inh, topology="all_to_all", weight=150)
    net.connect(inh, exc, topology="all_to_all", weight=-300, compartment=1)

    sim = nc.Simulator()       # or nc.Chip(port="COM3") for hardware
    sim.deploy(net)

    sim.inject(exc[:8], current=1200)
    result = sim.run(timesteps=100)

    result.raster_plot()
    print(result.firing_rates())
"""

from .network import Network, Population, PopulationSlice, Connection, NeuronParams
from .compiler import Compiler, CompiledNetwork, Placement
from .simulator import Simulator
from .chip import Chip
try:
    from .gpu_simulator import GpuSimulator
except ImportError:
    pass  # PyTorch not installed; GpuSimulator unavailable
from .result import RunResult
from .microcode import (
    LearningRule,
    encode_instruction, decode_instruction, execute_program,
    OP_NOP, OP_ADD, OP_SUB, OP_MUL, OP_SHR, OP_SHL,
    OP_MAX, OP_MIN, OP_LOADI, OP_STORE_W, OP_STORE_E,
    OP_SKIP_Z, OP_SKIP_NZ, OP_HALT,
    R_TRACE1, R_TRACE2, R_WEIGHT, R_ELIG, R_CONST,
    R_TEMP0, R_TEMP1, R_REWARD,
)
from .exceptions import (
    NeurocoreError, NetworkTooLargeError, FanoutOverflowError,
    PoolOverflowError, RouteOverflowError,
    WeightOutOfRangeError, PlacementError, InvalidParameterError,
    ChipCommunicationError,
)

__version__ = "1.0.0"  # Loihi 1 parity: P14-P20 (noise, traces, delays, formats, microcode, routing)

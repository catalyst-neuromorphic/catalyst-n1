from .network import Network, Population, PopulationSlice, Connection, NeuronParams
from .compiler import Compiler, CompiledNetwork, Placement
from .simulator import Simulator
from .chip import Chip
try:
    from .gpu_simulator import GpuSimulator
except ImportError:
    pass
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

__version__ = "1.0.0"

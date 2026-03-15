"""Custom exception hierarchy for neurocore."""


class NeurocoreError(Exception):
    """Base exception for all neurocore errors."""


class NetworkTooLargeError(NeurocoreError):
    """Network exceeds hardware capacity (cores * neurons_per_core)."""


class PoolOverflowError(NeurocoreError):
    """Per-core CSR connection pool exhausted (>POOL_DEPTH entries)."""


# Legacy alias — P13a replaced fixed fanout with CSR pool
FanoutOverflowError = PoolOverflowError


class RouteOverflowError(NeurocoreError):
    """A source neuron exceeds ROUTE_FANOUT (8) multicast slots."""


class WeightOutOfRangeError(NeurocoreError):
    """Weight value outside signed 16-bit range [-32768, 32767]."""


class InvalidParameterError(NeurocoreError):
    """Invalid neuron parameter ID or value."""


class PlacementError(NeurocoreError):
    """Compiler could not place or route the network onto hardware."""


class ChipCommunicationError(NeurocoreError):
    """UART communication failure with hardware."""

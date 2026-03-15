from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    MAX_CORES, NEURONS_PER_CORE, WEIGHT_MIN, WEIGHT_MAX, COMPARTMENTS,
    DEFAULT_THRESHOLD, DEFAULT_LEAK, DEFAULT_RESTING, DEFAULT_REFRAC,
    DEFAULT_DEND_THRESHOLD, DEFAULT_NOISE_CONFIG, DEFAULT_TAU1, DEFAULT_TAU2,
    ROUTE_FANOUT, MAX_DELAY, VALID_FORMATS,
)
from .exceptions import (
    NetworkTooLargeError, WeightOutOfRangeError, NeurocoreError,
)

@dataclass
class NeuronParams:
    threshold: int = DEFAULT_THRESHOLD
    leak: int = DEFAULT_LEAK
    resting: int = DEFAULT_RESTING
    refrac: int = DEFAULT_REFRAC
    dend_threshold: int = DEFAULT_DEND_THRESHOLD
    noise_config: int = DEFAULT_NOISE_CONFIG
    tau1: int = DEFAULT_TAU1
    tau2: int = DEFAULT_TAU2

    @staticmethod
    def from_dict(d):
        p = NeuronParams()
        for k, v in d.items():
            if not hasattr(p, k):
                raise ValueError(f"Unknown neuron parameter: '{k}'")
            setattr(p, k, int(v))
        return p

class PopulationSlice:

    def __init__(self, population, indices):
        self.population = population
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"PopulationSlice({self.population.label}, n={len(self.indices)})"

class Population:

    def __init__(self, pop_id, size, params=None, label=None):
        if size <= 0:
            raise ValueError(f"Population size must be positive, got {size}")
        self.id = pop_id
        self.size = size
        self.params = params or NeuronParams()
        self.label = label or f"pop_{pop_id}"
        self._placement = None

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = self.size + key
            if key < 0 or key >= self.size:
                raise IndexError(f"Neuron index {key} out of range for population size {self.size}")
            return PopulationSlice(self, [key])
        elif isinstance(key, slice):
            indices = range(*key.indices(self.size))
            return PopulationSlice(self, indices)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"Population('{self.label}', size={self.size})"

@dataclass
class Connection:
    source: Population
    target: Population
    topology: str = "all_to_all"
    weight: int = 200
    p: float = 0.1
    compartment: int = 0
    seed: Optional[int] = None
    fan_in: int = 8
    fan_out: int = 8
    delay: int = 0
    format: str = 'sparse'
    weight_matrix: object = None

class Network:

    def __init__(self):
        self.populations = []
        self.connections = []
        self._next_pop_id = 0
        self._learning_rule = None

    def population(self, size, params=None, label=None):
        if isinstance(params, dict):
            params = NeuronParams.from_dict(params)
        pop = Population(self._next_pop_id, size, params, label)
        self._next_pop_id += 1
        self.populations.append(pop)
        return pop

    def connect(self, source, target, topology="all_to_all", weight=200,
                p=0.1, compartment=0, seed=None, fan_in=8, fan_out=8,
                delay=0, format='sparse', weight_matrix=None):
        if weight_matrix is not None:
            import numpy as np
            wm = np.asarray(weight_matrix, dtype=np.int32)
            if wm.shape != (source.size, target.size):
                raise ValueError(
                    f"weight_matrix shape {wm.shape} doesn't match "
                    f"({source.size}, {target.size})")
            if np.any(wm < WEIGHT_MIN) or np.any(wm > WEIGHT_MAX):
                raise WeightOutOfRangeError(
                    f"weight_matrix values outside [{WEIGHT_MIN}, {WEIGHT_MAX}]")
        else:
            if weight < WEIGHT_MIN or weight > WEIGHT_MAX:
                raise WeightOutOfRangeError(
                    f"Weight {weight} outside range [{WEIGHT_MIN}, {WEIGHT_MAX}]")
        if compartment < 0 or compartment >= COMPARTMENTS:
            raise ValueError(
                f"Compartment {compartment} outside range [0, {COMPARTMENTS - 1}]")
        if delay < 0 or delay > MAX_DELAY:
            raise ValueError(
                f"Delay {delay} outside range [0, {MAX_DELAY}]")
        if format not in VALID_FORMATS:
            raise ValueError(
                f"Unknown format '{format}'. Valid: {list(VALID_FORMATS)}")
        conn = Connection(
            source=source, target=target, topology=topology,
            weight=weight, p=p, compartment=compartment, seed=seed,
            fan_in=fan_in, fan_out=fan_out, delay=delay, format=format,
            weight_matrix=weight_matrix,
        )
        self.connections.append(conn)
        return conn

    def set_learning_rule(self, rule):
        self._learning_rule = rule

    def total_neurons(self):
        return sum(p.size for p in self.populations)

    def validate(self):
        warnings = []
        total = self.total_neurons()
        capacity = MAX_CORES * NEURONS_PER_CORE
        if total > capacity:
            raise NetworkTooLargeError(
                f"Network has {total} neurons but hardware supports {capacity}")
        if total == 0:
            warnings.append("Network has no neurons")
        for conn in self.connections:
            if conn.source not in self.populations:
                raise NeurocoreError(
                    f"Connection source {conn.source} not in this network")
            if conn.target not in self.populations:
                raise NeurocoreError(
                    f"Connection target {conn.target} not in this network")
        return warnings

    def __repr__(self):
        return (f"Network(populations={len(self.populations)}, "
                f"connections={len(self.connections)}, "
                f"neurons={self.total_neurons()})")

import os
import sys

import serial

from .backend import Backend
from .compiler import Compiler, CompiledNetwork
from .network import Network, Population, PopulationSlice
from .constants import NEURONS_PER_CORE
from .exceptions import ChipCommunicationError, NeurocoreError

_FPGA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "fpga"))
if _FPGA_DIR not in sys.path:
    sys.path.insert(0, _FPGA_DIR)

class Chip(Backend):

    def __init__(self, port="COM3", baud=115200, timeout=10):
        from host import NeuromorphicChip
        try:
            self._hw = NeuromorphicChip(port, baud, timeout)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"Failed to connect: {e}") from e
        self._compiled = None
        self._compiler = Compiler()

    def deploy(self, network_or_compiled):
        if isinstance(network_or_compiled, Network):
            self._compiled = self._compiler.compile(network_or_compiled)
        elif isinstance(network_or_compiled, CompiledNetwork):
            self._compiled = network_or_compiled
        else:
            raise TypeError(f"Expected Network or CompiledNetwork, got {type(network_or_compiled)}")

        try:
            for cmd in self._compiled.prog_neuron_cmds:
                self._hw.prog_neuron(**cmd)

            for cmd in self._compiled.prog_index_cmds:
                self._hw.prog_index(**cmd)

            for cmd in self._compiled.prog_pool_cmds:
                self._hw.prog_pool(**cmd)

            for cmd in self._compiled.prog_route_cmds:
                self._hw.prog_route(**cmd)

            for cmd in self._compiled.prog_delay_cmds:
                self._hw.prog_delay(**cmd)

            for cmd in self._compiled.prog_learn_cmds:
                self._hw.prog_learn(**cmd)

            for cmd in self._compiled.prog_global_route_cmds:
                self._hw.prog_global_route(**cmd)

            cfg = self._compiled.learn_config
            self._hw.set_learning(**cfg)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"Deploy failed: {e}") from e

    def inject(self, target, current):
        resolved = self._resolve_targets(target)
        try:
            for core, neuron in resolved:
                self._hw.stimulus(core, neuron, current)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"Stimulus failed: {e}") from e

    def run(self, timesteps):
        from .result import RunResult
        try:
            spike_count = self._hw.run(timesteps)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"Run failed: {e}") from e
        return RunResult(
            total_spikes=spike_count,
            timesteps=timesteps,
            spike_trains={},
            placement=self._compiled.placement if self._compiled else None,
            backend="chip",
        )

    def set_learning(self, learn=False, graded=False, dendritic=False,
                     async_mode=False, three_factor=False, noise=False):
        try:
            self._hw.set_learning(learn, graded, dendritic, async_mode,
                                  three_factor, noise_enable=noise)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"set_learning failed: {e}") from e

    def reward(self, value):
        try:
            self._hw.reward(value)
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"reward failed: {e}") from e

    def status(self):
        try:
            state, ts = self._hw.status()
            return {"state": state, "timestep_count": ts}
        except (serial.SerialException, serial.SerialTimeoutException, OSError,
                TimeoutError, ValueError) as e:
            raise ChipCommunicationError(f"Status query failed: {e}") from e

    def close(self):
        self._hw.close()

    def _resolve_targets(self, target):
        if isinstance(target, list):
            return target
        if self._compiled is None:
            raise NeurocoreError("No network deployed. Call deploy() first.")
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

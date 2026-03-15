"""Hardware backend: communicates with the neuromorphic FPGA over UART.

Wraps the existing fpga/host.py NeuromorphicChip class.

P13 update: CSR pool programming (prog_pool, prog_index),
multicast routing with slots, reward signal command.
"""

import os
import sys

from .backend import Backend
from .compiler import Compiler, CompiledNetwork
from .network import Network, Population, PopulationSlice
from .constants import NEURONS_PER_CORE
from .exceptions import ChipCommunicationError, NeurocoreError

# Import host.py from the fpga directory (two levels up from this file)
_FPGA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "fpga"))
if _FPGA_DIR not in sys.path:
    sys.path.insert(0, _FPGA_DIR)


class Chip(Backend):
    """Hardware backend communicating via UART."""

    def __init__(self, port="COM3", baud=115200, timeout=10):
        from host import NeuromorphicChip
        try:
            self._hw = NeuromorphicChip(port, baud, timeout)
        except Exception as e:
            raise ChipCommunicationError(f"Failed to connect: {e}") from e
        self._compiled = None
        self._compiler = Compiler()

    def deploy(self, network_or_compiled):
        """Deploy a Network or CompiledNetwork to the FPGA.

        P13 deploy order: neuron params -> CSR index -> CSR pool -> routes -> learning config
        """
        if isinstance(network_or_compiled, Network):
            self._compiled = self._compiler.compile(network_or_compiled)
        elif isinstance(network_or_compiled, CompiledNetwork):
            self._compiled = network_or_compiled
        else:
            raise TypeError(f"Expected Network or CompiledNetwork, got {type(network_or_compiled)}")

        try:
            # 1. Neuron params first
            for cmd in self._compiled.prog_neuron_cmds:
                self._hw.prog_neuron(**cmd)

            # 2. CSR index table
            for cmd in self._compiled.prog_index_cmds:
                self._hw.prog_index(**cmd)

            # 3. CSR pool entries
            for cmd in self._compiled.prog_pool_cmds:
                self._hw.prog_pool(**cmd)

            # 4. Inter-core routes (with multicast slot)
            for cmd in self._compiled.prog_route_cmds:
                self._hw.prog_route(**cmd)

            # 4b. Delay commands (P17)
            for cmd in self._compiled.prog_delay_cmds:
                self._hw.prog_delay(**cmd)

            # 4c. Microcode learning programs (P19)
            for cmd in self._compiled.prog_learn_cmds:
                self._hw.prog_learn(**cmd)

            # 4d. Global route commands (P20)
            for cmd in self._compiled.prog_global_route_cmds:
                self._hw.prog_global_route(**cmd)

            # 5. Learning config
            cfg = self._compiled.learn_config
            self._hw.set_learning(**cfg)
        except Exception as e:
            raise ChipCommunicationError(f"Deploy failed: {e}") from e

    def inject(self, target, current):
        """Inject stimulus. Target: Population, PopulationSlice, or [(core, neuron)]."""
        resolved = self._resolve_targets(target)
        try:
            for core, neuron in resolved:
                self._hw.stimulus(core, neuron, current)
        except Exception as e:
            raise ChipCommunicationError(f"Stimulus failed: {e}") from e

    def run(self, timesteps):
        """Run and return results.

        Note: hardware only returns total spike count, not per-neuron data.
        Use Simulator backend for raster plots and per-neuron analysis.
        """
        from .result import RunResult
        try:
            spike_count = self._hw.run(timesteps)
        except Exception as e:
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
        except Exception as e:
            raise ChipCommunicationError(f"set_learning failed: {e}") from e

    def reward(self, value):
        """Send reward signal to hardware (P13c CMD_REWARD)."""
        try:
            self._hw.reward(value)
        except Exception as e:
            raise ChipCommunicationError(f"reward failed: {e}") from e

    def status(self):
        try:
            state, ts = self._hw.status()
            return {"state": state, "timestep_count": ts}
        except Exception as e:
            raise ChipCommunicationError(f"Status query failed: {e}") from e

    def close(self):
        self._hw.close()

    def _resolve_targets(self, target):
        """Convert Population/PopulationSlice/list to [(core, neuron)] pairs."""
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

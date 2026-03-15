"""
Neuromorphic Chip Host Controller
==================================
Python script to communicate with the neuromorphic FPGA over UART.

v1.0 Loihi parity: CSR pool, multicast routing, noise, dual traces,
axon delays, synapse formats, microcode learning, hierarchical routing.

Usage:
    python fpga/host.py --port COM3          # Windows
    python fpga/host.py --port /dev/ttyUSB1  # Linux

Commands:
    python fpga/host.py --port COM3 --demo   # Run demo (program chain, stimulate, run)
    python fpga/host.py --port COM3 --status # Query chip status
"""

import serial
import struct
import time
import argparse
import sys


class NeuromorphicChip:
    """Interface to the neuromorphic FPGA over UART."""

    # Command opcodes (Phase 13a protocol)
    CMD_PROG_POOL   = 0x01
    CMD_PROG_ROUTE  = 0x02
    CMD_STIMULUS    = 0x03
    CMD_RUN         = 0x04
    CMD_STATUS      = 0x05
    CMD_LEARN_CFG   = 0x06
    CMD_PROG_NEURON = 0x07
    CMD_PROG_INDEX  = 0x08
    CMD_REWARD      = 0x09
    CMD_PROG_DELAY  = 0x0A
    CMD_PROG_LEARN  = 0x0C
    CMD_PROG_GLOBAL_ROUTE = 0x10

    # Parameter IDs for CMD_PROG_NEURON
    PARAM_THRESHOLD      = 0
    PARAM_LEAK           = 1
    PARAM_RESTING        = 2
    PARAM_REFRAC         = 3
    PARAM_DEND_THRESHOLD = 4

    # Response codes
    RESP_ACK  = 0xAA
    RESP_DONE = 0xDD

    def __init__(self, port, baud=115200, timeout=10):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self._pool_alloc = {}  # per-core pool bump allocator: core -> next_addr
        print(f"Connected to {port} @ {baud} baud")

    def close(self):
        self.ser.close()

    def _send(self, data):
        """Send raw bytes."""
        self.ser.write(bytes(data))

    def _recv(self, n):
        """Receive exactly n bytes."""
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError(f"Expected {n} bytes, got {len(data)}")
        return data

    def _wait_ack(self):
        """Wait for ACK (0xAA) response."""
        resp = self._recv(1)
        if resp[0] != self.RESP_ACK:
            raise ValueError(f"Expected ACK (0xAA), got 0x{resp[0]:02X}")

    def _alloc_pool(self, core, count=1):
        """Allocate pool entries for a core (bump allocator)."""
        if core not in self._pool_alloc:
            self._pool_alloc[core] = 0
        addr = self._pool_alloc[core]
        self._pool_alloc[core] += count
        return addr

    def prog_pool(self, core, pool_addr, src, target, weight, comp=0):
        """Program a connection pool entry.

        Args:
            core: Core ID
            pool_addr: Pool address (0 to POOL_DEPTH-1)
            src: Source neuron (for reverse table, 0-1023)
            target: Target neuron (0-1023)
            weight: Signed 16-bit weight
            comp: Compartment ID (0=soma, 1-3=dendrites)
        """
        w = weight & 0xFFFF
        # Pack flags: {comp[1:0], src[9:8], target[9:8], 2'b00}
        flags = ((comp & 0x3) << 6) | (((src >> 8) & 0x3) << 4) | (((target >> 8) & 0x3) << 2)
        self._send([
            self.CMD_PROG_POOL,
            core & 0xFF,
            (pool_addr >> 8) & 0xFF, pool_addr & 0xFF,
            flags,
            src & 0xFF,
            target & 0xFF,
            (w >> 8) & 0xFF, w & 0xFF
        ])
        self._wait_ack()

    def prog_index(self, core, neuron, base_addr, count, format=0, base_target=0):
        """Program a CSR index entry (base_addr + count for a neuron).

        Args:
            core: Core ID
            neuron: Neuron ID (0-1023)
            base_addr: Pool base address
            count: Number of connections
            format: Synapse format (0=sparse, 1=dense, 2=pop)
            base_target: Base target neuron for dense/pop formats
        """
        self._send([
            self.CMD_PROG_INDEX,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            (base_addr >> 8) & 0xFF, base_addr & 0xFF,
            (count >> 8) & 0xFF, count & 0xFF,
            ((format & 0x3) << 6) | ((base_target >> 8) & 0x3),
            base_target & 0xFF,
        ])
        self._wait_ack()

    def prog_conn(self, core, src, targets_weights, comp=0):
        """High-level: program connections for a source neuron using pool allocator.

        Args:
            core: Core ID
            src: Source neuron
            targets_weights: List of (target, weight) tuples
            comp: Compartment ID (default 0=soma)
        """
        if not targets_weights:
            return
        base = self._alloc_pool(core, len(targets_weights))
        for i, (target, weight) in enumerate(targets_weights):
            self.prog_pool(core, base + i, src, target, weight, comp)
        self.prog_index(core, src, base, len(targets_weights))

    def prog_route(self, src_core, src_neuron, dest_core, dest_neuron, weight, slot=0):
        """Program an inter-core route (multicast slot).

        Args:
            src_core: Source core ID
            src_neuron: Source neuron (0-1023)
            dest_core: Destination core ID
            dest_neuron: Destination neuron (0-1023)
            weight: Signed 16-bit weight
            slot: Route slot (0-7) for multicast fanout
        """
        w = weight & 0xFFFF
        self._send([
            self.CMD_PROG_ROUTE,
            src_core & 0xFF,
            (src_neuron >> 8) & 0xFF, src_neuron & 0xFF,
            slot & 0xFF,
            dest_core & 0xFF,
            (dest_neuron >> 8) & 0xFF, dest_neuron & 0xFF,
            (w >> 8) & 0xFF, w & 0xFF
        ])
        self._wait_ack()

    def stimulus(self, core, neuron, current):
        """Set external stimulus current for next RUN.

        Args:
            core: Target core ID
            neuron: Target neuron (0-1023)
            current: Signed 16-bit current value
        """
        c = current & 0xFFFF
        self._send([
            self.CMD_STIMULUS,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            (c >> 8) & 0xFF, c & 0xFF
        ])
        self._wait_ack()

    def run(self, timesteps):
        """Run the mesh for N timesteps.

        Args:
            timesteps: Number of timesteps (1-65535)

        Returns:
            Number of spikes that occurred during the run.
        """
        ts = timesteps & 0xFFFF
        self._send([
            self.CMD_RUN,
            (ts >> 8) & 0xFF, ts & 0xFF
        ])
        resp = self._recv(5)
        if resp[0] != self.RESP_DONE:
            raise ValueError(f"Expected DONE (0xDD), got 0x{resp[0]:02X}")
        spikes = struct.unpack('>I', resp[1:5])[0]
        return spikes

    def reward(self, value):
        """Set reward value for 3-factor learning.

        Args:
            value: Signed 16-bit reward (0 = no reward)
        """
        v = value & 0xFFFF
        self._send([
            self.CMD_REWARD,
            (v >> 8) & 0xFF, v & 0xFF
        ])
        self._wait_ack()

    def set_learning(self, learn_enable, graded_enable=False, dendritic_enable=False,
                      async_enable=False, threefactor_enable=False, noise_enable=False):
        """Configure learning mode flags."""
        flags = ((int(learn_enable) & 1)
                 | ((int(graded_enable) & 1) << 1)
                 | ((int(dendritic_enable) & 1) << 2)
                 | ((int(async_enable) & 1) << 3)
                 | ((int(threefactor_enable) & 1) << 4)
                 | ((int(noise_enable) & 1) << 5))
        self._send([self.CMD_LEARN_CFG, flags])
        self._wait_ack()

    def prog_delay(self, core, pool_addr, delay):
        """Program an axon delay for a pool entry (P17).

        Args:
            core: Core ID
            pool_addr: Pool address of the connection
            delay: Delay in timesteps (0-63)
        """
        self._send([
            self.CMD_PROG_DELAY,
            core & 0xFF,
            (pool_addr >> 8) & 0xFF, pool_addr & 0xFF,
            delay & 0x3F,
        ])
        self._wait_ack()

    def prog_learn(self, core, addr, instr):
        """Program a microcode learning instruction (P19).

        Args:
            core: Core ID
            addr: Instruction address (0-63)
            instr: 32-bit instruction word
        """
        self._send([
            self.CMD_PROG_LEARN,
            core & 0xFF,
            addr & 0x3F,
            (instr >> 24) & 0xFF,
            (instr >> 16) & 0xFF,
            (instr >> 8) & 0xFF,
            instr & 0xFF,
        ])
        self._wait_ack()

    def prog_global_route(self, src_core, src_neuron, dest_core, dest_neuron,
                           weight, slot=0):
        """Program an inter-cluster global route (P20).

        Args:
            src_core: Source core ID
            src_neuron: Source neuron (0-1023)
            dest_core: Destination core ID
            dest_neuron: Destination neuron (0-1023)
            weight: Signed 16-bit weight
            slot: Route slot (0-3)
        """
        w = weight & 0xFFFF
        self._send([
            self.CMD_PROG_GLOBAL_ROUTE,
            src_core & 0xFF,
            (src_neuron >> 8) & 0xFF, src_neuron & 0xFF,
            slot & 0xFF,
            dest_core & 0xFF,
            (dest_neuron >> 8) & 0xFF, dest_neuron & 0xFF,
            (w >> 8) & 0xFF, w & 0xFF,
        ])
        self._wait_ack()

    def async_mode(self, enable=True):
        """Enable or disable async event-driven mode."""
        self.set_learning(False, False, False, async_enable=enable)

    def prog_neuron(self, core, neuron, param_id, value):
        """Program a per-neuron parameter.

        Args:
            core: Core ID
            neuron: Neuron ID (0-1023)
            param_id: Parameter (PARAM_THRESHOLD=0, PARAM_LEAK=1, etc.)
            value: Signed 16-bit value
        """
        v = value & 0xFFFF
        self._send([
            self.CMD_PROG_NEURON,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            param_id & 0xFF,
            (v >> 8) & 0xFF, v & 0xFF
        ])
        self._wait_ack()

    def status(self):
        """Query chip status.

        Returns:
            Tuple of (state, timestep_count)
        """
        self._send([self.CMD_STATUS])
        resp = self._recv(5)
        state = resp[0]
        ts_count = struct.unpack('>I', resp[1:5])[0]
        return state, ts_count


def demo(chip):
    """Run a demonstration: program a spike chain and observe propagation."""

    print("\n" + "=" * 60)
    print("  Neuromorphic Chip Demo (Phase 13b: CSR + Multicast)")
    print("=" * 60)

    state, ts = chip.status()
    print(f"\nInitial status: state={state}, timesteps={ts}")

    # Program a spike chain: Core 0, N0→N1→N2→N3
    print("\nProgramming spike chain: Core 0, N0 -> N1 -> N2 -> N3")
    chip.prog_conn(0, 0, [(1, 1200)])
    print("  N0 -> N1 (w=1200) OK")
    chip.prog_conn(0, 1, [(2, 1200)])
    print("  N1 -> N2 (w=1200) OK")
    chip.prog_conn(0, 2, [(3, 1200)])
    print("  N2 -> N3 (w=1200) OK")

    # Program cross-core route: Core 0 N3 → Core 1 N0
    print("\nProgramming cross-core route: C0:N3 -> C1:N0")
    chip.prog_route(src_core=0, src_neuron=3,
                    dest_core=1, dest_neuron=0, weight=1200)
    print("  Route OK")

    # Core 1 chain
    print("Programming Core 1 chain: N0 -> N1 -> N2")
    chip.prog_conn(1, 0, [(1, 1200)])
    chip.prog_conn(1, 1, [(2, 1200)])
    print("  Core 1 chain OK")

    # Stimulate and run
    print("\nApplying stimulus: Core 0, N0, current=1200")
    chip.stimulus(core=0, neuron=0, current=1200)

    print("Running 20 timesteps...")
    t_start = time.time()
    spikes = chip.run(20)
    elapsed = time.time() - t_start
    print(f"  Done! {spikes} spikes in {elapsed:.3f}s")

    # Run more without stimulus
    print("\nRunning 10 more timesteps (no stimulus)...")
    spikes2 = chip.run(10)
    print(f"  {spikes2} spikes (should be 0 - no input)")

    # Final status
    state, ts = chip.status()
    print(f"\nFinal status: state={state}, timesteps={ts}")

    print("\n" + "=" * 60)
    print("  Demo complete! The chip is alive.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Chip Host Controller")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/ttyUSB1)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--demo", action="store_true", help="Run demo program")
    parser.add_argument("--status", action="store_true", help="Query chip status")
    args = parser.parse_args()

    chip = NeuromorphicChip(args.port, args.baud)

    try:
        if args.status:
            state, ts = chip.status()
            print(f"State: {state} ({'idle' if state == 0 else 'busy'})")
            print(f"Timestep count: {ts}")
        elif args.demo:
            demo(chip)
        else:
            print("No command specified. Use --demo or --status")
            print("Or import NeuromorphicChip in Python for programmatic access:")
            print("")
            print("  from host import NeuromorphicChip")
            print("  chip = NeuromorphicChip('COM3')")
            print("  chip.prog_conn(0, 0, [(1, 1200), (2, 800)])  # N0 -> N1(w=1200), N2(w=800)")
            print("  chip.prog_index(0, 0, 0, 2)  # Or use prog_conn() which handles this")
            print("  chip.stimulus(core=0, neuron=0, current=1200)")
            print("  spikes = chip.run(100)")
    finally:
        chip.close()


if __name__ == "__main__":
    main()

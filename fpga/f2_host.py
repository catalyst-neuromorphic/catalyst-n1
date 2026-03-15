"""
Neuromorphic Chip F2 Host Controller
=====================================
Python driver for the neuromorphic FPGA on AWS F2, communicating via
PCIe MMIO (AXI-Lite registers) instead of UART.

Same byte-level protocol as host.py, different transport layer.

Usage:
    python fpga/f2_host.py --demo                   # Run demo (fpga_mgmt transport)
    python fpga/f2_host.py --status                  # Query chip status
    python fpga/f2_host.py --test-loopback           # Connectivity test
    python fpga/f2_host.py --test-spike              # Spike chain test
    python fpga/f2_host.py --transport mmap           # Use mmap transport

Register map (BAR0 offsets, via fpga_mgmt BAR0):
    0x000 [W]   TX_DATA    - write byte to host_interface
    0x004 [R]   TX_STATUS  - bit[0] = ready (TX FIFO not full)
    0x008 [R]   RX_DATA    - read response byte (auto-pops)
    0x00C [R]   RX_STATUS  - bit[0] = not empty
    0x010 [R/W] CONTROL    - bit[0] = soft reset (self-clearing)
    0x014 [R]   VERSION    - firmware version (0xF2020310 = 16-core)
    0x018 [R/W] SCRATCH    - loopback register
    0x01C [R]   CORE_COUNT - number of cores

FPGA BRAM init workaround:
    On FPGA, all SRAMs init to 0. For compartment system correctness,
    each used neuron must have is_root=1 (param_id=24) and
    parent_ptr=1023 (param_id=22) set explicitly. Use setup_neuron().
"""

import struct
import time
import argparse
import sys


class MmapTransport:
    """MMIO via mmap of /dev/fpga0_ocl BAR0."""

    def __init__(self, device="/dev/fpga0_ocl", bar_size=0x10000):
        import mmap
        import os
        fd = os.open(device, os.O_RDWR | os.O_SYNC)
        self._mm = mmap.mmap(fd, bar_size, access=mmap.ACCESS_WRITE)
        os.close(fd)  # mmap keeps its own reference

    def write32(self, offset, value):
        struct.pack_into('<I', self._mm, offset, value & 0xFFFFFFFF)

    def read32(self, offset):
        return struct.unpack_from('<I', self._mm, offset)[0]

    def close(self):
        self._mm.close()


class FpgaMgmtTransport:
    """MMIO via AWS FPGA Management Library (libfpga_mgmt.so)."""

    def __init__(self, slot=0, bar=0):
        import ctypes
        self._lib = ctypes.CDLL("libfpga_mgmt.so")

        # fpga_mgmt_init()
        rc = self._lib.fpga_mgmt_init()
        if rc != 0:
            raise RuntimeError(f"fpga_mgmt_init failed: {rc}")

        # fpga_pci_attach(slot, pf_id=0, bar, flags=0, &handle)
        self._handle = ctypes.c_int()
        rc = self._lib.fpga_pci_attach(slot, 0, bar, 0,
                                        ctypes.byref(self._handle))
        if rc != 0:
            raise RuntimeError(f"fpga_pci_attach failed: {rc}")

        self._poke = self._lib.fpga_pci_poke
        self._peek = self._lib.fpga_pci_peek
        self._ctypes = ctypes

    def write32(self, offset, value):
        rc = self._poke(self._handle, offset, value & 0xFFFFFFFF)
        if rc != 0:
            raise RuntimeError(f"fpga_pci_poke(0x{offset:X}, 0x{value:X}) failed: {rc}")

    def read32(self, offset):
        val = self._ctypes.c_uint32()
        rc = self._peek(self._handle, offset, self._ctypes.byref(val))
        if rc != 0:
            raise RuntimeError(f"fpga_pci_peek(0x{offset:X}) failed: {rc}")
        return val.value

    def close(self):
        self._lib.fpga_pci_detach(self._handle)


class F2NeuromorphicChip:
    """Interface to the neuromorphic FPGA via PCIe MMIO."""

    # Register offsets
    REG_TX_DATA    = 0x000
    REG_TX_STATUS  = 0x004
    REG_RX_DATA    = 0x008
    REG_RX_STATUS  = 0x00C
    REG_CONTROL    = 0x010
    REG_VERSION    = 0x014
    REG_SCRATCH    = 0x018
    REG_CORE_COUNT = 0x01C

    # Command opcodes (same as host.py)
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

    # Parameter IDs
    PARAM_THRESHOLD      = 0
    PARAM_LEAK           = 1
    PARAM_RESTING        = 2
    PARAM_REFRAC         = 3
    PARAM_DEND_THRESHOLD = 4
    PARAM_DECAY_V        = 16
    PARAM_DECAY_U        = 17
    PARAM_BIAS_CFG       = 18
    PARAM_PARENT_PTR     = 22
    PARAM_JOINOP         = 23
    PARAM_IS_ROOT        = 24

    # Response codes
    RESP_ACK  = 0xAA
    RESP_DONE = 0xDD

    def __init__(self, transport='fpga_mgmt', slot=0, timeout=5.0):
        if transport == 'mmap':
            self._t = MmapTransport()
        elif transport == 'fpga_mgmt':
            self._t = FpgaMgmtTransport(slot=slot)
        else:
            raise ValueError(f"Unknown transport: {transport}")

        self._timeout = timeout
        self._pool_alloc = {}

        # Verify connectivity
        ver = self._t.read32(self.REG_VERSION)
        cores = self._t.read32(self.REG_CORE_COUNT)
        self._num_cores = cores
        print(f"Connected via {transport}: version=0x{ver:08X}, cores={cores}")

    def close(self):
        self._t.close()

    def _send(self, data):
        """Send bytes to host_interface via TX FIFO."""
        for b in data:
            deadline = time.monotonic() + self._timeout
            while True:
                status = self._t.read32(self.REG_TX_STATUS)
                if status & 1:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("TX FIFO full timeout")
            self._t.write32(self.REG_TX_DATA, b & 0xFF)

    def _recv(self, n):
        """Receive n bytes from host_interface via RX FIFO."""
        result = bytearray()
        deadline = time.monotonic() + self._timeout
        while len(result) < n:
            status = self._t.read32(self.REG_RX_STATUS)
            if status & 1:  # not empty
                val = self._t.read32(self.REG_RX_DATA)
                result.append(val & 0xFF)
                deadline = time.monotonic() + self._timeout  # Reset per byte
            elif time.monotonic() > deadline:
                raise TimeoutError(
                    f"RX timeout: got {len(result)}/{n} bytes")
        return bytes(result)

    def _wait_ack(self):
        """Wait for ACK (0xAA) response."""
        resp = self._recv(1)
        if resp[0] != self.RESP_ACK:
            raise ValueError(f"Expected ACK (0xAA), got 0x{resp[0]:02X}")

    def _alloc_pool(self, core, count=1):
        """Allocate pool entries (bump allocator)."""
        if core not in self._pool_alloc:
            self._pool_alloc[core] = 0
        addr = self._pool_alloc[core]
        self._pool_alloc[core] += count
        return addr

    def soft_reset(self):
        """Issue a soft reset (clears FIFOs)."""
        self._t.write32(self.REG_CONTROL, 1)
        time.sleep(0.001)

    def read_version(self):
        return self._t.read32(self.REG_VERSION)

    def read_core_count(self):
        return self._t.read32(self.REG_CORE_COUNT)

    def test_scratch(self, value=0xDEADBEEF):
        """Write/read SCRATCH register for loopback test."""
        self._t.write32(self.REG_SCRATCH, value)
        readback = self._t.read32(self.REG_SCRATCH)
        return readback == value, readback

    def prog_pool(self, core, pool_addr, src, target, weight, comp=0):
        w = weight & 0xFFFF
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
        self._send([
            self.CMD_PROG_INDEX,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            (base_addr >> 8) & 0xFF, base_addr & 0xFF,
            ((format & 0x3) << 6) | ((count >> 8) & 0x3F), count & 0xFF,
        ])
        self._wait_ack()

    def prog_conn(self, core, src, targets_weights, comp=0):
        if not targets_weights:
            return
        base = self._alloc_pool(core, len(targets_weights))
        for i, (target, weight) in enumerate(targets_weights):
            self.prog_pool(core, base + i, src, target, weight, comp)
        self.prog_index(core, src, base, len(targets_weights))

    def prog_route(self, src_core, src_neuron, dest_core, dest_neuron, weight, slot=0):
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
        c = current & 0xFFFF
        self._send([
            self.CMD_STIMULUS,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            (c >> 8) & 0xFF, c & 0xFF
        ])
        self._wait_ack()

    def run(self, timesteps):
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

    def status(self):
        self._send([self.CMD_STATUS])
        resp = self._recv(5)
        state = resp[0]
        ts_count = struct.unpack('>I', resp[1:5])[0]
        return state, ts_count

    def reward(self, value):
        v = value & 0xFFFF
        self._send([
            self.CMD_REWARD,
            (v >> 8) & 0xFF, v & 0xFF
        ])
        self._wait_ack()

    def set_learning(self, learn_enable, graded_enable=False, dendritic_enable=False,
                      async_enable=False, threefactor_enable=False, noise_enable=False):
        flags = ((int(learn_enable) & 1)
                 | ((int(graded_enable) & 1) << 1)
                 | ((int(dendritic_enable) & 1) << 2)
                 | ((int(async_enable) & 1) << 3)
                 | ((int(threefactor_enable) & 1) << 4)
                 | ((int(noise_enable) & 1) << 5))
        self._send([self.CMD_LEARN_CFG, flags])
        self._wait_ack()

    def prog_neuron(self, core, neuron, param_id, value):
        v = value & 0xFFFF
        self._send([
            self.CMD_PROG_NEURON,
            core & 0xFF,
            (neuron >> 8) & 0xFF, neuron & 0xFF,
            param_id & 0xFF,
            (v >> 8) & 0xFF, v & 0xFF
        ])
        self._wait_ack()

    def setup_neuron(self, core, neuron, threshold=1000):
        """Configure a neuron for standalone operation on FPGA.

        FPGA BRAMs init to 0, which breaks the compartment system:
        - is_root=0 means spikes never counted externally
        - parent_ptr=0 means all neurons cascade to neuron 0

        This sets threshold + is_root=1 + parent_ptr=sentinel for
        correct standalone operation.
        """
        self.prog_neuron(core, neuron, self.PARAM_THRESHOLD, threshold)
        self.prog_neuron(core, neuron, self.PARAM_PARENT_PTR, 1023)  # no-parent sentinel
        self.prog_neuron(core, neuron, self.PARAM_IS_ROOT, 1)

    def setup_neurons(self, neuron_list):
        """Setup multiple neurons. neuron_list: [(core, neuron, threshold), ...]"""
        for core, neuron, threshold in neuron_list:
            self.setup_neuron(core, neuron, threshold)

    def prog_delay(self, core, pool_addr, delay):
        self._send([
            self.CMD_PROG_DELAY,
            core & 0xFF,
            (pool_addr >> 8) & 0xFF, pool_addr & 0xFF,
            delay & 0x3F,
        ])
        self._wait_ack()

    def prog_learn(self, core, addr, instr):
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


def test_loopback(chip):
    """Basic connectivity test: registers only, no mesh interaction."""
    print("\n" + "=" * 60)
    print("  F2 Loopback Test")
    print("=" * 60)
    passed = 0
    total = 0

    # VERSION
    total += 1
    ver = chip.read_version()
    if ver == 0xF2020310:
        print(f"  [PASS] VERSION = 0x{ver:08X}")
        passed += 1
    else:
        print(f"  [FAIL] VERSION = 0x{ver:08X} (expected 0xF2020310)")

    # CORE_COUNT
    total += 1
    cores = chip.read_core_count()
    if cores == 16:
        print(f"  [PASS] CORE_COUNT = {cores}")
        passed += 1
    else:
        print(f"  [FAIL] CORE_COUNT = {cores} (expected 16)")

    # SCRATCH
    total += 1
    ok, val = chip.test_scratch(0xDEADBEEF)
    if ok:
        print(f"  [PASS] SCRATCH loopback = 0x{val:08X}")
        passed += 1
    else:
        print(f"  [FAIL] SCRATCH = 0x{val:08X} (expected 0xDEADBEEF)")

    total += 1
    ok, val = chip.test_scratch(0x12345678)
    if ok:
        print(f"  [PASS] SCRATCH loopback = 0x{val:08X}")
        passed += 1
    else:
        print(f"  [FAIL] SCRATCH = 0x{val:08X} (expected 0x12345678)")

    # STATUS command
    total += 1
    try:
        state, ts = chip.status()
        print(f"  [PASS] STATUS: state={state}, ts_count={ts}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] STATUS: {e}")

    print(f"\n  Result: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


def test_spike(chip):
    """Program a 2-neuron chain, inject spike, verify propagation."""
    print("\n" + "=" * 60)
    print("  F2 Spike Test")
    print("=" * 60)

    # Soft reset to clear any previous state
    chip.soft_reset()
    chip._pool_alloc = {}

    state, ts = chip.status()
    print(f"  Initial: state={state}, ts={ts}")

    # Setup neurons (FPGA BRAM init workaround)
    print("  Setting up neurons (is_root=1, parent_ptr=1023)...")
    chip.setup_neuron(0, 0, threshold=1000)
    chip.setup_neuron(0, 1, threshold=1000)

    # Program: Core 0, N0→N1 (w=1200 > threshold=1000)
    print("  Programming: N0 -> N1 (w=1200)")
    chip.prog_conn(0, 0, [(1, 1200)])

    # Stimulate N0
    print("  Stimulating: Core 0, N0, current=1200")
    chip.stimulus(core=0, neuron=0, current=1200)

    # Run 5 timesteps
    print("  Running 5 timesteps...")
    t0 = time.monotonic()
    spikes = chip.run(5)
    dt = time.monotonic() - t0
    print(f"  Result: {spikes} spikes in {dt*1000:.1f} ms")

    if spikes > 0:
        print("  [PASS] Spike propagation detected")
    else:
        print("  [FAIL] No spikes (expected > 0)")

    print("=" * 60)
    return spikes > 0


def demo(chip):
    """Run full demo: program cross-core spike chain, run, observe."""
    print("\n" + "=" * 60)
    print("  Neuromorphic Chip F2 Demo (16-core, PCIe MMIO)")
    print("=" * 60)

    chip.soft_reset()
    chip._pool_alloc = {}

    state, ts = chip.status()
    print(f"\nInitial status: state={state}, timesteps={ts}")

    # Setup neurons (FPGA BRAM init workaround)
    print("\nSetting up neurons (is_root=1, parent_ptr=1023)...")
    neurons = [(0, i, 1000) for i in range(4)] + [(1, i, 1000) for i in range(3)]
    chip.setup_neurons(neurons)
    print(f"  {len(neurons)} neurons configured")

    # Program a spike chain: Core 0, N0→N1→N2→N3
    print("\nProgramming spike chain: Core 0, N0 -> N1 -> N2 -> N3")
    chip.prog_conn(0, 0, [(1, 1200)])
    print("  N0 -> N1 (w=1200) OK")
    chip.prog_conn(0, 1, [(2, 1200)])
    print("  N1 -> N2 (w=1200) OK")
    chip.prog_conn(0, 2, [(3, 1200)])
    print("  N2 -> N3 (w=1200) OK")

    # Cross-core route: Core 0 N3 → Core 1 N0
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
    t0 = time.monotonic()
    spikes = chip.run(20)
    dt = time.monotonic() - t0
    print(f"  Done! {spikes} spikes in {dt*1000:.1f} ms")
    print(f"  Throughput: {20/dt:.0f} timesteps/sec")

    # Run more without stimulus
    print("\nRunning 10 more timesteps (no stimulus)...")
    spikes2 = chip.run(10)
    print(f"  {spikes2} spikes (should be 0 - no input)")

    # Final status
    state, ts = chip.status()
    print(f"\nFinal status: state={state}, timesteps={ts}")

    print("\n" + "=" * 60)
    print("  Demo complete! The chip is alive on F2.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Neuromorphic Chip F2 Host Controller (PCIe MMIO)")
    parser.add_argument("--transport", choices=["mmap", "fpga_mgmt"],
                        default="fpga_mgmt", help="MMIO transport (default: fpga_mgmt)")
    parser.add_argument("--slot", type=int, default=0,
                        help="FPGA slot (default: 0)")
    parser.add_argument("--demo", action="store_true",
                        help="Run full demo")
    parser.add_argument("--status", action="store_true",
                        help="Query chip status")
    parser.add_argument("--test-loopback", action="store_true",
                        help="Run loopback connectivity test")
    parser.add_argument("--test-spike", action="store_true",
                        help="Run spike propagation test")
    args = parser.parse_args()

    chip = F2NeuromorphicChip(transport=args.transport, slot=args.slot)

    try:
        if args.test_loopback:
            ok = test_loopback(chip)
            sys.exit(0 if ok else 1)
        elif args.test_spike:
            ok = test_spike(chip)
            sys.exit(0 if ok else 1)
        elif args.status:
            state, ts = chip.status()
            print(f"State: {state} ({'idle' if state == 0 else 'busy'})")
            print(f"Timestep count: {ts}")
        elif args.demo:
            demo(chip)
        else:
            print("No command specified. Use --demo, --status, --test-loopback, or --test-spike")
    finally:
        chip.close()


if __name__ == "__main__":
    main()

"""Tests for the compiler: CSR placement, pool allocation, multicast routing."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.compiler import Compiler
from neurocore.exceptions import (
    PoolOverflowError, RouteOverflowError, PlacementError, NetworkTooLargeError,
)
from neurocore.constants import NEURONS_PER_CORE, POOL_DEPTH, ROUTE_FANOUT


class TestPlacement:
    def test_single_core(self):
        net = nc.Network()
        net.population(100)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 1

    def test_two_cores(self):
        net = nc.Network()
        # P13: 1024 neurons/core, so need >1024 for 2 cores
        net.population(1025)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 2

    def test_exact_core_boundary(self):
        net = nc.Network()
        net.population(NEURONS_PER_CORE)  # exactly 1024
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 1

    def test_multiple_populations(self):
        net = nc.Network()
        net.population(800)
        net.population(400)
        c = Compiler()
        compiled = c.compile(net)
        # 800 + 400 = 1200 => 2 cores (1024 + 176)
        assert compiled.placement.num_cores_used == 2
        assert compiled.placement.total_neurons == 1200

    def test_too_many_neurons(self):
        net = nc.Network()
        net.population(128 * NEURONS_PER_CORE + 1)
        c = Compiler()
        with pytest.raises(NetworkTooLargeError):
            c.compile(net)


class TestCSRPool:
    """Tests for CSR (Compressed Sparse Row) pool allocation."""

    def test_pool_entries_generated(self):
        """Intra-core connections generate pool entries."""
        net = nc.Network()
        a = net.population(4)
        b = net.population(4)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        # 4 * 4 = 16 pool entries
        assert len(compiled.prog_pool_cmds) == 16
        assert len(compiled.prog_route_cmds) == 0

    def test_index_entries_generated(self):
        """Each source neuron with connections gets an index entry."""
        net = nc.Network()
        a = net.population(4)
        b = net.population(4)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        # 4 source neurons, each connects to 4 targets
        assert len(compiled.prog_index_cmds) == 4
        # Check first index entry
        idx0 = compiled.prog_index_cmds[0]
        assert idx0["count"] == 4
        assert idx0["base_addr"] == 0

    def test_bump_allocator_contiguous(self):
        """Pool addresses should be contiguous per core."""
        net = nc.Network()
        a = net.population(3)
        b = net.population(6)
        net.connect(a, b, topology="all_to_all", weight=100)
        c = Compiler()
        compiled = c.compile(net)
        # 3 source neurons, each with 6 connections = 18 pool entries
        assert len(compiled.prog_pool_cmds) == 18
        # Check addresses are contiguous
        addrs = [cmd["pool_addr"] for cmd in compiled.prog_pool_cmds]
        assert addrs == list(range(18))

    def test_variable_fanout(self):
        """Different source neurons can have different connection counts."""
        net = nc.Network()
        src1 = net.population(1)
        src2 = net.population(1)
        tgt_small = net.population(5)
        tgt_large = net.population(10)
        net.connect(src1, tgt_small, topology="all_to_all", weight=100)
        net.connect(src2, tgt_large, topology="all_to_all", weight=100)
        c = Compiler()
        compiled = c.compile(net)
        counts = sorted([cmd["count"] for cmd in compiled.prog_index_cmds])
        assert counts == [5, 10]

    def test_high_fanout_no_error(self):
        """With CSR pool, >32 connections per source is now allowed."""
        net = nc.Network()
        src = net.population(1)
        tgt = net.population(100)
        net.connect(src, tgt, topology="all_to_all", weight=100)
        c = Compiler()
        # This used to raise FanoutOverflowError with fixed slots!
        compiled = c.compile(net)
        assert len(compiled.prog_pool_cmds) == 100

    def test_pool_overflow(self):
        """Exceeding POOL_DEPTH per core should raise PoolOverflowError."""
        net = nc.Network()
        src = net.population(200)
        net.connect(src, src, topology="all_to_all", weight=100)
        c = Compiler()
        with pytest.raises(PoolOverflowError):
            c.compile(net)

    def test_legacy_prog_conn_alias(self):
        """prog_conn_cmds property should alias prog_pool_cmds."""
        net = nc.Network()
        a = net.population(2)
        b = net.population(2)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.prog_conn_cmds is compiled.prog_pool_cmds


class TestMulticastRouting:
    """Tests for P13b multicast inter-core routing."""

    def test_single_route(self):
        """One inter-core route per source should work."""
        net = nc.Network()
        a = net.population(NEURONS_PER_CORE)  # fills core 0
        b = net.population(1)                 # on core 1
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        # 1024 sources, each with 1 route to b[0] on core 1
        assert len(compiled.prog_route_cmds) == NEURONS_PER_CORE
        # Each route should have slot=0
        assert all(cmd["slot"] == 0 for cmd in compiled.prog_route_cmds)

    def test_multicast_two_destinations(self):
        """One source routing to 2 targets on another core (2 route slots)."""
        net = nc.Network()
        # src fills entire core 0 — targets MUST go elsewhere
        src = net.population(NEURONS_PER_CORE)
        tgt1 = net.population(1)  # core 1 neuron 0
        tgt2 = net.population(1)  # core 1 neuron 1
        net.connect(src, tgt1, topology="all_to_all", weight=200)
        net.connect(src, tgt2, topology="all_to_all", weight=200)
        comp = Compiler()
        compiled = comp.compile(net)
        # src neuron 0 should have 2 multicast route slots (to tgt1 and tgt2)
        src_core, src_neuron = compiled.placement.neuron_map[(src.id, 0)]
        routes_for_src0 = [r for r in compiled.prog_route_cmds
                           if r["src_neuron"] == src_neuron and r["src_core"] == src_core]
        assert len(routes_for_src0) == 2
        slots = sorted(r["slot"] for r in routes_for_src0)
        assert slots == [0, 1]

    def test_multicast_8_way(self):
        """Max 8 multicast destinations should work."""
        net = nc.Network()
        # src fills core 0
        src = net.population(NEURONS_PER_CORE)
        targets = []
        for _ in range(8):
            targets.append(net.population(1))
        for t in targets:
            net.connect(src, t, topology="all_to_all", weight=100)
        comp = Compiler()
        compiled = comp.compile(net)
        src_core, src_neuron = compiled.placement.neuron_map[(src.id, 0)]
        routes_for_src0 = [r for r in compiled.prog_route_cmds
                           if r["src_neuron"] == src_neuron and r["src_core"] == src_core]
        assert len(routes_for_src0) == 8

    def test_multicast_overflow(self):
        """More than ROUTE_FANOUT unique destinations should raise RouteOverflowError."""
        net = nc.Network()
        # src fills core 0
        src = net.population(NEURONS_PER_CORE)
        targets = []
        for _ in range(ROUTE_FANOUT + 1):  # 9 unique destinations
            targets.append(net.population(1))
        for t in targets:
            net.connect(src, t, topology="all_to_all", weight=100)
        comp = Compiler()
        with pytest.raises(RouteOverflowError):
            comp.compile(net)

    def test_route_deduplication(self):
        """Multiple connections to same (dest_core, dest_neuron) use 1 route slot."""
        net = nc.Network()
        a = net.population(NEURONS_PER_CORE)  # fills core 0
        b = net.population(1)                 # core 1
        # Connect entire a -> b (all 1024 source neurons to 1 target)
        # Each source gets 1 route to the same (core 1, neuron 0)
        net.connect(a, b, topology="all_to_all", weight=200)
        # Connect again with different weight — but same source->dest pairs
        net.connect(a, b, topology="all_to_all", weight=300)
        comp = Compiler()
        compiled = comp.compile(net)
        # For neuron 0 of core 0, should have only 1 route (deduplicated)
        routes_for_n0 = [r for r in compiled.prog_route_cmds
                         if r["src_neuron"] == 0 and r["src_core"] == 0]
        assert len(routes_for_n0) == 1


class TestNeuronParams:
    def test_non_default_params(self):
        net = nc.Network()
        net.population(4, params={"threshold": 800, "leak": 5})
        c = Compiler()
        compiled = c.compile(net)
        # 4 neurons * 2 non-default params = 8 commands
        assert len(compiled.prog_neuron_cmds) == 8

    def test_default_params_no_commands(self):
        net = nc.Network()
        net.population(4)  # all defaults
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_neuron_cmds) == 0


class TestCompiledSummary:
    def test_summary(self, small_network):
        net, _, _ = small_network
        c = Compiler()
        compiled = c.compile(net)
        s = compiled.summary()
        assert "pool entries" in s
        assert "inter-core" in s

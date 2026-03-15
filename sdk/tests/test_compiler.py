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
        net.population(1025)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 2

    def test_exact_core_boundary(self):
        net = nc.Network()
        net.population(NEURONS_PER_CORE)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 1

    def test_multiple_populations(self):
        net = nc.Network()
        net.population(800)
        net.population(400)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.placement.num_cores_used == 2
        assert compiled.placement.total_neurons == 1200

    def test_too_many_neurons(self):
        net = nc.Network()
        net.population(128 * NEURONS_PER_CORE + 1)
        c = Compiler()
        with pytest.raises(NetworkTooLargeError):
            c.compile(net)

class TestCSRPool:

    def test_pool_entries_generated(self):
        net = nc.Network()
        a = net.population(4)
        b = net.population(4)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_pool_cmds) == 16
        assert len(compiled.prog_route_cmds) == 0

    def test_index_entries_generated(self):
        net = nc.Network()
        a = net.population(4)
        b = net.population(4)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_index_cmds) == 4
        idx0 = compiled.prog_index_cmds[0]
        assert idx0["count"] == 4
        assert idx0["base_addr"] == 0

    def test_bump_allocator_contiguous(self):
        net = nc.Network()
        a = net.population(3)
        b = net.population(6)
        net.connect(a, b, topology="all_to_all", weight=100)
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_pool_cmds) == 18
        addrs = [cmd["pool_addr"] for cmd in compiled.prog_pool_cmds]
        assert addrs == list(range(18))

    def test_variable_fanout(self):
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
        net = nc.Network()
        src = net.population(1)
        tgt = net.population(100)
        net.connect(src, tgt, topology="all_to_all", weight=100)
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_pool_cmds) == 100

    def test_pool_overflow(self):
        net = nc.Network()
        src = net.population(200)
        net.connect(src, src, topology="all_to_all", weight=100)
        c = Compiler()
        with pytest.raises(PoolOverflowError):
            c.compile(net)

    def test_legacy_prog_conn_alias(self):
        net = nc.Network()
        a = net.population(2)
        b = net.population(2)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        assert compiled.prog_conn_cmds is compiled.prog_pool_cmds

class TestMulticastRouting:

    def test_single_route(self):
        net = nc.Network()
        a = net.population(NEURONS_PER_CORE)
        b = net.population(1)
        net.connect(a, b, topology="all_to_all", weight=200)
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_route_cmds) == NEURONS_PER_CORE
        assert all(cmd["slot"] == 0 for cmd in compiled.prog_route_cmds)

    def test_multicast_two_destinations(self):
        net = nc.Network()
        src = net.population(NEURONS_PER_CORE)
        tgt1 = net.population(1)
        tgt2 = net.population(1)
        net.connect(src, tgt1, topology="all_to_all", weight=200)
        net.connect(src, tgt2, topology="all_to_all", weight=200)
        comp = Compiler()
        compiled = comp.compile(net)
        src_core, src_neuron = compiled.placement.neuron_map[(src.id, 0)]
        routes_for_src0 = [r for r in compiled.prog_route_cmds
                           if r["src_neuron"] == src_neuron and r["src_core"] == src_core]
        assert len(routes_for_src0) == 2
        slots = sorted(r["slot"] for r in routes_for_src0)
        assert slots == [0, 1]

    def test_multicast_8_way(self):
        net = nc.Network()
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
        net = nc.Network()
        src = net.population(NEURONS_PER_CORE)
        targets = []
        for _ in range(ROUTE_FANOUT + 1):
            targets.append(net.population(1))
        for t in targets:
            net.connect(src, t, topology="all_to_all", weight=100)
        comp = Compiler()
        with pytest.raises(RouteOverflowError):
            comp.compile(net)

    def test_route_deduplication(self):
        net = nc.Network()
        a = net.population(NEURONS_PER_CORE)
        b = net.population(1)
        net.connect(a, b, topology="all_to_all", weight=200)
        net.connect(a, b, topology="all_to_all", weight=300)
        comp = Compiler()
        compiled = comp.compile(net)
        routes_for_n0 = [r for r in compiled.prog_route_cmds
                         if r["src_neuron"] == 0 and r["src_core"] == 0]
        assert len(routes_for_n0) == 1

class TestNeuronParams:
    def test_non_default_params(self):
        net = nc.Network()
        net.population(4, params={"threshold": 800, "leak": 5})
        c = Compiler()
        compiled = c.compile(net)
        assert len(compiled.prog_neuron_cmds) == 8

    def test_default_params_no_commands(self):
        net = nc.Network()
        net.population(4)
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

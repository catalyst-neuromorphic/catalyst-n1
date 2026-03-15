import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc
from neurocore.exceptions import (
    NetworkTooLargeError, WeightOutOfRangeError, NeurocoreError,
)
from neurocore.constants import MAX_CORES, NEURONS_PER_CORE

class TestPopulation:
    def test_create_population(self):
        net = nc.Network()
        pop = net.population(64, label="test")
        assert pop.size == 64
        assert pop.label == "test"
        assert pop.id == 0

    def test_population_params_dict(self):
        net = nc.Network()
        pop = net.population(16, params={"threshold": 800, "leak": 5})
        assert pop.params.threshold == 800
        assert pop.params.leak == 5
        assert pop.params.resting == 0

    def test_population_invalid_param(self):
        net = nc.Network()
        with pytest.raises(ValueError, match="Unknown neuron parameter"):
            net.population(16, params={"bogus": 42})

    def test_population_zero_size(self):
        net = nc.Network()
        with pytest.raises(ValueError, match="positive"):
            net.population(0)

    def test_population_slicing(self):
        net = nc.Network()
        pop = net.population(32)
        s = pop[:8]
        assert len(s) == 8
        assert s.indices == list(range(8))

    def test_population_single_index(self):
        net = nc.Network()
        pop = net.population(10)
        s = pop[5]
        assert len(s) == 1
        assert s.indices == [5]

    def test_population_negative_index(self):
        net = nc.Network()
        pop = net.population(10)
        s = pop[-1]
        assert s.indices == [9]

    def test_population_index_out_of_range(self):
        net = nc.Network()
        pop = net.population(10)
        with pytest.raises(IndexError):
            pop[10]

class TestConnection:
    def test_create_connection(self):
        net = nc.Network()
        a = net.population(8)
        b = net.population(8)
        conn = net.connect(a, b, topology="all_to_all", weight=200)
        assert conn.source is a
        assert conn.target is b
        assert conn.weight == 200

    def test_weight_out_of_range(self):
        net = nc.Network()
        a = net.population(8)
        b = net.population(8)
        with pytest.raises(WeightOutOfRangeError):
            net.connect(a, b, weight=40000)

    def test_invalid_compartment(self):
        net = nc.Network()
        a = net.population(8)
        b = net.population(8)
        with pytest.raises(ValueError, match="Compartment"):
            net.connect(a, b, compartment=5)

    def test_negative_weight(self):
        net = nc.Network()
        a = net.population(8)
        b = net.population(8)
        conn = net.connect(a, b, weight=-300)
        assert conn.weight == -300

class TestNetwork:
    def test_total_neurons(self):
        net = nc.Network()
        net.population(64)
        net.population(16)
        assert net.total_neurons() == 80

    def test_validate_ok(self, small_network):
        net, _, _ = small_network
        warnings = net.validate()
        assert warnings == []

    def test_validate_too_large(self):
        net = nc.Network()
        net.population(MAX_CORES * NEURONS_PER_CORE + 1)
        with pytest.raises(NetworkTooLargeError):
            net.validate()

    def test_validate_empty(self):
        net = nc.Network()
        warnings = net.validate()
        assert "no neurons" in warnings[0].lower()

    def test_repr(self):
        net = nc.Network()
        net.population(10)
        assert "neurons=10" in repr(net)

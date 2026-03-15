import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurocore as nc

@pytest.fixture
def small_network():
    net = nc.Network()
    exc = net.population(8, params={"threshold": 1000, "leak": 3}, label="exc")
    inh = net.population(4, params={"threshold": 800, "leak": 5}, label="inh")
    net.connect(exc, inh, topology="all_to_all", weight=200)
    net.connect(inh, exc, topology="all_to_all", weight=-300)
    return net, exc, inh

@pytest.fixture
def chain_network():
    net = nc.Network()
    pop = net.population(4, label="chain")
    net.connect(pop, pop, topology="one_to_one", weight=1200)
    return net, pop

@pytest.fixture
def chain_network_manual():
    net = nc.Network()
    n0 = net.population(1, label="n0")
    n1 = net.population(1, label="n1")
    n2 = net.population(1, label="n2")
    n3 = net.population(1, label="n3")
    net.connect(n0, n1, topology="all_to_all", weight=1200)
    net.connect(n1, n2, topology="all_to_all", weight=1200)
    net.connect(n2, n3, topology="all_to_all", weight=1200)
    return net, n0, n1, n2, n3

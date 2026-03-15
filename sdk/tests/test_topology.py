import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurocore import topology as topo

class TestAllToAll:
    def test_basic(self):
        pairs = topo.all_to_all(3, 4)
        assert len(pairs) == 12
        assert (0, 0) in pairs
        assert (2, 3) in pairs

    def test_self_connection(self):
        pairs = topo.all_to_all(2, 2)
        assert len(pairs) == 4

class TestOneToOne:
    def test_basic(self):
        pairs = topo.one_to_one(5, 5)
        assert len(pairs) == 5
        assert pairs == [(i, i) for i in range(5)]

    def test_size_mismatch(self):
        with pytest.raises(ValueError, match="equal sizes"):
            topo.one_to_one(3, 5)

class TestRandomSparse:
    def test_reproducible(self):
        p1 = topo.random_sparse(10, 10, p=0.5, seed=42)
        p2 = topo.random_sparse(10, 10, p=0.5, seed=42)
        assert p1 == p2

    def test_different_seeds(self):
        p1 = topo.random_sparse(10, 10, p=0.5, seed=42)
        p2 = topo.random_sparse(10, 10, p=0.5, seed=99)
        assert p1 != p2

    def test_approximate_density(self):
        pairs = topo.random_sparse(100, 100, p=0.1, seed=0)
        assert 500 < len(pairs) < 1500

class TestFixedFanIn:
    def test_basic(self):
        pairs = topo.fixed_fan_in(10, 5, fan_in=3, seed=42)
        from collections import Counter
        tgt_counts = Counter(t for _, t in pairs)
        assert all(c == 3 for c in tgt_counts.values())
        assert len(tgt_counts) == 5

    def test_fan_in_exceeds_sources(self):
        pairs = topo.fixed_fan_in(3, 5, fan_in=10, seed=42)
        from collections import Counter
        tgt_counts = Counter(t for _, t in pairs)
        assert all(c == 3 for c in tgt_counts.values())

class TestFixedFanOut:
    def test_basic(self):
        pairs = topo.fixed_fan_out(5, 10, fan_out=4, seed=42)
        from collections import Counter
        src_counts = Counter(s for s, _ in pairs)
        assert all(c == 4 for c in src_counts.values())
        assert len(src_counts) == 5

class TestRegistry:
    def test_generate(self):
        pairs = topo.generate("all_to_all", 2, 3)
        assert len(pairs) == 6

    def test_unknown_topology(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            topo.generate("bogus", 2, 3)

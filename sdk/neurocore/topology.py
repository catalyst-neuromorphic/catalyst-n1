import numpy as np

def all_to_all(src_size, tgt_size, **kwargs):
    pairs = []
    for s in range(src_size):
        for t in range(tgt_size):
            pairs.append((s, t))
    return pairs

def one_to_one(src_size, tgt_size, **kwargs):
    if src_size != tgt_size:
        raise ValueError(
            f"one_to_one requires equal sizes, got {src_size} and {tgt_size}")
    return [(i, i) for i in range(src_size)]

def random_sparse(src_size, tgt_size, p=0.1, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    pairs = []
    for s in range(src_size):
        for t in range(tgt_size):
            if rng.random() < p:
                pairs.append((s, t))
    return pairs

def fixed_fan_in(src_size, tgt_size, fan_in=8, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    pairs = []
    for t in range(tgt_size):
        sources = rng.choice(src_size, size=min(fan_in, src_size), replace=False)
        for s in sources:
            pairs.append((int(s), t))
    return pairs

def fixed_fan_out(src_size, tgt_size, fan_out=8, seed=None, **kwargs):
    rng = np.random.default_rng(seed)
    pairs = []
    for s in range(src_size):
        targets = rng.choice(tgt_size, size=min(fan_out, tgt_size), replace=False)
        for t in targets:
            pairs.append((s, int(t)))
    return pairs

TOPOLOGY_REGISTRY = {
    "all_to_all": all_to_all,
    "one_to_one": one_to_one,
    "random_sparse": random_sparse,
    "fixed_fan_in": fixed_fan_in,
    "fixed_fan_out": fixed_fan_out,
}

def generate(name, src_size, tgt_size, **kwargs):
    if name not in TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology '{name}'. Available: {list(TOPOLOGY_REGISTRY)}")
    return TOPOLOGY_REGISTRY[name](src_size, tgt_size, **kwargs)

from dataclasses import dataclass, field
from collections import defaultdict

from . import topology as topo_mod
from .constants import (
    MAX_CORES, NEURONS_PER_CORE, POOL_DEPTH, ROUTE_FANOUT,
    WEIGHT_MIN, WEIGHT_MAX,
    PARAM_THRESHOLD, PARAM_LEAK, PARAM_RESTING, PARAM_REFRAC,
    PARAM_DEND_THRESHOLD, PARAM_NOISE_CFG, PARAM_TAU1, PARAM_TAU2,
    DEFAULT_THRESHOLD, DEFAULT_LEAK, DEFAULT_RESTING,
    DEFAULT_REFRAC, DEFAULT_DEND_THRESHOLD,
    DEFAULT_NOISE_CONFIG, DEFAULT_TAU1, DEFAULT_TAU2,
    VALID_FORMATS, FMT_SPARSE, FMT_DENSE, FMT_POP,
    DEFAULT_CLUSTER_SIZE, GLOBAL_ROUTE_SLOTS,
)
from .exceptions import (
    NetworkTooLargeError, PoolOverflowError, RouteOverflowError, PlacementError,
)

@dataclass
class Placement:
    neuron_map: dict = field(default_factory=dict)
    core_assignments: dict = field(default_factory=lambda: defaultdict(list))
    num_cores_used: int = 0
    total_neurons: int = 0

@dataclass
class CompiledNetwork:
    prog_pool_cmds: list = field(default_factory=list)
    prog_index_cmds: list = field(default_factory=list)
    prog_route_cmds: list = field(default_factory=list)
    prog_neuron_cmds: list = field(default_factory=list)
    prog_delay_cmds: list = field(default_factory=list)
    prog_learn_cmds: list = field(default_factory=list)
    prog_global_route_cmds: list = field(default_factory=list)
    learning_rule: object = None
    placement: Placement = None
    learn_config: dict = field(default_factory=lambda: {
        "learn_enable": False,
        "graded_enable": False,
        "dendritic_enable": False,
        "async_enable": False,
    })
    adjacency: dict = field(default_factory=lambda: defaultdict(list))
    neuron_params: dict = field(default_factory=dict)

    @property
    def prog_conn_cmds(self):
        return self.prog_pool_cmds

    def summary(self):
        total_pool = len(self.prog_pool_cmds)
        total_index = len(self.prog_index_cmds)
        total_routes = len(self.prog_route_cmds)
        return (
            f"CompiledNetwork: {total_pool} pool entries, "
            f"{total_index} index entries, "
            f"{total_routes} inter-core routes, "
            f"{len(self.prog_neuron_cmds)} neuron param overrides, "
            f"{self.placement.num_cores_used} cores used"
        )

class Compiler:

    def __init__(self, max_cores=MAX_CORES, cluster_size=DEFAULT_CLUSTER_SIZE,
                 pool_depth=POOL_DEPTH):
        self.max_cores = max_cores
        self.cluster_size = cluster_size
        self.pool_depth = pool_depth

    def compile(self, network):
        network.validate()

        placement = self._place(network)
        compiled = CompiledNetwork(placement=placement)

        uses_dendrites = any(c.compartment > 0 for c in network.connections)
        if uses_dendrites:
            compiled.learn_config["dendritic_enable"] = True

        uses_noise = any(p.params.noise_config != DEFAULT_NOISE_CONFIG
                         for p in network.populations)
        if uses_noise:
            compiled.learn_config["noise_enable"] = True

        self._generate_neuron_params(network, placement, compiled)

        self._route(network, placement, compiled)

        if network._learning_rule is not None:
            compiled.learning_rule = network._learning_rule
            program = network._learning_rule.get_program()
            for core in range(placement.num_cores_used):
                for addr, instr in enumerate(program):
                    if instr != 0:
                        compiled.prog_learn_cmds.append({
                            "core": core, "addr": addr, "instr": instr,
                        })

        return compiled

    def _place(self, network):
        total = network.total_neurons()
        capacity = self.max_cores * NEURONS_PER_CORE
        if total > capacity:
            raise NetworkTooLargeError(
                f"Network has {total} neurons, hardware supports {capacity} "
                f"({self.max_cores} cores x {NEURONS_PER_CORE} neurons)")

        placement = Placement(total_neurons=total)
        current_core = 0
        current_offset = 0

        conn_count = defaultdict(int)
        for c in network.connections:
            conn_count[c.source.id] += 1
            conn_count[c.target.id] += 1

        sorted_pops = sorted(
            network.populations,
            key=lambda p: conn_count.get(p.id, 0),
            reverse=True,
        )

        for pop in sorted_pops:
            remaining = pop.size
            local_idx = 0
            pop._placement = []

            while remaining > 0:
                space = NEURONS_PER_CORE - current_offset
                chunk = min(remaining, space)

                for i in range(chunk):
                    core_neuron = current_offset + i
                    placement.neuron_map[(pop.id, local_idx)] = (current_core, core_neuron)
                    placement.core_assignments[current_core].append((pop.id, local_idx))
                    pop._placement.append((current_core, core_neuron))
                    local_idx += 1

                current_offset += chunk
                remaining -= chunk

                if current_offset >= NEURONS_PER_CORE:
                    current_core += 1
                    current_offset = 0

        placement.num_cores_used = current_core + (1 if current_offset > 0 else 0)
        return placement

    def _generate_neuron_params(self, network, placement, compiled):
        for pop in network.populations:
            params = pop.params
            for local_idx in range(pop.size):
                core, neuron = placement.neuron_map[(pop.id, local_idx)]
                global_id = self._global_id(core, neuron)
                compiled.neuron_params[global_id] = params

                if params.threshold != DEFAULT_THRESHOLD:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_THRESHOLD, "value": params.threshold,
                    })
                if params.leak != DEFAULT_LEAK:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_LEAK, "value": params.leak,
                    })
                if params.resting != DEFAULT_RESTING:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_RESTING, "value": params.resting,
                    })
                if params.refrac != DEFAULT_REFRAC:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_REFRAC, "value": params.refrac,
                    })
                if params.dend_threshold != DEFAULT_DEND_THRESHOLD:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_DEND_THRESHOLD, "value": params.dend_threshold,
                    })
                if params.noise_config != DEFAULT_NOISE_CONFIG:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_NOISE_CFG, "value": params.noise_config,
                    })
                if params.tau1 != DEFAULT_TAU1:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_TAU1, "value": params.tau1,
                    })
                if params.tau2 != DEFAULT_TAU2:
                    compiled.prog_neuron_cmds.append({
                        "core": core, "neuron": neuron,
                        "param_id": PARAM_TAU2, "value": params.tau2,
                    })

    def _route(self, network, placement, compiled):
        intra_conns = defaultdict(list)
        route_slots = defaultdict(list)
        src_format = {}

        for conn in network.connections:
            fmt_id = VALID_FORMATS.get(conn.format, FMT_SPARSE)

            if conn.weight_matrix is not None:
                import numpy as np
                wm = np.asarray(conn.weight_matrix, dtype=np.int32)
                pairs_weights = []
                for s in range(conn.source.size):
                    for t in range(conn.target.size):
                        if wm[s, t] != 0:
                            pairs_weights.append((s, t, int(wm[s, t])))
            else:
                pairs = topo_mod.generate(
                    conn.topology, conn.source.size, conn.target.size,
                    p=conn.p, seed=conn.seed,
                    fan_in=conn.fan_in, fan_out=conn.fan_out,
                )
                pairs_weights = [(s, t, conn.weight) for s, t in pairs]

            for src_local, tgt_local, w in pairs_weights:
                src_core, src_neuron = placement.neuron_map[(conn.source.id, src_local)]
                tgt_core, tgt_neuron = placement.neuron_map[(conn.target.id, tgt_local)]

                src_global = self._global_id(src_core, src_neuron)
                tgt_global = self._global_id(tgt_core, tgt_neuron)
                compiled.adjacency[src_global].append(
                    (tgt_global, w, conn.compartment, conn.delay))

                if src_core == tgt_core:
                    intra_conns[(src_core, src_neuron)].append(
                        (tgt_neuron, w, conn.compartment, conn.delay))
                    key = (src_core, src_neuron)
                    if key in src_format and src_format[key] != fmt_id:
                        src_format[key] = FMT_SPARSE
                    else:
                        src_format[key] = fmt_id
                else:
                    route_slots[(src_core, src_neuron)].append(
                        (tgt_core, tgt_neuron, w))

        pool_next_free = defaultdict(int)

        sorted_keys = sorted(intra_conns.keys())

        for core, src_neuron in sorted_keys:
            targets = intra_conns[(core, src_neuron)]
            format_id = src_format.get((core, src_neuron), FMT_SPARSE)

            if format_id == FMT_POP:
                pool_count = 1
            else:
                pool_count = len(targets)

            base_addr = pool_next_free[core]

            if base_addr + pool_count > self.pool_depth:
                raise PoolOverflowError(
                    f"Core {core} CSR pool exhausted: need {base_addr + pool_count} "
                    f"entries but pool_depth={self.pool_depth}. "
                    f"Neuron {src_neuron} has {len(targets)} connections at base {base_addr}.")

            if format_id == FMT_DENSE:
                targets_sorted = sorted(targets, key=lambda t: t[0])
                base_target = targets_sorted[0][0]

                compiled.prog_index_cmds.append({
                    "core": core, "neuron": src_neuron,
                    "base_addr": base_addr, "count": len(targets_sorted),
                    "format": FMT_DENSE,
                    "base_target": base_target,
                })

                for offset, (tgt_neuron, weight, comp, delay) in enumerate(targets_sorted):
                    compiled.prog_pool_cmds.append({
                        "core": core, "pool_addr": base_addr + offset,
                        "target": tgt_neuron, "weight": weight, "comp": comp,
                    })
                    if delay > 0:
                        compiled.prog_delay_cmds.append({
                            "core": core, "pool_addr": base_addr + offset,
                            "delay": delay,
                        })

            elif format_id == FMT_POP:
                shared_weight = targets[0][1]
                shared_comp = targets[0][2]
                base_target = min(t[0] for t in targets)

                compiled.prog_index_cmds.append({
                    "core": core, "neuron": src_neuron,
                    "base_addr": base_addr, "count": len(targets),
                    "format": FMT_POP,
                    "base_target": base_target,
                })

                compiled.prog_pool_cmds.append({
                    "core": core, "pool_addr": base_addr,
                    "target": base_target, "weight": shared_weight,
                    "comp": shared_comp,
                })
                for tgt_neuron, weight, comp, delay in targets:
                    if delay > 0:
                        compiled.prog_delay_cmds.append({
                            "core": core, "pool_addr": base_addr,
                            "delay": delay,
                        })
                        break

            else:
                compiled.prog_index_cmds.append({
                    "core": core, "neuron": src_neuron,
                    "base_addr": base_addr, "count": len(targets),
                    "format": FMT_SPARSE,
                })

                for offset, (tgt_neuron, weight, comp, delay) in enumerate(targets):
                    compiled.prog_pool_cmds.append({
                        "core": core, "pool_addr": base_addr + offset,
                        "target": tgt_neuron, "weight": weight, "comp": comp,
                    })
                    if delay > 0:
                        compiled.prog_delay_cmds.append({
                            "core": core, "pool_addr": base_addr + offset,
                            "delay": delay,
                        })

            pool_next_free[core] = base_addr + pool_count

        cluster_size = self.cluster_size

        for (src_core, src_neuron), dests in sorted(route_slots.items()):
            seen = {}
            for dest_core, dest_neuron, weight in dests:
                key = (dest_core, dest_neuron)
                if key not in seen:
                    seen[key] = weight

            unique_dests = list(seen.items())

            src_cluster = src_core // cluster_size
            local_dests = []
            global_dests = []
            for (dest_core, dest_neuron), weight in unique_dests:
                dest_cluster = dest_core // cluster_size
                if src_cluster == dest_cluster:
                    local_dests.append(((dest_core, dest_neuron), weight))
                else:
                    global_dests.append(((dest_core, dest_neuron), weight))

            if len(local_dests) > ROUTE_FANOUT:
                raise RouteOverflowError(
                    f"Source neuron (core {src_core}, neuron {src_neuron}) needs "
                    f"{len(local_dests)} local routes but ROUTE_FANOUT={ROUTE_FANOUT}.")

            if len(global_dests) > GLOBAL_ROUTE_SLOTS:
                raise RouteOverflowError(
                    f"Source neuron (core {src_core}, neuron {src_neuron}) needs "
                    f"{len(global_dests)} global routes but GLOBAL_ROUTE_SLOTS={GLOBAL_ROUTE_SLOTS}.")

            for slot, ((dest_core, dest_neuron), weight) in enumerate(local_dests):
                compiled.prog_route_cmds.append({
                    "src_core": src_core, "src_neuron": src_neuron,
                    "slot": slot,
                    "dest_core": dest_core, "dest_neuron": dest_neuron,
                    "weight": weight,
                })

            for slot, ((dest_core, dest_neuron), weight) in enumerate(global_dests):
                compiled.prog_global_route_cmds.append({
                    "src_core": src_core, "src_neuron": src_neuron,
                    "slot": slot,
                    "dest_core": dest_core, "dest_neuron": dest_neuron,
                    "weight": weight,
                })

    @staticmethod
    def _global_id(core, neuron):
        return core * NEURONS_PER_CORE + neuron

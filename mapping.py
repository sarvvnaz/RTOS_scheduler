"""Mapping nodes onto cores (OC-HEFT).

The algorithm has two steps.

**Step 1, the upward rank** decides the ORDER in which nodes are placed,
not where they go.

**Step 2, the placement** walks that order and gives each node the core
with the lowest cost

    Cost = w1 * Exec + w2 * Comm + w3 * RC

subject to two conditions that a core must satisfy to be eligible at all:

    FinishTime <= Deadline
    Utilization <= Umax = 1

The mapping happens offline and is final: a node never changes core at
run time.

    Rank(v) = c_v + max over successors s of ( comm(v, s) + Rank(s) )

Read it as: "how much work is still ahead of me before this task graph can
finish, counting myself?" A node with a large rank sits at the head of a
long remaining chain, so it is placed first and gets the best core.

The sink has no successors, so its rank is just its own execution time
(zero, because the sink is a dummy node). Every other rank builds on the
ranks behind it, which is why the nodes are walked in REVERSE topological
order: by the time we reach a node, all of its successors already have a
rank.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from config import Config
from model import Core, Node, Task, TaskSet


def upward_rank(task: Task) -> Dict[int, float]:
    """Upward rank of every node, keyed by node id.

    Walks the graph backwards, from the sink towards the source.
    """
    rank: Dict[int, float] = {}

    for node_id in reversed(list(nx.topological_sort(task.graph))):
        successors = list(task.graph.successors(node_id))

        if successors:
            # the longest way onwards, through whichever successor is worst
            ahead = max(task.comm_cost(node_id, s) + rank[s]
                        for s in successors)
        else:
            ahead = 0.0                      # the sink: nothing comes after

        rank[node_id] = task.nodes[node_id].wcet + ahead

    return rank


def placement_order(task: Task) -> List[int]:
    """Node ids sorted by decreasing rank: the order to place them in.

    The highest rank goes first, because that node begins the longest
    remaining chain of work and so constrains the schedule the most.
    """
    rank = upward_rank(task)
    return sorted(rank, key=lambda node_id: -rank[node_id])


def rank_path(task: Task) -> List[int]:
    """The chain the ranking considers most urgent.

    Starts at the source and always follows the successor with the highest
    rank. This is the path whose length the source's rank measures.
    """
    rank = upward_rank(task)
    path = [task.source_id]

    while True:
        successors = list(task.graph.successors(path[-1]))
        if not successors:
            return path
        path.append(max(successors, key=lambda s: rank[s]))


# ---------------------------------------------------------------------------
# Step 2: choosing a core
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    """Where one node ended up, and why."""

    task_id: int
    node_id: int
    core_id: Optional[int]          # None when nothing could take the node
    exec_cost: float = 0.0
    comm_cost: float = 0.0
    contention: float = 0.0
    cost: float = 0.0
    finish_time: float = 0.0
    rejected: Dict[int, str] = field(default_factory=dict)   # core -> reason

    @property
    def placed(self) -> bool:
        return self.core_id is not None


class Mapper:
    """Assigns every node of every task to a core, offline.

    The mapper keeps three running totals per core, because each one feeds
    a part of the decision:

    * ``utilization`` -- for the ``Utilization <= 1`` constraint
    * ``available``   -- when the core next goes idle, for the finish time
    * ``demand``      -- time spent holding each resource, for Usage_j(r)
    """

    def __init__(self, taskset: TaskSet, cfg: Config):
        self.taskset = taskset
        self.cfg = cfg
        self.w1, self.w2, self.w3 = cfg.cost_weights

        self.available: Dict[int, float] = {c.id: 0.0 for c in taskset.cores}
        # core id -> resource id -> total holding time over one hyperperiod
        self.demand: Dict[int, Dict[int, float]] = {
            c.id: {r.id: 0.0 for r in taskset.resources} for c in taskset.cores}

        self.finish: Dict[Tuple[int, int], float] = {}   # (task, node) -> EFT
        self.placements: List[Placement] = []

    # -- the three cost terms -------------------------------------------
    def exec_cost(self, node: Node, core: Core) -> float:
        """Exec(T_i, P_j) = WCET_i / Speed_j.

        The only reason an edge core is attractive: the same node finishes
        sooner there.
        """
        return core.exec_time(node.wcet)

    def comm_cost(self, task: Task, node: Node, core: Core) -> float:
        """What this placement would cost in communication.

        Only edges coming from already-placed predecessors count -- the
        successors have no core yet. An edge is free when both ends sit on
        the same core, so this term is what pulls a chain of dependent
        nodes together onto one core.
        """
        total = 0.0
        for pred in task.graph.predecessors(node.id):
            pred_core = task.nodes[pred].core_id
            if pred_core is not None and pred_core != core.id:
                total += task.comm_cost(pred, node.id)
        return total

    def usage(self, core: Core, resource_id: int) -> float:
        """Usage_j(r): the share of the hyperperiod core j already spends
        holding resource r.

        This is the same ratio as ``TaskSet.resource_usage_ratio``, but
        counted over one core instead of the whole system.
        """
        hyperperiod = self.taskset.hyperperiod
        if not hyperperiod:
            return 0.0
        return self.demand[core.id][resource_id] / hyperperiod

    def contention(self, node: Node, core: Core) -> float:
        """RC = sum over r of Demand_i(r) * Usage_j(r).

        High only when the node wants a resource that this core is already
        busy holding, so it steers nodes competing for the same resource
        onto different cores.
        """
        return sum(node.demand(r.id) * self.usage(core, r.id)
                   for r in self.taskset.resources)

    # -- the constraints -------------------------------------------------
    def node_utilization(self, task: Task, node: Node, core: Core) -> float:
        """How much of this core the node would take up.

        A faster core runs the node in less time, so the same node costs
        less utilization on an edge core than on a local one.
        """
        return self.exec_cost(node, core) / task.period

    def earliest_finish(self, task: Task, node: Node, core: Core) -> float:
        """When the node would finish on this core.

        It cannot start before every predecessor has finished and its data
        has arrived, nor before the core itself is free.
        """
        ready = 0.0
        for pred in task.graph.predecessors(node.id):
            pred_core = task.nodes[pred].core_id
            if pred_core is None:
                continue
            arrival = self.finish[(task.id, pred)]
            if pred_core != core.id:
                arrival += task.comm_cost(pred, node.id)
            ready = max(ready, arrival)

        start = max(ready, self.available[core.id])
        return start + self.exec_cost(node, core)

    def rejection_reason(self, task: Task, node: Node,
                         core: Core) -> Optional[str]:
        """Why this core may not take the node, or None if it may.

        These are the two ``subject to`` conditions. A dummy node takes no
        time, so it can never break either one.
        """
        if node.is_dummy:
            return None

        if core.utilization + self.node_utilization(task, node, core) > 1.0:
            return "utilization > 1"

        if self.earliest_finish(task, node, core) > task.deadline:
            return "finish time > deadline"

        return None

    # -- the decision ----------------------------------------------------
    def place(self, task: Task, node: Node) -> Placement:
        """Give one node the cheapest core that satisfies both conditions."""
        result = Placement(task_id=task.id, node_id=node.id, core_id=None)
        best = None

        for core in self.taskset.cores:
            reason = self.rejection_reason(task, node, core)
            if reason is not None:
                result.rejected[core.id] = reason
                continue

            e = self.exec_cost(node, core)
            c = self.comm_cost(task, node, core)
            rc = self.contention(node, core)
            cost = self.w1 * e + self.w2 * c + self.w3 * rc

            if best is None or cost < best[0]:
                best = (cost, core, e, c, rc)

        if best is None:
            return result                      # no core can take this node

        cost, core, e, c, rc = best
        result.core_id = core.id
        result.exec_cost, result.comm_cost, result.contention = e, c, rc
        result.cost = cost
        result.finish_time = self.earliest_finish(task, node, core)
        self._commit(task, node, core, result.finish_time)
        return result

    def _commit(self, task: Task, node: Node, core: Core,
                finish_time: float) -> None:
        """Record the placement and update everything that depends on it."""
        node.core_id = core.id
        core.assign(task.id, node.id, self.node_utilization(task, node, core))

        self.available[core.id] = finish_time
        self.finish[(task.id, node.id)] = finish_time

        # the node will hold its resources on this core, once per release
        releases = self.taskset.hyperperiod / task.period
        for resource in self.taskset.resources:
            self.demand[core.id][resource.id] += \
                releases * node.demand(resource.id)

    def run(self) -> List[Placement]:
        """Map every node of every task.

        Tasks are handled one after another; inside a task the nodes go in
        placement order, so a node is only ever placed after the nodes it
        depends on.

        ``available`` is reset between tasks. The tasks are periodic and
        each one is judged against its own deadline (D_i = T_i), so a node
        must fit inside its own task's release rather than queue behind
        some other task on a single global timeline. What stops a core
        being handed more work than it can carry across tasks is the
        utilization constraint, not this timeline.
        """
        for task in self.taskset.tasks:
            self.available = {c.id: 0.0 for c in self.taskset.cores}
            for node_id in placement_order(task):
                self.placements.append(self.place(task, task.nodes[node_id]))
        return self.placements

    # -- results ---------------------------------------------------------
    @property
    def unplaced(self) -> List[Placement]:
        return [p for p in self.placements if not p.placed]

    @property
    def succeeded(self) -> bool:
        """True when every node found a core.

        A task set that fails here is not schedulable by this mapping.
        """
        return not self.unplaced


def map_taskset(taskset: TaskSet, cfg: Config) -> Mapper:
    """Run OC-HEFT over a whole task set."""
    mapper = Mapper(taskset, cfg)
    mapper.run()
    return mapper

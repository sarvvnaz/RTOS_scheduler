"""The classes that describe the system: Segment, Node, Task, Resource,
Core and TaskSet."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx
import numpy as np


@dataclass
class Segment:
    """One piece of a node's execution.

    A critical segment holds ``resource_id`` exclusively for its whole
    length; a normal segment uses no resource.
    """

    length: float
    is_critical: bool = False
    resource_id: Optional[int] = None

    def __repr__(self):
        if self.is_critical:
            return f"CS(R{self.resource_id}, {self.length:.2f})"
        return f"NS({self.length:.2f})"


@dataclass
class Node:
    """One vertex of a task graph."""

    id: int
    utilization: float = 0.0        # u, always below 1
    wcet: float = 0.0               # execution time = u * T, at speed 1.0
    is_dummy: bool = False          # True for the source and the sink
    segments: List[Segment] = field(default_factory=list)
    core_id: Optional[int] = None   # set by the mapping step (OC-HEFT)

    @property
    def resources_used(self) -> List[int]:
        """Resource ids this node locks, in execution order."""
        return [s.resource_id for s in self.segments if s.is_critical]

    @property
    def num_critical_sections(self) -> int:
        return len(self.resources_used)

    @property
    def critical_time(self) -> float:
        return sum(s.length for s in self.segments if s.is_critical)

    def demand(self, resource_id: int) -> float:
        """Total time this node holds the given resource.

        This is Demand_i(r) in the OC-HEFT contention cost.
        """
        return sum(s.length for s in self.segments
                   if s.is_critical and s.resource_id == resource_id)

    def __repr__(self):
        return f"Node({self.id}, wcet={self.wcet:.2f}, cs={self.num_critical_sections})"


@dataclass
class Task:
    """A periodic DAG task. The deadline equals the period."""

    id: int
    period: float                                    # T
    utilization: float                               # u
    graph: nx.DiGraph
    nodes: Dict[int, Node]
    source_id: int
    sink_id: int

    @property
    def deadline(self) -> float:
        """D = T (implicit deadline)."""
        return self.period

    def comm_cost(self, u: int, v: int) -> float:
        """Communication cost of the edge u -> v.

        This is the *nominal* cost. It is only really paid when the two
        nodes end up on different cores; on the same core it is zero.
        """
        return self.graph.edges[u, v].get("comm", 0.0)

    @property
    def wcet(self) -> float:
        """C = total execution time of all nodes."""
        return sum(n.wcet for n in self.nodes.values())

    @property
    def critical_path_length(self) -> float:
        """L = the longest path from the source to the sink.

        Walks the nodes in topological order, keeping the longest distance
        that reaches each one.
        """
        dist = {}
        for v in nx.topological_sort(self.graph):
            before = [dist[p] for p in self.graph.predecessors(v)]
            dist[v] = (max(before) if before else 0.0) + self.nodes[v].wcet
        return dist[self.sink_id]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def real_nodes(self) -> List[Node]:
        """Every node except the source and the sink."""
        return [n for n in self.nodes.values() if not n.is_dummy]

    @property
    def resources_used(self) -> List[int]:
        """Distinct resources this task accesses."""
        return sorted({r for n in self.real_nodes() for r in n.resources_used})

    def access_count(self, resource_id: int) -> int:
        """How many times this task locks the given resource."""
        return sum(n.resources_used.count(resource_id) for n in self.real_nodes())

    def __repr__(self):
        return f"Task({self.id}, T={self.period:.0f}, |V|={self.num_nodes})"


@dataclass
class Resource:
    """A shared resource. Only one task may hold it at a time."""

    id: int
    total_accesses: int                                   # N_q
    accesses_per_task: Dict[int, int] = field(default_factory=dict)
    is_free: bool = True                                  # used by the protocol later

    def lock(self) -> bool:
        """Take the resource if it is free."""
        if not self.is_free:
            return False
        self.is_free = False
        return True

    def release(self) -> None:
        self.is_free = True

    def __repr__(self):
        return f"Resource(R{self.id}, N={self.total_accesses})"


@dataclass
class Core:
    """One processor, either on the local machine or on an edge server.

    ``server_id`` is None for a local core, otherwise the number of the
    edge server it belongs to. Edge cores have a higher ``speed``, which
    makes the same node run faster there.

    Only one node runs on a core at a time, so ``running`` is a single
    value, not a list. The queue and ``running`` stay empty until the
    scheduler is added.
    """

    id: int
    speed: float = 1.0
    server_id: Optional[int] = None                  # None = local core

    # filled by the mapping step
    assigned_nodes: List[tuple] = field(default_factory=list)   # (task_id, node_id)
    utilization: float = 0.0

    # filled by the scheduler
    execution_queue: List[tuple] = field(default_factory=list)
    running: Optional[tuple] = None

    @property
    def is_edge(self) -> bool:
        return self.server_id is not None

    @property
    def name(self) -> str:
        return f"E{self.server_id}.{self.id}" if self.is_edge else f"L{self.id}"

    def exec_time(self, wcet: float) -> float:
        """Exec(T_i, P_j) = WCET_i / Speed_j."""
        return wcet / self.speed

    def assign(self, task_id: int, node_id: int, utilization: float) -> None:
        self.assigned_nodes.append((task_id, node_id))
        self.utilization += utilization

    def __repr__(self):
        return f"Core({self.name}, speed={self.speed}, u={self.utilization:.3f})"


@dataclass
class TaskSet:
    """Everything that was generated: tasks, resources and cores."""

    tasks: List[Task]
    resources: List[Resource]
    cores: List[Core]

    @property
    def total_utilization(self) -> float:
        return sum(t.utilization for t in self.tasks)

    @property
    def u_norm(self) -> float:
        """U divided by the number of *local* cores.

        U_norm is defined as the load of the local machine, so the edge
        cores must not be counted here or the number would shrink as soon
        as edge servers are added.
        """
        local = len(self.local_cores)
        return self.total_utilization / local if local else 0.0

    @property
    def total_nodes(self) -> int:
        return sum(t.num_nodes for t in self.tasks)

    @property
    def total_critical_sections(self) -> int:
        return sum(n.num_critical_sections
                   for t in self.tasks for n in t.real_nodes())

    # -- platform ------------------------------------------------------
    @property
    def local_cores(self) -> List[Core]:
        return [c for c in self.cores if not c.is_edge]

    @property
    def edge_cores(self) -> List[Core]:
        return [c for c in self.cores if c.is_edge]

    @property
    def edge_server_ids(self) -> List[int]:
        return sorted({c.server_id for c in self.edge_cores})

    def core(self, core_id: int) -> Core:
        return self.cores[core_id]

    @property
    def hyperperiod(self) -> float:
        """Least common multiple of the task periods.

        The whole schedule repeats after this long, so simulations and the
        resource-usage percentages are measured over one hyperperiod.
        """
        periods = [int(round(t.period)) for t in self.tasks]
        return float(np.lcm.reduce(periods)) if periods else 0.0

    def resource_demand(self, resource_id: int) -> float:
        """Total time all nodes hold this resource, over one hyperperiod.

        Each task runs hyperperiod / period times, so its demand counts
        that many times.
        """
        total = 0.0
        for task in self.tasks:
            releases = self.hyperperiod / task.period
            per_release = sum(n.demand(resource_id) for n in task.real_nodes())
            total += releases * per_release
        return total

    def resource_usage_ratio(self, resource_id: int) -> float:
        """Share of the hyperperiod spent holding this resource.

        This is the "R1 = 60%" figure from the project definition.
        """
        return (self.resource_demand(resource_id) / self.hyperperiod
                if self.hyperperiod else 0.0)

    def check(self) -> None:
        """Make sure the generated task set really follows the rules."""
        for task in self.tasks:
            assert nx.is_directed_acyclic_graph(task.graph), "graph has a cycle"

            # a chain longer than the deadline is impossible on any number
            # of cores, so it must never reach the scheduler
            assert task.critical_path_length <= task.deadline, \
                f"task {task.id} has a critical path longer than its deadline"

            roots = [v for v in task.graph if task.graph.in_degree(v) == 0]
            leaves = [v for v in task.graph if task.graph.out_degree(v) == 0]
            assert roots == [task.source_id], "graph needs exactly one source"
            assert leaves == [task.sink_id], "graph needs exactly one sink"

            total_u = sum(n.utilization for n in task.nodes.values())
            assert abs(total_u - task.utilization) < 1e-6, \
                "node utilizations do not add up to the task utilization"

            for node in task.nodes.values():
                if node.is_dummy:
                    assert node.wcet == 0.0, "source and sink take no time"
                    continue
                assert node.utilization < 1.0, "a node utilization must be below 1"
                assert abs(sum(s.length for s in node.segments) - node.wcet) < 1e-6, \
                    "segments do not add up to the node execution time"
                # non-nested: a normal section always separates two critical ones
                for a, b in zip(node.segments, node.segments[1:]):
                    assert not (a.is_critical and b.is_critical), \
                        "two critical sections next to each other"

        for res in self.resources:
            assigned = sum(res.accesses_per_task.values())
            assert assigned == res.total_accesses, "wrong number of accesses"
            real = sum(t.access_count(res.id) for t in self.tasks)
            assert real == res.total_accesses, "accesses do not match the nodes"

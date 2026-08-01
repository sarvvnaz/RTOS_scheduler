"""The classes that describe the system: Segment, Node, Task, Resource,
Core and TaskSet."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx


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
    wcet: float = 0.0               # execution time = u * T
    is_dummy: bool = False          # True for the source and the sink
    segments: List[Segment] = field(default_factory=list)

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
    core_id: Optional[int] = None                    # set by the partitioner later

    @property
    def deadline(self) -> float:
        """D = T (implicit deadline)."""
        return self.period

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
    """One processor.

    Only one task runs on a core at a time, so ``running_task`` is a single
    value, not a list. The queue and the running task stay empty until the
    scheduler is added.
    """

    id: int
    assigned_tasks: List[int] = field(default_factory=list)
    utilization: float = 0.0
    execution_queue: List[int] = field(default_factory=list)
    running_task: Optional[int] = None

    def assign(self, task_id: int, utilization: float) -> None:
        self.assigned_tasks.append(task_id)
        self.utilization += utilization

    def __repr__(self):
        return f"Core({self.id}, u={self.utilization:.3f})"


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
        return self.total_utilization / len(self.cores)

    @property
    def total_nodes(self) -> int:
        return sum(t.num_nodes for t in self.tasks)

    @property
    def total_critical_sections(self) -> int:
        return sum(n.num_critical_sections
                   for t in self.tasks for n in t.real_nodes())

    def check(self) -> None:
        """Make sure the generated task set really follows the rules."""
        for task in self.tasks:
            assert nx.is_directed_acyclic_graph(task.graph), "graph has a cycle"

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

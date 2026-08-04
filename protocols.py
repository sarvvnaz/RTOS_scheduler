"""Shared-resource protocols.

A node that wants a resource another core is holding has to wait. The two
protocols differ in what it does while waiting:

* **MSRP** -- it *spins*. The core is held for the whole wait, so waiting
  shows up as extra work on that core.
* **POMIP** -- it *suspends*. The core is released, so the wait does not
  burn it, but each request pays context switches and a lock holder may
  migrate to another cluster to make progress.

That difference is the comparison the project asks for: spinning wastes
processor time but is simple and predictable; suspending frees the
processor but pays overhead on every request.

POMIP comes from Wang, Jiang, Guan, Tang and Liu, "Locking Protocols for
Parallel Real-Time Tasks With Semaphores Under Federated Scheduling"
(IEEE TCAD 41(9), 2022). The paper analyses *federated* scheduling, where
a heavy task owns m_i dedicated processors. Here the tasks are mapped by
OC-HEFT and run under partitioned EDF, so we read m_i as **the number of
distinct cores a task's nodes were mapped onto** -- the cluster OC-HEFT
happened to give it. Everything else follows the paper.

Blocking is computed *after* mapping: until a node has a core there is no
way to tell a remote request from a local one.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from model import Node, Task, TaskSet


class Protocol(Enum):
    """Which protocol the analysis is run under."""

    MSRP = "msrp"            # spin-based
    POMIP = "pomip"          # suspension-based, migratory priority inheritance
    ADAPTIVE = "adaptive"    # picks between them -- not decided yet


@dataclass
class Blocking:
    """What one node waits for, split by how it waits."""

    task_id: int
    node_id: int
    spin: float = 0.0        # waiting that burns the core (MSRP)
    suspend: float = 0.0     # waiting that releases the core (POMIP)
    overhead: float = 0.0    # context switches (POMIP only)

    @property
    def total(self) -> float:
        return self.spin + self.suspend + self.overhead

    @property
    def on_core(self) -> float:
        """The part that occupies the core, so it inflates its load.

        Spinning holds the core; suspending gives it up. The context
        switches are paid on the core either way.
        """
        return self.spin + self.overhead


@dataclass
class TaskBlocking:
    """Task-level blocking, which is the level POMIP is defined at."""

    task_id: int
    po_blocking: float = 0.0        # sum over resources of max(F^I + F^O)
    response_time: float = 0.0      # R_i from equation (11)
    num_cores: int = 1              # m_i, the cluster OC-HEFT gave the task
    per_resource: Dict[int, float] = field(default_factory=dict)

    def meets_deadline(self, deadline: float) -> bool:
        return self.response_time <= deadline


class ResourceAnalysis:
    """Blocking for a mapped task set, under one protocol.

    Needs a finished ``mapping.Mapper``: which requests are remote depends
    on where the nodes ended up.
    """

    def __init__(self, taskset: TaskSet, mapper, protocol: Protocol,
                 context_switch: float = 0.0):
        if protocol is Protocol.ADAPTIVE:
            raise NotImplementedError(
                "the adaptive protocol is not decided yet; use MSRP or POMIP")

        self.taskset = taskset
        self.mapper = mapper
        self.protocol = protocol
        self.context_switch = context_switch
        self.blocking: Dict[tuple, Blocking] = {}
        self.task_blocking: Dict[int, TaskBlocking] = {}

    # -- quantities the paper is written in ------------------------------
    def cores_of(self, task: Task) -> int:
        """m_i: how many distinct cores this task's nodes landed on.

        The paper's federated model hands a task m_i dedicated processors;
        here OC-HEFT decides it instead.
        """
        cores = {n.core_id for n in task.real_nodes() if n.core_id is not None}
        return max(len(cores), 1)

    def max_cs(self, task: Task, resource_id: int) -> float:
        """L_{i,q}: the longest single critical section for this resource."""
        longest = 0.0
        for node in task.real_nodes():
            for segment in node.segments:
                if segment.is_critical and segment.resource_id == resource_id:
                    longest = max(longest, segment.length)
        return longest

    def num_requests(self, task: Task, resource_id: int) -> int:
        """N_{i,q}: how many times this task locks the resource."""
        return task.access_count(resource_id)

    def interference_count(self, task: Task, other: Task,
                           resource_id: int) -> int:
        """eta^q_{i,j} = ceil((D_i + D_j) / T_j) * N_{j,q}.

        How many requests of ``other`` can overlap one job of ``task``.
        """
        releases = math.ceil((task.deadline + other.deadline) / other.period)
        return releases * self.num_requests(other, resource_id)

    # -- MSRP: spinning ---------------------------------------------------
    def holders(self, resource_id: int) -> Dict[int, float]:
        """Longest critical section for a resource, per core.

        A core runs one node at a time, so a waiting node meets at most
        one holder per core -- the longest is the worst case.
        """
        longest: Dict[int, float] = {}
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                if node.core_id is None:
                    continue
                for segment in node.segments:
                    if segment.is_critical and segment.resource_id == resource_id:
                        current = longest.get(node.core_id, 0.0)
                        longest[node.core_id] = max(current, segment.length)
        return longest

    def spin_delay(self, node: Node, resource_id: int) -> float:
        """Worst-case spin for one request, under FIFO.

        The request queues behind at most one request from each *other*
        core using the resource; its own core cannot compete with itself.
        """
        per_core = self.holders(resource_id)
        return sum(length for core_id, length in per_core.items()
                   if core_id != node.core_id)

    # -- POMIP: suspending ------------------------------------------------
    def intra_task_blocking(self, task: Task, resource_id: int,
                            on_path: int) -> float:
        """F^I from Lemma 8: blocking by this task's own other vertices.

        ``on_path`` is N^lambda_{i,q}, the number of requests on the path
        being analysed. With none on the path the path is never suspended
        by this resource, so there is nothing to count.
        """
        if on_path <= 0:
            return 0.0
        total = self.num_requests(task, resource_id)
        m_i = self.cores_of(task)
        return (total - on_path) * (m_i - 1) * self.max_cs(task, resource_id)

    def inter_task_blocking(self, task: Task, resource_id: int,
                            on_path: int) -> float:
        """F^O from Lemma 10: blocking by every other task.

        Each other task contributes its own requests, plus -- once this
        path holds the resource at all -- the migratory term the paper
        bounds with Delta.
        """
        m_i = self.cores_of(task)
        own_requests = self.num_requests(task, resource_id)
        total = 0.0

        for other in self.taskset.tasks:
            if other.id == task.id:
                continue
            eta = self.interference_count(task, other, resource_id)
            delta = min(on_path, 1) * min(eta, own_requests)
            total += (eta + (m_i - 1) * delta) * self.max_cs(other, resource_id)

        return total

    def po_blocking(self, task: Task, resource_id: int) -> float:
        """The worst F^I + F^O over every possible N^lambda_{i,q}.

        Algorithm 1 of the paper enumerates this count from 0 up to the
        task's total number of requests, which is just a loop -- no need
        to enumerate the paths themselves.
        """
        total = self.num_requests(task, resource_id)
        return max((self.intra_task_blocking(task, resource_id, x)
                    + self.inter_task_blocking(task, resource_id, x)
                    for x in range(total + 1)), default=0.0)

    def response_time(self, task: Task) -> TaskBlocking:
        """R_i from equation (11), with the POMIP blocking bounds.

            R_i <= [ C_i + (m_i - 1) L_i + sum_q max(F^I + F^O) ] / m_i
        """
        m_i = self.cores_of(task)
        result = TaskBlocking(task_id=task.id, num_cores=m_i)

        for resource_id in task.resources_used:
            blocking = self.po_blocking(task, resource_id)
            result.per_resource[resource_id] = blocking
            result.po_blocking += blocking

        work = (task.wcet
                + (m_i - 1) * task.critical_path_length
                + result.po_blocking)
        result.response_time = work / m_i
        return result

    # -- running it -------------------------------------------------------
    def analyse_node(self, task: Task, node: Node) -> Blocking:
        """Blocking suffered by one node."""
        result = Blocking(task_id=task.id, node_id=node.id)
        if node.core_id is None:
            return result

        for resource_id in node.resources_used:
            if self.protocol is Protocol.MSRP:
                # spinning: the core stays busy for the whole wait
                result.spin += self.spin_delay(node, resource_id)
            else:
                # suspending: the core is free, but the switches are not
                result.overhead += 2 * self.context_switch

        return result

    def run(self) -> Dict[tuple, Blocking]:
        """Analyse every mapped node, and every task under POMIP."""
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                self.blocking[(task.id, node.id)] = self.analyse_node(task, node)

            if self.protocol is Protocol.POMIP:
                blocking = self.response_time(task)
                self.task_blocking[task.id] = blocking
                # the suspension is real waiting, even if it frees the core
                for node in task.real_nodes():
                    self.blocking[(task.id, node.id)].suspend = \
                        blocking.po_blocking / max(len(task.real_nodes()), 1)

        return self.blocking

    # -- results -----------------------------------------------------------
    def for_node(self, task_id: int, node_id: int) -> Blocking:
        return self.blocking.get((task_id, node_id),
                                 Blocking(task_id=task_id, node_id=node_id))

    def core_utilization(self, core_id: int) -> float:
        """Utilization of a core once blocking is counted.

        Only the part that occupies the core is added: spin time and
        context switches inflate the work the core must do, suspension
        does not.
        """
        core = self.taskset.core(core_id)
        total = 0.0
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                if node.core_id != core_id:
                    continue
                inflated = (core.exec_time(node.wcet)
                            + self.for_node(task.id, node.id).on_core)
                total += inflated / task.period
        return total

    @property
    def total_spin(self) -> float:
        return sum(b.spin for b in self.blocking.values())

    @property
    def total_suspend(self) -> float:
        return sum(b.suspend for b in self.blocking.values())

    @property
    def total_overhead(self) -> float:
        return sum(b.overhead for b in self.blocking.values())


def analyse(taskset: TaskSet, mapper, protocol: Protocol,
            context_switch: float = 0.0) -> ResourceAnalysis:
    """Run the blocking analysis over a mapped task set."""
    analysis = ResourceAnalysis(taskset, mapper, protocol, context_switch)
    analysis.run()
    return analysis

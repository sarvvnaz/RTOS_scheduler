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
    H2LP = "h2lp"            # hybrid: spin behind a per-cluster token
    ADAPTIVE = "adaptive"    # picks between them -- not decided yet


@dataclass
class Blocking:
    """What one node waits for, split by how it waits."""

    task_id: int
    node_id: int
    spin: float = 0.0        # remote waiting that burns the core (MSRP, H2LP)
    token: float = 0.0       # waiting for this cluster's CS token (H2LP)
    overhead: float = 0.0    # context switches (POMIP, H2LP)

    @property
    def total(self) -> float:
        return self.spin + self.token + self.overhead

    @property
    def on_core(self) -> float:
        """The part that occupies the core, so it inflates its load.

        Spinning holds the core; suspension gives it up, which is why it
        never appears here. Context switches are paid either way.

        Token waiting is a suspension too -- that is the whole point of
        the token -- so it is excluded for the same reason.

        Arrival blocking is not here because it is charged once per job,
        not once per node -- see ``local_blocking``.
        """
        return self.spin + self.overhead


@dataclass
class TaskBlocking:
    """Task-level blocking, which is the level POMIP is defined at."""

    task_id: int
    blocking: float = 0.0           # everything the task waits for
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
                "the adaptive protocol is not decided yet; "
                "use MSRP, POMIP or H2LP")

        self.taskset = taskset
        self.mapper = mapper
        self.protocol = protocol
        self.context_switch = context_switch
        self.blocking: Dict[tuple, Blocking] = {}
        self.task_blocking: Dict[int, TaskBlocking] = {}
        self._holders: Dict[int, Dict[int, float]] = {}   # cache, see holders()

    # -- quantities the paper is written in ------------------------------
    def cores_of(self, task: Task) -> int:
        """m_i: the size of the task's cluster.

        Under a federated allocation this is exactly the paper's m_i --
        cores the task owns and nobody else may touch -- so the response
        time bound is being used in the setting it was proved for.

        Without an allocation there is no cluster, so we fall back to the
        number of distinct cores the mapping happened to use. That reading
        is weaker: those cores are shared with other tasks, so dividing by
        it credits parallelism the task does not exclusively have.
        """
        allocation = getattr(self.mapper, "allocation", None)
        if allocation is not None and task.id in allocation.clusters:
            return len(allocation.clusters[task.id])

        # A light task owns nothing: it shares the leftover pool with the
        # other light tasks. Its m_i is what the mapping actually used, not
        # the size of that pool -- dividing by the whole pool would credit
        # it with every core it merely had access to.
        cores = {n.core_id for n in task.real_nodes() if n.core_id is not None}
        return max(len(cores), 1)

    # -- H2LP: spinning behind a per-cluster token ------------------------
    def cluster_of(self, core_id: int):
        """Which cluster a core belongs to.

        Heavy tasks own a cluster each; every core left over belongs to
        the one shared pool. Without a federated allocation there are no
        clusters, so each core stands alone -- which is exactly MSRP.
        """
        allocation = getattr(self.mapper, "allocation", None)
        if allocation is None:
            return ("core", core_id)

        for task_id, cores in allocation.clusters.items():
            if core_id in cores:
                return ("cluster", task_id)

        # A core in the leftover pool is a cluster of its own. The paper
        # schedules light tasks as sequential tasks, one processor each,
        # so their "cluster" is a single core -- which is exactly why it
        # says H2LP reduces to MSRP for light tasks.
        #
        # Treating the whole pool as one cluster instead makes every
        # light task look like it shares a token with all the others, and
        # the spin bound collapses to zero.
        return ("core", core_id)

    def h2lp_spin_delay(self, node: Node, resource_id: int) -> float:
        """Worst-case spin for one request under H2LP.

        This is where H2LP earns its keep. Under MSRP a request queues
        behind one request from every other *core* that uses the resource.
        Under H2LP a cluster holds one CS token per resource, so at most
        one vertex per *cluster* can be spinning for it at a time, and the
        queue is one deep per cluster instead of one deep per core.

        On a machine where a cluster holds many cores that is a large
        reduction, and it is the reason the protocol scales where a plain
        spin lock does not.
        """
        per_core = self.holders(resource_id)
        own = self.cluster_of(node.core_id) if node.core_id is not None else None

        worst_per_cluster: Dict[tuple, float] = {}
        for core_id, length in per_core.items():
            cluster = self.cluster_of(core_id)
            if cluster == own:
                continue                  # own cluster is the token's job
            worst_per_cluster[cluster] = max(
                worst_per_cluster.get(cluster, 0.0), length)

        return sum(worst_per_cluster.values())

    def token_delay(self, task: Task, node: Node, resource_id: int) -> float:
        """Waiting for this cluster's CS token, for one request.

        The token is what stops several vertices of the same task spinning
        for one resource at once. The price is that a vertex may have to
        wait for its own siblings to finish with it first, and that wait
        is a suspension rather than a spin.

        Only a heavy task pays this: a light task has a cluster to itself
        in the analysis, so a token is always free for it, which is why
        the paper says H2LP reduces to MSRP for light tasks.
        """
        if not self.is_heavy(task):
            return 0.0

        siblings = max(self.num_requests(task, resource_id) - 1, 0)
        if not siblings:
            return 0.0

        # at most the other requests of this task, each held for its own
        # longest critical section
        return siblings * self.max_cs(task, resource_id)

    def is_heavy(self, task: Task) -> bool:
        """Whether the task owns a cluster, in this allocation."""
        allocation = getattr(self.mapper, "allocation", None)
        if allocation is None:
            return False
        return task.id in allocation.clusters

    def cluster_speed(self, task: Task) -> float:
        """How fast the cores this task runs on are.

        C_i and L_i are stated at speed 1.0, but a task whose cluster is
        made of edge cores really does finish that work twice as fast.
        The slowest core in the cluster is the safe one to use.
        """
        cores = [self.taskset.core(n.core_id) for n in task.real_nodes()
                 if n.core_id is not None]
        return min((c.speed for c in cores), default=1.0)

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

        Cached: the mapping does not change while an analysis runs, and
        this is asked once per request, which is often thousands of times.
        """
        cached = self._holders.get(resource_id)
        if cached is not None:
            return cached

        longest: Dict[int, float] = {}
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                if node.core_id is None:
                    continue
                for segment in node.segments:
                    if segment.is_critical and segment.resource_id == resource_id:
                        current = longest.get(node.core_id, 0.0)
                        longest[node.core_id] = max(current, segment.length)

        self._holders[resource_id] = longest
        return longest

    def spin_delay(self, node: Node, resource_id: int) -> float:
        """Worst-case spin for one request, under FIFO.

        The request queues behind at most one request from each *other*
        core using the resource; its own core cannot compete with itself.
        """
        per_core = self.holders(resource_id)
        return sum(length for core_id, length in per_core.items()
                   if core_id != node.core_id)

    def local_blocking(self, task: Task, core_id: int) -> float:
        """Blocking by another task already spinning on this core.

        Spinning under MSRP is non-preemptable, so a job arriving on a
        core can be held up by a job of *another* task already inside a
        critical section there. Only the longest such section matters.

        Charged **once per job per core the task uses**, not once per
        node: it is an arrival effect, and a node dispatch is not a new
        arrival. Charging it per node made a core hosting 49 nodes pay it
        49 times and pushed utilization from 0.9 to 1.54 on its own.

        Without this term MSRP looks free whenever a core hosts several
        tasks, which is exactly what the mapping tends to produce.
        """
        longest = 0.0
        for other in self.taskset.tasks:
            if other.id == task.id:
                continue
            for candidate in other.real_nodes():
                if candidate.core_id != core_id:
                    continue
                for segment in candidate.segments:
                    if segment.is_critical:
                        longest = max(longest, segment.length)
        return longest

    def cores_used_by(self, task: Task) -> set:
        """The distinct cores a task's nodes were mapped onto."""
        return {n.core_id for n in task.real_nodes() if n.core_id is not None}

    def task_local_blocking(self, task: Task) -> float:
        """Arrival blocking, over every core the task actually uses.

        This is B^A. Under MSRP every task can suffer it. Under H2LP only
        a *light* task can: a heavy task owns its cluster outright, so no
        other task is ever running there to block its arrival -- which is
        exactly what the paper states. POMIP suspends instead of holding
        the core, so it has no arrival blocking of this kind.
        """
        if self.protocol is Protocol.MSRP:
            pass
        elif self.protocol is Protocol.H2LP and not self.is_heavy(task):
            pass
        else:
            return 0.0

        return sum(self.local_blocking(task, core_id)
                   for core_id in self.cores_used_by(task))

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

        Algorithm 1 enumerates this count from 0 up to the task's total
        number of requests, but only two of those values can ever win:

        * F^I is ``(N - x)(m-1)L``, which *decreases* in x, so among all
          x >= 1 the largest is x = 1
        * F^O depends on x only through ``min(x, 1)``, so it is constant
          for every x >= 1

        That leaves x = 0 and x = 1. Checking those two is exactly the
        same answer as the full sweep, and does not get slower when a
        resource is requested 150 times.
        """
        total = self.num_requests(task, resource_id)
        if total <= 0:
            return 0.0

        return max(self.intra_task_blocking(task, resource_id, x)
                   + self.inter_task_blocking(task, resource_id, x)
                   for x in (0, 1))

    def response_time(self, task: Task) -> TaskBlocking:
        """R_i from equation (11), under either protocol.

            R_i <= [ C_i + (m_i - 1) L_i + blocking ] / m_i

        The shape comes from the paper and is the same for both; only the
        blocking term differs, which is the point of the comparison:

        * POMIP -- the path-oriented bound, sum over resources of the
          worst F^I + F^O, plus the context switches every request pays
        * MSRP  -- the spinning and the arrival blocking its own nodes
          suffer, which is the equivalent quantity for a spin lock
        * H2LP  -- the same shape as the paper's Corollary 4.9,
          ``L_i + B^S + B^T + B^A + I_i / n_i``: spin blocking counted per
          cluster rather than per core, token blocking for a heavy task,
          arrival blocking for a light one

        Every protocol is judged by this same test. Running it for one
        alone would fail it for a reason the others are never asked about.
        """
        m_i = self.cores_of(task)
        result = TaskBlocking(task_id=task.id, num_cores=m_i)

        if self.protocol is Protocol.POMIP:
            for resource_id in task.resources_used:
                blocking = self.po_blocking(task, resource_id)
                result.per_resource[resource_id] = blocking
                result.blocking += blocking
            # the switches are real work and belong in the bound too
            result.blocking += sum(self.for_node(task.id, n.id).overhead
                                   for n in task.real_nodes())
        else:
            result.blocking = (sum(self.for_node(task.id, n.id).total
                                   for n in task.real_nodes())
                               + self.task_local_blocking(task))

        # C_i and L_i are quoted at speed 1.0; the cluster may run faster.
        # The blocking term is left unscaled: it is caused by other tasks
        # on their own cores, so this task's speed does not shorten it.
        speed = self.cluster_speed(task)
        work = (task.wcet / speed
                + (m_i - 1) * task.critical_path_length / speed
                + result.blocking)
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

            elif self.protocol is Protocol.H2LP:
                # hybrid: spin once the token is held, suspend before that
                result.spin += self.h2lp_spin_delay(node, resource_id)
                waiting = self.token_delay(task, node, resource_id)
                result.token += waiting
                if waiting:
                    result.overhead += 2 * self.context_switch

            else:
                # suspending: the core is free, but the switches are not
                result.overhead += 2 * self.context_switch

        return result

    def run(self) -> Dict[tuple, Blocking]:
        """Analyse every mapped node, then every task."""
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                self.blocking[(task.id, node.id)] = self.analyse_node(task, node)

        # the task-level bound reads the node results, so it comes second
        for task in self.taskset.tasks:
            self.task_blocking[task.id] = self.response_time(task)

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
            here = [n for n in task.real_nodes() if n.core_id == core_id]
            if not here:
                continue

            for node in here:
                inflated = (core.exec_time(node.wcet)
                            + self.for_node(task.id, node.id).on_core)
                total += inflated / task.period

            # arrival blocking is per job on this core, not per node
            total += self.local_blocking(task, core_id) / task.period

        return total

    @property
    def total_spin(self) -> float:
        return sum(b.spin for b in self.blocking.values())

    @property
    def total_local(self) -> float:
        """Arrival blocking, which lives at task level, not node level."""
        return sum(self.task_local_blocking(t) for t in self.taskset.tasks)

    @property
    def total_token(self) -> float:
        """Time spent waiting for a CS token (H2LP only)."""
        return sum(b.token for b in self.blocking.values())

    @property
    def total_overhead(self) -> float:
        return sum(b.overhead for b in self.blocking.values())

    @property
    def total_blocking(self) -> float:
        """Everything every task waits for, at the level it is defined."""
        return sum(b.blocking for b in self.task_blocking.values())


def analyse(taskset: TaskSet, mapper, protocol: Protocol,
            context_switch: float = 0.0) -> ResourceAnalysis:
    """Run the blocking analysis over a mapped task set."""
    analysis = ResourceAnalysis(taskset, mapper, protocol, context_switch)
    analysis.run()
    return analysis

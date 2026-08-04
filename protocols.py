"""Shared-resource protocols.

A node that wants a resource another core is holding has to wait. The two
protocols differ only in what it does while waiting:

* **MSRP** -- it *spins*. The core is held the whole time, so the waiting
  shows up as extra execution time on that core.
* **LPP-P** -- it *suspends*. The core is released to other work, so the
  waiting does not consume the core, but each request pays two context
  switches.

That single difference is the whole comparison the project asks for: spin
wastes processor time but is cheap and predictable; suspension frees the
processor but pays overhead every time.

Blocking is computed *after* mapping, because until a node has a core we
do not know which requests are remote and which are local.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from model import Node, Task, TaskSet


class Protocol(Enum):
    """Which protocol the analysis is run under."""

    MSRP = "msrp"            # spin-based
    LPP_P = "lpp-p"          # suspension-based
    ADAPTIVE = "adaptive"    # picks per resource -- not implemented yet


@dataclass
class Blocking:
    """What one node has to wait for, split by where it comes from."""

    task_id: int
    node_id: int
    spin: float = 0.0        # waiting that burns the core (MSRP)
    suspend: float = 0.0     # waiting that releases the core (LPP-P)
    overhead: float = 0.0    # context switches (LPP-P only)

    @property
    def total(self) -> float:
        """Everything the node waits for, however it waits."""
        return self.spin + self.suspend + self.overhead

    @property
    def on_core(self) -> float:
        """The part that occupies the core, so it inflates execution time.

        Spinning holds the core; suspending gives it up. The context
        switches are paid on the core either way.
        """
        return self.spin + self.overhead


class ResourceAnalysis:
    """Blocking for every node of a mapped task set, under one protocol.

    Needs a finished ``mapping.Mapper``: which requests are remote depends
    on where the nodes ended up.
    """

    def __init__(self, taskset: TaskSet, mapper, protocol: Protocol,
                 context_switch: float = 0.0):
        if protocol is Protocol.ADAPTIVE:
            raise NotImplementedError(
                "the adaptive protocol is not decided yet; use MSRP or LPP-P")

        self.taskset = taskset
        self.mapper = mapper
        self.protocol = protocol
        self.context_switch = context_switch
        self.blocking: Dict[tuple, Blocking] = {}

    # -- who holds what, and where --------------------------------------
    def holders(self, resource_id: int) -> Dict[int, float]:
        """The longest single critical section on each core, per resource.

        A core can only run one node at a time, so however many nodes on a
        core want the resource, a waiting node meets at most one of them
        at a time -- the longest one is the worst case.
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

    def remote_delay(self, node: Node, resource_id: int) -> float:
        """Worst-case wait for one request to a resource held elsewhere.

        Under FIFO, a request queues behind at most one request from every
        *other* core that uses the resource -- its own core cannot compete
        with itself, because only one node runs there at a time.
        """
        per_core = self.holders(resource_id)
        return sum(length for core_id, length in per_core.items()
                   if core_id != node.core_id)

    # -- the two protocols ----------------------------------------------
    def analyse_node(self, task: Task, node: Node) -> Blocking:
        """Total blocking this node suffers, under the chosen protocol."""
        result = Blocking(task_id=task.id, node_id=node.id)
        if node.core_id is None:
            return result

        for resource_id in node.resources_used:
            delay = self.remote_delay(node, resource_id)

            if self.protocol is Protocol.MSRP:
                # spinning: the core is busy for the whole wait
                result.spin += delay
            else:
                # suspending: the core is free, but the switch is not
                result.suspend += delay
                result.overhead += 2 * self.context_switch

        return result

    def run(self) -> Dict[tuple, Blocking]:
        """Analyse every mapped node."""
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                key = (task.id, node.id)
                self.blocking[key] = self.analyse_node(task, node)
        return self.blocking

    # -- results ---------------------------------------------------------
    def for_node(self, task_id: int, node_id: int) -> Blocking:
        return self.blocking.get((task_id, node_id),
                                 Blocking(task_id=task_id, node_id=node_id))

    def core_utilization(self, core_id: int) -> float:
        """Utilization of a core once blocking is counted.

        Only the part that occupies the core is added: spin time and
        context switches inflate the work the core must do, suspension
        does not.
        """
        total = 0.0
        for task in self.taskset.tasks:
            for node in task.real_nodes():
                if node.core_id != core_id:
                    continue
                core = self.taskset.core(core_id)
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

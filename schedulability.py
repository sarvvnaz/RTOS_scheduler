"""Is a mapped task set schedulable under partitioned EDF?

Every core runs its own EDF queue over the nodes OC-HEFT gave it, and a
node never moves once assigned. For EDF on a single processor the classic
condition is that the total utilization does not exceed 1, so with shared
resources the test becomes, on each core,

    sum over its nodes of (Exec + blocking that occupies the core) / T <= 1

Which blocking occupies the core is exactly what separates the protocols:
MSRP spins so the wait is on the core, POMIP suspends so it is not.

A task set passes only if all three hold:

  1. every node found a core at all (the mapping succeeded)
  2. every core passes the EDF utilization test
  3. every task meets its deadline by the response-time bound

The third test is applied to *both* protocols, with each one's own
blocking term. Running it for one protocol only would fail that protocol
for a reason the other is never asked about, which is not a comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from model import TaskSet
from protocols import Protocol, ResourceAnalysis


@dataclass
class Result:
    """The verdict for one task set, and why."""

    schedulable: bool
    mapped: bool                                  # did every node get a core
    protocol: Protocol
    core_utilization: Dict[int, float] = field(default_factory=dict)
    overloaded: List[int] = field(default_factory=list)     # core ids over 1
    missed: List[int] = field(default_factory=list)         # task ids over D
    unplaced_nodes: int = 0
    allocation: object = None                     # federated.Allocation, if any

    # -- diagnostics: what the inputs looked like, not just the verdict --
    nodes_total: int = 0
    nodes_on_edge: int = 0
    heavy_tasks: int = 0
    critical_path_ratio: float = 0.0    # mean L / D over the tasks
    comm_ratio: float = 0.0             # mean cost of one edge / D
    tasks_total: int = 0

    @property
    def reason(self) -> str:
        """One line saying what went wrong, for the report."""
        if self.schedulable:
            return "schedulable"
        if self.allocation is not None and not self.allocation.feasible:
            return self.allocation.reason
        if not self.mapped:
            return f"{self.unplaced_nodes} node(s) could not be mapped"
        parts = []
        if self.overloaded:
            parts.append(f"{len(self.overloaded)} core(s) over utilization 1")
        if self.missed:
            parts.append(f"{len(self.missed)} task(s) miss their deadline")
        return ", ".join(parts) or "not schedulable"

    @property
    def max_utilization(self) -> float:
        return max(self.core_utilization.values(), default=0.0)

    @property
    def quality_of_service(self) -> float:
        """Share of tasks that do meet their deadline.

        A task set that fails is rarely all-or-nothing, so this is the
        softer measure the report plots next to schedulability.
        """
        return self._qos

    _qos: float = 0.0


def test(taskset: TaskSet, mapper, analysis: ResourceAnalysis,
         allocation=None) -> Result:
    """Run the partitioned-EDF test over a mapped and analysed task set."""
    result = Result(schedulable=False,
                    mapped=mapper.succeeded,
                    protocol=analysis.protocol,
                    unplaced_nodes=len(mapper.unplaced),
                    allocation=allocation)

    # 1. per-core EDF utilization, blocking included
    for core in taskset.cores:
        if not core.assigned_nodes:
            continue
        utilization = analysis.core_utilization(core.id)
        result.core_utilization[core.id] = utilization
        if utilization > 1.0 + 1e-9:
            result.overloaded.append(core.id)

    overloaded = set(result.overloaded)

    # 2. a verdict per task, so a partly failing set is not all-or-nothing
    for task in taskset.tasks:
        cores_used = {n.core_id for n in task.real_nodes()}

        if None in cores_used:
            result.missed.append(task.id)       # some node found no core
            continue

        if cores_used & overloaded:
            result.missed.append(task.id)       # shares an overloaded core
            continue

        # the response-time bound, which both protocols now face
        blocking = analysis.task_blocking.get(task.id)
        if blocking and not blocking.meets_deadline(task.deadline):
            result.missed.append(task.id)

    met = len(taskset.tasks) - len(result.missed)
    result._qos = met / len(taskset.tasks) if taskset.tasks else 0.0
    result.schedulable = (mapper.succeeded
                          and not result.overloaded
                          and not result.missed)
    _describe(taskset, allocation, result)
    return result


def _describe(taskset: TaskSet, allocation, result: Result) -> None:
    """Record what the input looked like, so a flat curve can be explained.

    A verdict on its own cannot tell you whether a chart is flat because
    the scheduling is hard or because the task sets were impossible from
    the start.
    """
    ratios = []
    comms = []

    for task in taskset.tasks:
        ratios.append(task.critical_path_length / task.deadline)
        edges = [task.comm_cost(u, v) for u, v in task.graph.edges]
        if edges:
            comms.append(sum(edges) / len(edges) / task.deadline)

        for node in task.real_nodes():
            result.nodes_total += 1
            if node.core_id is not None and taskset.core(node.core_id).is_edge:
                result.nodes_on_edge += 1

    result.tasks_total = len(taskset.tasks)
    result.heavy_tasks = len(allocation.heavy) if allocation else 0
    result.critical_path_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    result.comm_ratio = sum(comms) / len(comms) if comms else 0.0


def evaluate(cfg, protocol: Protocol) -> Result:
    """Generate, map, analyse and test one task set in one call.

    This is the unit the experiments repeat: everything from a config to a
    yes/no answer.

    With ``cfg.federated`` the tasks are split into heavy and light first
    and each heavy task is given its own cluster; OC-HEFT then maps inside
    those clusters. An allocation that cannot be made at all is already a
    negative answer -- there are not enough cores for the heavy tasks --
    so there is nothing left to map.
    """
    from federated import solve
    from generator import generate_taskset
    from mapping import map_taskset
    from protocols import analyse

    taskset = generate_taskset(cfg)

    if cfg.federated:
        allocation, mapper, analysis = solve(taskset, cfg, protocol)
        if mapper is None or not allocation.feasible:
            return Result(schedulable=False, mapped=False, protocol=protocol,
                          allocation=allocation,
                          unplaced_nodes=taskset.total_nodes)
        return test(taskset, mapper, analysis, allocation)

    mapper = map_taskset(taskset, cfg)
    analysis = analyse(taskset, mapper, protocol, cfg.context_switch)
    return test(taskset, mapper, analysis)

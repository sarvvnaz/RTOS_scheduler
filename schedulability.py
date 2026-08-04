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
  3. under POMIP, every task also meets its deadline by the paper's
     response-time bound
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

    @property
    def reason(self) -> str:
        """One line saying what went wrong, for the report."""
        if self.schedulable:
            return "schedulable"
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


def test(taskset: TaskSet, mapper, analysis: ResourceAnalysis) -> Result:
    """Run the partitioned-EDF test over a mapped and analysed task set."""
    result = Result(schedulable=False,
                    mapped=mapper.succeeded,
                    protocol=analysis.protocol,
                    unplaced_nodes=len(mapper.unplaced))

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

        # under POMIP the response-time bound has to hold as well
        if analysis.protocol is Protocol.POMIP:
            blocking = analysis.task_blocking.get(task.id)
            if blocking and not blocking.meets_deadline(task.deadline):
                result.missed.append(task.id)

    met = len(taskset.tasks) - len(result.missed)
    result._qos = met / len(taskset.tasks) if taskset.tasks else 0.0
    result.schedulable = (mapper.succeeded
                          and not result.overloaded
                          and not result.missed)
    return result


def evaluate(cfg, protocol: Protocol) -> Result:
    """Generate, map, analyse and test one task set in one call.

    This is the unit the experiments repeat: everything from a config to a
    yes/no answer.
    """
    from generator import generate_taskset
    from mapping import map_taskset
    from protocols import analyse

    taskset = generate_taskset(cfg)
    mapper = map_taskset(taskset, cfg)
    analysis = analyse(taskset, mapper, protocol, cfg.context_switch)
    return test(taskset, mapper, analysis)

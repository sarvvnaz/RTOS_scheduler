"""Federated scheduling: splitting the tasks and handing out the cores.

A task is **heavy** when its utilization is above 1, meaning one core can
never run it in time however well it is scheduled. A heavy task is given
its own **dedicated cluster** of cores that nothing else may touch. Every
remaining task is **light** -- it fits inside a single core -- and all the
light tasks share whatever cores are left over.

The number of cores a heavy task needs is the classic federated bound

    m_i = ceil( (C_i - L_i) / (D_i - L_i) )

Read it as: the work that is *not* on the critical path has to be spread
over the slack that the critical path leaves behind. If the critical path
alone already exceeds the deadline (L_i >= D_i) no number of cores can
help, and the task is rejected immediately.

Why this matters here: POMIP's response-time bound divides by m_i because
the paper assumes those m_i processors belong to the task. Under plain
partitioning they do not, and the bound is being used outside the setting
it was proved in. Federated allocation restores that assumption -- inside
a cluster the task really is alone, so the bound applies as written.

Within a cluster the nodes are still mapped by OC-HEFT and each core
still runs its own EDF queue; the cluster only limits *which* cores the
mapper is allowed to choose from.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

from model import Core, Task, TaskSet


@dataclass
class Allocation:
    """Which cores went to which task."""

    clusters: Dict[int, List[int]] = field(default_factory=dict)   # heavy only
    shared: List[int] = field(default_factory=list)                # light tasks
    heavy: List[int] = field(default_factory=list)                 # task ids
    light: List[int] = field(default_factory=list)                 # task ids
    feasible: bool = True
    reason: str = ""

    def cores_for(self, task: Task) -> List[int]:
        """The cores this task is allowed to use."""
        if task.id in self.clusters:
            return self.clusters[task.id]
        return self.shared

    def cluster_size(self, task: Task) -> int:
        """m_i: how many cores the task owns, or shares if it is light."""
        return len(self.cores_for(task)) or 1

    @property
    def num_heavy(self) -> int:
        return len(self.heavy)

    @property
    def num_light(self) -> int:
        return len(self.light)

    @property
    def cores_used(self) -> int:
        return sum(len(c) for c in self.clusters.values()) + len(self.shared)


def is_heavy(task: Task, all_heavy: bool = False) -> bool:
    """A task no single core could ever finish in time.

    ``all_heavy`` forces every task into its own exclusive cluster, even
    the tiny ones. That is the "-H" variant the H2LP paper compares
    against: it shows what exclusive clustering costs when a system is
    full of small tasks, each holding a whole cluster it cannot fill.
    """
    return all_heavy or task.utilization > 1.0


def required_cores(task: Task, speed: float = 1.0) -> int:
    """m_i for a heavy task, on cores of the given speed.

    A faster core shrinks both C_i and L_i by the same factor, so it can
    genuinely reduce the number of cores needed.

    Returns 0 when the task is hopeless: its critical path alone does not
    fit in the deadline, and adding cores cannot shorten a chain.
    """
    c = task.wcet / speed
    length = task.critical_path_length / speed

    if length >= task.deadline:
        return 0                                  # no cluster size helps

    return max(1, math.ceil((c - length) / (task.deadline - length)))


def allocate(taskset: TaskSet, sizes: Dict[int, int] = None,
             prefer_edge: bool = True, all_heavy: bool = False) -> Allocation:
    """Split the tasks and hand out the cores.

    Heavy tasks are served first, largest first, because they are the ones
    with a hard requirement -- a light task can always share. Faster cores
    are offered first when ``prefer_edge`` is set, since a heavy task
    needs fewer of them.

    ``sizes`` overrides the cluster size for individual heavy tasks. The
    federated loop uses it to grow a cluster that turned out too small
    once blocking was counted.
    """
    result = Allocation()
    sizes = sizes or {}

    heavy = [t for t in taskset.tasks if is_heavy(t, all_heavy)]
    light = [t for t in taskset.tasks if not is_heavy(t, all_heavy)]
    result.heavy = [t.id for t in heavy]
    result.light = [t.id for t in light]

    # fastest first: a heavy task placed on edge cores needs fewer of them
    pool: List[Core] = sorted(taskset.cores,
                              key=lambda c: -c.speed if prefer_edge else c.speed)
    free = list(pool)

    # the biggest appetite goes first, so it is not starved by small ones
    for task in sorted(heavy, key=lambda t: -t.utilization):
        if not free:
            result.feasible = False
            result.reason = f"no cores left for heavy task {task.id}"
            return result

        speed = free[0].speed
        needed = sizes.get(task.id) or required_cores(task, speed)

        if needed == 0:
            result.feasible = False
            result.reason = (f"task {task.id} has L >= D: its critical path "
                             f"cannot fit in its deadline on any cluster")
            return result

        if needed > len(free):
            result.feasible = False
            result.reason = (f"heavy task {task.id} needs {needed} cores, "
                             f"only {len(free)} left")
            return result

        # take them from one speed class so the cluster is uniform
        same_speed = [c for c in free if c.speed == speed][:needed]
        if len(same_speed) < needed:
            same_speed = free[:needed]

        result.clusters[task.id] = [c.id for c in same_speed]
        taken = set(result.clusters[task.id])
        free = [c for c in free if c.id not in taken]

    result.shared = [c.id for c in free]

    if light and not result.shared:
        result.feasible = False
        result.reason = ("the heavy tasks used every core, leaving none for "
                         f"{len(light)} light task(s)")

    return result


def solve(taskset: TaskSet, cfg, protocol, max_rounds: int = 12,
          all_heavy: bool = False):
    """Algorithm 1 of the paper: grow the clusters until they are enough.

    The first cluster size ignores blocking, because blocking cannot be
    known before the nodes are mapped and the mapping needs a cluster to
    map into. So the whole thing is a loop: allocate, map, analyse, and
    give another core to every heavy task that still misses its deadline.

    It stops when no task needs more, or when the heavy tasks between them
    ask for more cores than the machine has -- which is the paper's
    unschedulable answer.

    Returns ``(allocation, mapper, analysis)``; the mapper and analysis are
    None when no allocation could be made at all.
    """
    from mapping import map_taskset
    from protocols import analyse

    sizes: Dict[int, int] = {}

    for _ in range(max_rounds):
        allocation = allocate(taskset, sizes, all_heavy=all_heavy)
        if not allocation.feasible:
            return allocation, None, None

        mapper = map_taskset(taskset, cfg, allocation)
        analysis = analyse(taskset, mapper, protocol, cfg.context_switch)

        grew = False
        for task in taskset.tasks:
            if task.id not in allocation.clusters:
                continue                      # light tasks have no cluster
            blocking = analysis.task_blocking.get(task.id)
            if blocking and not blocking.meets_deadline(task.deadline):
                sizes[task.id] = len(allocation.clusters[task.id]) + 1
                grew = True

        if not grew:
            return allocation, mapper, analysis

    # ran out of rounds: report the last state rather than pretending
    allocation.feasible = False
    allocation.reason = (f"cluster sizes did not settle in {max_rounds} rounds")
    return allocation, mapper, analysis

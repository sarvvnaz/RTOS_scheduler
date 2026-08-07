"""Generating a task set, step by step.

  1. create the cores
  2. decide how many nodes each task has
  3. split U = m * U_norm over the tasks          (RandFixedSum)
  4. build each task graph (Erdos-Renyi + source and sink)
  5. split the task utilization over its nodes    (UUniFast), get the WCETs
  6. create the resources and spread the accesses over tasks, then nodes
  7. build each node body: normal / critical / normal / ...
"""

from collections import defaultdict
from typing import Dict, List

import networkx as nx
import numpy as np

from algorithms import randfixedsum, uunifast
from config import Config
from model import Core, Node, Resource, Segment, Task, TaskSet

SOURCE_ID = 0


def generate_taskset(cfg: Config) -> TaskSet:
    """Generate one complete task set."""
    cfg.check()
    rng = np.random.default_rng(cfg.seed)

    # 1. cores: the local machine plus the edge servers
    cores = _make_cores(cfg)

    # 2. node counts, decided first so we know the utilizations will fit
    node_counts = [cfg.pick_num_nodes(rng) for _ in range(cfg.num_tasks)]

    # 3. split the total utilization over the tasks
    task_utils = _split_over_tasks(cfg, node_counts, rng)

    # 4 + 5. build the tasks
    tasks = [_make_task(i, u, n, cfg, rng)
             for i, (u, n) in enumerate(zip(task_utils, node_counts), start=1)]

    # 6 + 7. resources, and the node bodies that use them
    resources = _make_resources(tasks, cfg, rng)

    taskset = TaskSet(tasks=tasks, resources=resources, cores=cores)
    taskset.check()
    return taskset


def generate_many(cfg: Config, count: int = 100) -> List[TaskSet]:
    """Generate several task sets (the experiments use 100 per setting)."""
    return [generate_taskset(cfg.copy_with(seed=cfg.seed + i))
            for i in range(count)]


# ---------------------------------------------------------------------------
# Step 1: the platform
# ---------------------------------------------------------------------------

def _make_cores(cfg: Config) -> List[Core]:
    """Local cores first, then the cores of each edge server.

    Every core gets a unique id, so a core is identified by one number no
    matter where it lives. Edge cores are faster, which is what makes
    offloading worthwhile.
    """
    cores = [Core(id=i, speed=cfg.local_speed, server_id=None)
             for i in range(cfg.num_cores)]

    next_id = cfg.num_cores
    for server in range(1, cfg.num_edge_servers + 1):
        for _ in range(cfg.cores_per_edge_server):
            cores.append(Core(id=next_id, speed=cfg.edge_speed,
                              server_id=server))
            next_id += 1

    return cores


# ---------------------------------------------------------------------------
# Step 3: utilization over the tasks
# ---------------------------------------------------------------------------

def _split_over_tasks(cfg: Config, node_counts: List[int],
                      rng: np.random.Generator) -> np.ndarray:
    """U = m * U_norm, split over the n tasks.

    A task utilization may be above 1, but it still has to fit inside the
    task's own nodes, because every node must stay below 1.
    """
    total = cfg.total_utilization
    largest = 0.9 * min(node_counts)

    if total > cfg.num_tasks * largest:
        needed = int(np.ceil(total / largest))
        raise ValueError(
            f"U = {total:.2f} does not fit in {cfg.num_tasks} task(s): a task "
            f"with {min(node_counts)} nodes holds at most {largest:.2f}. "
            f"Use at least {needed} tasks, or lower u_norm.")

    return randfixedsum(cfg.num_tasks, total, rng, high=largest)


# ---------------------------------------------------------------------------
# Steps 4 and 5: one task
# ---------------------------------------------------------------------------

def _make_task(task_id: int, utilization: float, num_nodes: int,
               cfg: Config, rng: np.random.Generator) -> Task:
    """Build one task graph, retrying until it is at least possible.

    A task whose critical path is longer than its deadline can never be
    scheduled: the path is a chain, so no number of cores shortens it.
    Such a task says nothing about the scheduler, and counting it as a
    scheduling failure would blame the analysis for the input.

    The graph shape and the utilization split both move the critical path,
    so drawing again usually fixes it. If it never does, the setting
    itself is the problem and the caller is told so.
    """
    for _ in range(cfg.generation_attempts):
        task = _draw_task(task_id, utilization, num_nodes, cfg, rng)
        if task.critical_path_length <= task.deadline:
            return task

    raise ValueError(
        f"task {task_id} (u={utilization:.2f}, {num_nodes} nodes) always came "
        f"out with a critical path longer than its deadline after "
        f"{cfg.generation_attempts} attempts: this utilization does not fit "
        f"in this many nodes at these periods")


def _draw_task(task_id: int, utilization: float, num_nodes: int,
               cfg: Config, rng: np.random.Generator) -> Task:
    """One attempt at building a task graph with its execution times."""
    graph = _make_graph(num_nodes, cfg.edge_prob, rng)
    sink_id = num_nodes + 1

    period = cfg.pick_period(rng)
    node_utils = uunifast(num_nodes, utilization, rng)   # every value below 1

    nodes = {}
    for node_id in graph.nodes:
        dummy = node_id in (SOURCE_ID, sink_id)
        u = 0.0 if dummy else float(node_utils[node_id - 1])
        nodes[node_id] = Node(id=node_id, utilization=u,
                              wcet=u * period, is_dummy=dummy)

    task = Task(id=task_id, period=period, utilization=float(node_utils.sum()),
                graph=graph, nodes=nodes,
                source_id=SOURCE_ID, sink_id=sink_id)
    _add_comm_costs(task, cfg, rng)
    return task


def _add_comm_costs(task: Task, cfg: Config, rng: np.random.Generator) -> None:
    """Give every edge of the graph a communication cost.

        average = CCR * C_i
        comm(u, v) = U(0.5 * average, 1.5 * average)

    The cost is stored on every edge, but it is only actually paid when the
    two endpoints are mapped to different cores. Two nodes on the same core
    communicate for free.
    """
    average = cfg.ccr * task.wcet
    for u, v in task.graph.edges:
        task.graph.edges[u, v]["comm"] = cfg.pick_comm_cost(average, rng)


def _make_graph(num_nodes: int, edge_prob: float,
                rng: np.random.Generator) -> nx.DiGraph:
    """Erdos-Renyi random graph, plus a single source and a single sink.

    Edges only go from a lower id to a higher id, so the graph can never
    contain a cycle. The generated graph may have several starting and
    ending nodes, so a source is joined to every node without a parent and
    a sink to every node without a child.
    """
    sink_id = num_nodes + 1
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes + 2))

    for u in range(1, num_nodes + 1):
        for v in range(u + 1, num_nodes + 1):
            if rng.uniform() < edge_prob:
                graph.add_edge(u, v)

    for v in range(1, num_nodes + 1):
        if graph.in_degree(v) == 0:
            graph.add_edge(SOURCE_ID, v)
        if graph.out_degree(v) == 0:
            graph.add_edge(v, sink_id)

    return graph


# ---------------------------------------------------------------------------
# Steps 6 and 7: resources and node bodies
# ---------------------------------------------------------------------------

def _make_resources(tasks: List[Task], cfg: Config,
                    rng: np.random.Generator) -> List[Resource]:
    """Create the resources, hand out the accesses, build the node bodies."""
    resources = []
    per_task: Dict[int, List[int]] = defaultdict(list)

    for resource_id in range(1, cfg.num_resources + 1):
        total = cfg.accesses_per_resource

        # give each access to a random task, one at a time
        counts: Dict[int, int] = defaultdict(int)
        for _ in range(total):
            counts[int(rng.choice([t.id for t in tasks]))] += 1

        resources.append(Resource(id=resource_id, total_accesses=total,
                                  accesses_per_task=dict(counts)))
        for task_id, count in counts.items():
            per_task[task_id].extend([resource_id] * count)

    for task in tasks:
        _spread_over_nodes(task, per_task[task.id], cfg, rng)

    return resources


def _spread_over_nodes(task: Task, accesses: List[int], cfg: Config,
                       rng: np.random.Generator) -> None:
    """Give each access to a random node, then build every node body."""
    real_nodes = task.real_nodes()

    per_node: Dict[int, List[int]] = defaultdict(list)
    for resource_id in accesses:
        node = real_nodes[int(rng.integers(len(real_nodes)))]
        per_node[node.id].append(resource_id)

    for node in real_nodes:
        _build_body(node, per_node[node.id], cfg.csp, rng)


def _build_body(node: Node, resource_ids: List[int], csp: float,
                rng: np.random.Generator) -> None:
    """Build the execution sequence of one node.

    With s critical sections the node becomes

        normal, critical, normal, critical, ..., critical, normal

    so every critical section sits between two normal ones. The critical
    sections together take ``csp`` of the node execution time and the rest
    is normal; both totals are split with RandFixedSum. A normal section
    may come out as zero, which means the node starts straight with a
    critical section.
    """
    s = len(resource_ids)
    if s == 0:
        node.segments = [Segment(length=node.wcet)]
        return

    rng.shuffle(resource_ids)                    # random order inside the node
    critical_total = csp * node.wcet
    normal_total = node.wcet - critical_total

    critical = randfixedsum(s, critical_total, rng)
    normal = randfixedsum(s + 1, normal_total, rng)

    segments = []
    for k in range(s):
        segments.append(Segment(length=float(normal[k])))
        segments.append(Segment(length=float(critical[k]), is_critical=True,
                                resource_id=resource_ids[k]))
    segments.append(Segment(length=float(normal[s])))
    node.segments = segments

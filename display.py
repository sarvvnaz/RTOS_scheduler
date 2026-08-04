"""Printing what was generated.

This is the deliverable for this stage: being able to see the graph of
every task, the execution time of every node, and the exact execution
sequence of each node.
"""

from model import Node, Task, TaskSet

LINE = "=" * 74
THIN = "-" * 74


def show_node(node: Node) -> str:
    """One node: its execution time and its sequence of segments."""
    if node.is_dummy:
        kind = "source" if node.id == 0 else "sink"
        return f"  node {node.id:<3} ({kind})"

    parts = []
    for seg in node.segments:
        if seg.is_critical:
            parts.append(f"CS R{seg.resource_id} {seg.length:.2f}")
        else:
            parts.append(f"NS {seg.length:.2f}")

    return (f"  node {node.id:<3} wcet={node.wcet:8.2f}  "
            f"u={node.utilization:.4f}  cs={node.num_critical_sections}\n"
            f"          {' | '.join(parts)}")


def show_task(task: Task) -> str:
    """One task: its numbers, its graph shape, and all of its nodes."""
    used = ", ".join(f"R{r}" for r in task.resources_used) or "none"
    counts = ", ".join(f"R{r}:{task.access_count(r)}" for r in task.resources_used)

    comms = [task.comm_cost(u, v) for u, v in task.graph.edges]
    comm_line = (f"  communication: {len(comms)} edges, "
                 f"avg {sum(comms) / len(comms):.2f}, "
                 f"range {min(comms):.2f} .. {max(comms):.2f}"
                 if comms and max(comms) > 0 else "")

    lines = [
        THIN,
        f"task {task.id}   T={task.period:.0f}   D={task.deadline:.0f}   "
        f"u={task.utilization:.4f}",
        f"  C (total execution time) = {task.wcet:.2f}",
        f"  L (critical path length) = {task.critical_path_length:.2f}",
        f"  graph: {task.num_nodes} nodes, {task.num_edges} edges",
    ]
    if comm_line:
        lines.append(comm_line)
    lines += [
        f"  resources: {used}" + (f"   ({counts})" if counts else ""),
        "  execution sequence of each node:",
    ]
    lines += [show_node(n) for n in task.nodes.values()]
    return "\n".join(lines)


def show_taskset(taskset: TaskSet, title: str = "") -> str:
    """The full report."""
    lines = [LINE]
    if title:
        lines += [title, LINE]

    lines.append(
        f"{len(taskset.tasks)} tasks, {len(taskset.resources)} resources, "
        f"{len(taskset.cores)} cores, {taskset.total_nodes} nodes, "
        f"{taskset.total_critical_sections} critical sections")
    lines.append(f"U = {taskset.total_utilization:.3f}   "
                 f"U_norm = {taskset.u_norm:.3f}")

    lines += [show_task(t) for t in taskset.tasks]

    lines.append(THIN)
    lines.append(f"resources ({len(taskset.resources)})")
    for res in taskset.resources:
        split = ", ".join(f"task{t}:{c}"
                          for t, c in sorted(res.accesses_per_task.items()))
        lines.append(f"  R{res.id}  accesses={res.total_accesses:<4} "
                     f"usage={taskset.resource_usage_ratio(res.id) * 100:5.1f}%  "
                     f"[{split}]")

    lines.append(show_platform(taskset))
    lines.append(LINE)
    return "\n".join(lines)


def show_platform(taskset: TaskSet) -> str:
    """The cores available, split into local and edge."""
    local = taskset.local_cores
    edge = taskset.edge_cores

    lines = [THIN, f"platform ({len(taskset.cores)} cores)"]
    lines.append(f"  local: {len(local)} cores at speed {local[0].speed}"
                 if local else "  local: none")

    if edge:
        for server in taskset.edge_server_ids:
            on_server = [c for c in edge if c.server_id == server]
            lines.append(f"  edge server {server}: {len(on_server)} cores "
                         f"at speed {on_server[0].speed}")
    else:
        lines.append("  edge: none (local-only scenario, no offloading)")

    assigned = sum(len(c.assigned_nodes) for c in taskset.cores)
    if assigned == 0:
        lines.append("  no nodes mapped yet (that is the next stage)")
    else:
        for core in taskset.cores:
            if core.assigned_nodes:
                lines.append(f"  {core.name}: u={core.utilization:.3f}  "
                             f"{len(core.assigned_nodes)} nodes")
    lines.append(f"  hyperperiod = {taskset.hyperperiod:.0f}")
    return "\n".join(lines)


def show_ranks(task: Task, limit: int = 12) -> str:
    """The upward rank of each node, in placement order.

    The top of this list is placed first by OC-HEFT.
    """
    from mapping import placement_order, rank_path, upward_rank

    rank = upward_rank(task)
    order = placement_order(task)
    path = rank_path(task)

    lines = [THIN,
             f"task {task.id}: upward rank (placement order)",
             f"  {'place':<6} {'node':<6} {'rank':>10} {'wcet':>9}   role"]

    for position, node_id in enumerate(order[:limit], start=1):
        node = task.nodes[node_id]
        role = ("source" if node_id == task.source_id else
                "sink" if node_id == task.sink_id else
                f"cs={node.num_critical_sections}")
        on_path = " <- longest chain" if node_id in path else ""
        lines.append(f"  {position:<6} {node_id:<6} {rank[node_id]:>10.2f} "
                     f"{node.wcet:>9.2f}   {role}{on_path}")

    if len(order) > limit:
        lines.append(f"  ... and {len(order) - limit} more")

    lines.append(f"  longest chain: {' -> '.join(str(v) for v in path)}")
    lines.append(f"  source rank = {rank[task.source_id]:.2f}   "
                 f"(L without communication = {task.critical_path_length:.2f})")
    return "\n".join(lines)


def show_mapping(taskset: TaskSet, mapper, limit: int = 10) -> str:
    """Where OC-HEFT put every node, and what it cost.

    ``mapper`` is a finished ``mapping.Mapper``.
    """
    lines = [THIN, "OC-HEFT mapping"]

    w1, w2, w3 = mapper.w1, mapper.w2, mapper.w3
    lines.append(f"  cost = {w1}*Exec + {w2}*Comm + {w3}*RC   "
                 f"subject to finish <= deadline and u <= 1")

    for task in taskset.tasks:
        placed = [p for p in mapper.placements if p.task_id == task.id]
        ok = [p for p in placed if p.placed]
        lines.append(f"  task {task.id}: {len(ok)}/{len(placed)} nodes placed")

        for p in placed[:limit]:
            if not p.placed:
                reasons = set(p.rejected.values()) or {"no core"}
                lines.append(f"    node {p.node_id:<4} UNPLACED     "
                             f"({', '.join(sorted(reasons))})")
                continue
            core = taskset.core(p.core_id)
            lines.append(
                f"    node {p.node_id:<4} -> {core.name:<7} "
                f"exec={p.exec_cost:8.2f} comm={p.comm_cost:9.2f} "
                f"rc={p.contention:7.3f}  finish={p.finish_time:9.2f}")

        if len(placed) > limit:
            lines.append(f"    ... and {len(placed) - limit} more")

    used = [c for c in taskset.cores if c.assigned_nodes]
    lines.append(f"  cores used: {len(used)} of {len(taskset.cores)}")
    for core in used:
        lines.append(f"    {core.name:<7} u={core.utilization:.3f}  "
                     f"{len(core.assigned_nodes)} nodes")

    if mapper.succeeded:
        lines.append("  every node was placed: the set is schedulable here")
    else:
        lines.append(f"  NOT schedulable: {len(mapper.unplaced)} node(s) "
                     f"had no core satisfying both conditions")
    return "\n".join(lines)


def show_summary(taskset: TaskSet) -> str:
    """Short version: one line per task."""
    lines = [f"{len(taskset.tasks)} tasks, {len(taskset.resources)} resources, "
             f"U={taskset.total_utilization:.3f}, U_norm={taskset.u_norm:.3f}"]
    for t in taskset.tasks:
        lines.append(
            f"  task{t.id}: T={t.period:<6.0f} |V|={t.num_nodes:<3} "
            f"|E|={t.num_edges:<4} u={t.utilization:.4f} "
            f"C={t.wcet:9.2f} L={t.critical_path_length:8.2f} "
            f"R={t.resources_used}")
    return "\n".join(lines)

"""Printing what was generated.

This is the deliverable for this stage: being able to see the graph of
every task, the execution time of every node, and the exact execution
sequence of each node.
"""

from .model import Node, Task, TaskSet

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

    lines = [
        THIN,
        f"task {task.id}   T={task.period:.0f}   D={task.deadline:.0f}   "
        f"u={task.utilization:.4f}",
        f"  C (total execution time) = {task.wcet:.2f}",
        f"  L (critical path length) = {task.critical_path_length:.2f}",
        f"  graph: {task.num_nodes} nodes, {task.num_edges} edges",
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
        lines.append(f"  R{res.id}  accesses={res.total_accesses:<4} [{split}]")

    lines.append(THIN)
    lines.append(f"cores ({len(taskset.cores)}) - created, "
                 f"no tasks assigned yet (that is the next stage)")
    lines.append(LINE)
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

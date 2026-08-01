"""Generating periodic DAG task sets with shared resources.

This stage builds the system setup only: the cores, the task graphs, the
execution sequence of every node (normal and critical sections) and the
shared resources - and prints all of it.

Still to come: the resource-management protocol, assigning tasks to cores
plus the scheduler, and offloading.
"""

from .config import Config
from .display import show_node, show_summary, show_task, show_taskset
from .generator import generate_many, generate_taskset
from .model import Core, Node, Resource, Segment, Task, TaskSet

__all__ = [
    "Config",
    "generate_taskset", "generate_many",
    "Segment", "Node", "Task", "Resource", "Core", "TaskSet",
    "show_taskset", "show_task", "show_node", "show_summary",
]

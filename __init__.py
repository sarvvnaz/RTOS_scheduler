"""Generating periodic DAG task sets with shared resources.

The modules are flat and use plain imports, so everything runs from
inside this folder:

    python main.py --tasks 3 --edge-servers 2

and from a script or notebook in this same folder:

    from config import Config
    from generator import generate_taskset
    from display import show_taskset

This stage builds the system setup: the cores (local machine plus edge
servers), the task graphs, the execution sequence of every node (normal
and critical sections), the shared resources, and the communication cost
of every edge - and prints all of it.

Still to come: mapping nodes onto cores with OC-HEFT, the partitioned-EDF
scheduler, and the spin-lock / suspension-based resource protocols.
"""

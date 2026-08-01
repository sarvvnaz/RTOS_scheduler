# Task-set generation with shared resources

Synthetic generator for periodic **DAG task sets** with **shared resources**,
for the real-time scheduling project supervised by Dr. Jafari.

**This stage is the system setup only:** create the cores, the task graphs,
the execution sequence of every node (normal and critical sections) and the
shared resources — and print all of it.

Not implemented yet, on purpose (her order of work):
the resource-management protocol → assigning tasks to cores + the scheduler
→ offloading.

There is **no mixed-criticality** here: every node has one execution time.

## Files

```
msrp_project/
├── config.py       # every setting, with setters and pickers
├── algorithms.py   # UUniFast and RandFixedSum
├── model.py        # Segment, Node, Task, Resource, Core, TaskSet
├── generator.py    # the 7 generation steps
├── display.py      # printing the result
└── main.py         # run it
```

## Run it

```bash
python -m msrp_project.main --tasks 3 --nodes 5 8 --resources 3
```

Small `--nodes` values keep the output short enough to check by hand.
Add `--short` for one line per task.

```python
from msrp_project import Config, generate_taskset, show_taskset

cfg = Config(seed=1, num_cores=8, num_tasks=4, csp=0.5)
print(show_taskset(generate_taskset(cfg)))
```

One node of the output — this is the execution sequence:

```
node 3   wcet=  500.13  u=0.0834  cs=3
         NS 33.18 | CS R3 19.28 | NS 20.04 | CS R1 102.87 | NS 73.33 | CS R1 127.92 | NS 123.51
```

`NS` = normal section, `CS Rq` = critical section holding resource `Rq`.

## Settings

All in `config.py`. Change a **value** with a setter, change a **rule** by
overriding a picker.

| Setting | Symbol | Default |
|---|---|---|
| cores | m | 8 |
| utilization per core | U_norm | 0.5 |
| tasks | n | 4 |
| nodes per task | \|V_i\| | 20–50 |
| edge probability | p | 0.1 |
| period | T_i | one of 2000/4000/6000 |
| resources | n_r | 4 |
| accesses per resource | N_q | 30 |
| critical-section share | CSP | 0.5 |

`U = m × U_norm` is derived, never stored.

```python
cfg = Config().set_num_cores(16).set_u_norm(0.7)       # change values

class MyConfig(Config):                                 # change a rule
    def pick_period(self, rng):
        return 3000.0

for u in [0.1, 0.3, 0.5]:                               # sweep
    sets = generate_many(cfg.copy_with(u_norm=u), 100)
```

## How generation works

1. **Cores** — `m` of them.
2. **Node counts** — drawn first, so each task's utilization is sure to fit
   inside its own nodes.
3. **Task utilizations** — split `U = m × U_norm` over the tasks with
   **RandFixedSum**. A task's utilization may go above 1.
4. **Task graphs** — Erdős–Rényi with probability `p`, then a `source` and a
   `sink` (both zero time) so the graph has exactly one start and one end.
5. **Node execution times** — split each task's utilization over its nodes
   with **UUniFast** (each node below 1), then `c = u × T`.
6. **Resources** — hand out `N_q` accesses per resource, randomly over the
   tasks and then over the nodes.
7. **Node bodies** — critical sections take `CSP` of the node's time (split
   with RandFixedSum), the rest is normal, in the order `NS, CS, NS, …, NS`.
   Accesses are non-nested because segments run one after another.

`TaskSet.check()` runs automatically and verifies all of it: acyclic graph,
one source and one sink, utilizations adding up, node utilizations below 1,
segments summing to the node WCET, no two critical sections adjacent, and
access counts matching the actual node segments.

## Dependencies

`numpy` and `networkx`.

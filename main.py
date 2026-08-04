"""Run the generator and print the result.

    python -m msrp_project.main
    python -m msrp_project.main --tasks 3 --nodes 5 8
    python -m msrp_project.main --cores 16 --u-norm 0.7 --short
"""

import argparse

from config import Config
from display import show_mapping, show_ranks, show_summary, show_taskset
from generator import generate_taskset


def main():
    p = argparse.ArgumentParser(description="Generate and show a task set")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--u-norm", type=float, default=0.5)
    p.add_argument("--tasks", type=int, default=4)
    p.add_argument("--resources", type=int, default=4)
    p.add_argument("--csp", type=float, default=0.5)
    p.add_argument("--edge-servers", type=int, default=2,
                   help="number of edge servers (0 = local only, no offloading)")
    p.add_argument("--edge-cores", type=int, default=8,
                   help="cores per edge server")
    p.add_argument("--edge-speed", type=float, default=2.0,
                   help="how much faster an edge core is than a local one")
    p.add_argument("--ccr", type=float, default=0.5,
                   help="communication to computation ratio")
    p.add_argument("--nodes", type=int, nargs=2, default=(20, 50),
                   metavar=("LOW", "HIGH"))
    p.add_argument("--short", action="store_true")
    p.add_argument("--ranks", action="store_true",
                   help="show the upward rank of each node (OC-HEFT step 1)")
    p.add_argument("--map", action="store_true",
                   help="run OC-HEFT and show where each node was placed")
    args = p.parse_args()

    # every setting is applied here, in one place
    cfg = Config(seed=args.seed, num_cores=args.cores, u_norm=args.u_norm,
                 num_tasks=args.tasks, num_resources=args.resources,
                 csp=args.csp, nodes_per_task=tuple(args.nodes),
                 num_edge_servers=args.edge_servers,
                 cores_per_edge_server=args.edge_cores,
                 edge_speed=args.edge_speed, ccr=args.ccr)

    taskset = generate_taskset(cfg)

    if args.short:
        print(cfg.describe())
        print(show_summary(taskset))
    else:
        print(show_taskset(taskset, title=cfg.describe()))

    if args.ranks:
        for task in taskset.tasks:
            print(show_ranks(task))

    if args.map:
        from mapping import map_taskset
        print(show_mapping(taskset, map_taskset(taskset, cfg)))


if __name__ == "__main__":
    main()

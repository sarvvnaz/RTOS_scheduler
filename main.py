"""Run the generator and print the result.

    python -m msrp_project.main
    python -m msrp_project.main --tasks 3 --nodes 5 8
    python -m msrp_project.main --cores 16 --u-norm 0.7 --short
"""

import argparse

from config import Config
from display import show_mapping, show_ranks, show_summary, show_taskset
from generator import generate_taskset


def run_experiments(cfg, count: int, directory: str) -> None:
    """Run every sweep, draw the charts, and write the numbers beside them.

    This is the final output the project asks for.
    """
    import os
    import time

    from charts import draw_all
    from experiments import SCENARIOS, SWEEPS, run_all, to_csv

    skipped = [s.name for s in SCENARIOS if not s.implemented]
    print(f"{len(SWEEPS)} sweeps x {count} task sets per point")
    if skipped:
        print(f"not run: {', '.join(skipped)} (not implemented yet)")

    def progress(sweep, scenario, point):
        print(f"  {sweep.key:<22} {scenario.name:<26} "
              f"{sweep.label}={point.value:<6} "
              f"sched={point.schedulability:.2f} qos={point.quality_of_service:.2f}")

    start = time.time()
    results = run_all(cfg, count=count, progress=progress)

    written = draw_all(results, directory)
    csv_path = os.path.join(directory, "results.csv")
    with open(csv_path, "w") as handle:
        handle.write(to_csv(results))

    print(f"\ndone in {time.time() - start:.0f}s")
    for path in written + [csv_path]:
        print(f"  {path}")


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
    p.add_argument("--experiments", action="store_true",
                   help="run every sweep and draw the charts (the final output)")
    p.add_argument("--count", type=int, default=100,
                   help="task sets per point (the definition asks for 100)")
    p.add_argument("--charts-dir", default="charts",
                   help="where to write the charts and the csv")
    args = p.parse_args()

    # every setting is applied here, in one place
    cfg = Config(seed=args.seed, num_cores=args.cores, u_norm=args.u_norm,
                 num_tasks=args.tasks, num_resources=args.resources,
                 csp=args.csp, nodes_per_task=tuple(args.nodes),
                 num_edge_servers=args.edge_servers,
                 cores_per_edge_server=args.edge_cores,
                 edge_speed=args.edge_speed, ccr=args.ccr)

    if args.experiments:
        run_experiments(cfg, args.count, args.charts_dir)
        return

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

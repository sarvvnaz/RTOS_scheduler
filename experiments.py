"""The experiments: every sweep from the project definition.

For each setting we generate many task sets, map them, analyse the
blocking and run the EDF test, then report two averages:

* **schedulability** -- the share of task sets where *every* task fits
* **quality of service** -- the share of individual tasks that meet their
  deadline, which still moves when a set fails only partly

Each sweep varies one parameter and is run under every scenario, so the
scenarios can be compared on the same axes.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from config import Config
from protocols import Protocol
from schedulability import evaluate


@dataclass
class Scenario:
    """One line on a chart: a protocol plus any platform changes."""

    name: str
    protocol: Protocol
    overrides: dict = field(default_factory=dict)
    implemented: bool = True

    def config(self, base: Config, **changes) -> Config:
        return base.copy_with(**{**self.overrides, **changes})


# Both protocols are run on both platforms, so the two effects can be
# told apart. With only one protocol on the local platform there is no
# way to know whether a local-vs-edge gap comes from the platform or from
# the protocol that happened to be paired with it.
#
# The adaptive protocol is not decided yet, so its slots are reserved but
# not run.
LOCAL_ONLY = {"num_edge_servers": 0}

# Parameters that only mean something when there is an edge platform.
EDGE_PARAMETERS = {"num_edge_servers", "cores_per_edge_server"}

SCENARIOS = [
    Scenario("local, spin (MSRP)", Protocol.MSRP, LOCAL_ONLY),
    Scenario("local, suspension (POMIP)", Protocol.POMIP, LOCAL_ONLY),
    Scenario("edge, spin (MSRP)", Protocol.MSRP),
    Scenario("edge, suspension (POMIP)", Protocol.POMIP),
    Scenario("local, adaptive", Protocol.ADAPTIVE, LOCAL_ONLY, implemented=False),
    Scenario("edge, adaptive", Protocol.ADAPTIVE, implemented=False),
]


@dataclass
class Sweep:
    """One chart: a parameter, the values it takes, and how to apply it.

    ``base`` adjusts the *other* settings for this chart only. One set of
    settings cannot put all eight parameters in the range where they
    matter: light enough for the load sweep to show a curve is so light
    that resource count changes nothing, and heavy enough for resource
    count to bite leaves every load point already failing. So each chart
    states the conditions it is measured under, and those conditions are
    written into the csv beside the numbers.
    """

    key: str                       # config field to vary
    label: str                     # axis label
    values: list
    title: str
    base: dict = field(default_factory=dict)

    def apply(self, base: Config, scenario: Scenario, value) -> Config:
        return scenario.config(base, **{**self.base, self.key: value})

    @property
    def conditions(self) -> str:
        """The adjusted settings, for the chart subtitle and the csv."""
        if not self.base:
            return ""
        return ", ".join(f"{k}={v}" for k, v in sorted(self.base.items()))


# Every chart the project definition asks for.
SWEEPS = [
    Sweep("u_norm", "U_norm", [0.1, 0.3, 0.5, 0.7, 1.0],
          "Schedulability vs normalized utilization"),

    Sweep("accesses_per_resource", "requests per resource",
          [10, 30, 50, 80, 150],
          "Schedulability vs requests per resource"),

    # resource count only bites once there is enough locking to contend on
    Sweep("num_resources", "resource types", [2, 4, 6, 8],
          "Schedulability vs number of resource types",
          {"accesses_per_resource": 50, "csp": 0.25}),

    Sweep("csp", "CSP", [0.1, 0.25, 0.5, 0.75, 1.0],
          "Schedulability vs critical-section share"),

    # same reason: with light locking every task count passes
    Sweep("num_tasks", "tasks", [4, 6, 8],
          "Schedulability vs number of tasks",
          {"accesses_per_resource": 50, "csp": 0.25}),

    Sweep("num_cores", "local cores m", [4, 8, 16, 32],
          "Schedulability vs local core count"),

    # offloading only pays when communication is cheap enough to cross a
    # machine boundary, and only matters when the local machine is loaded
    Sweep("num_edge_servers", "edge servers", [0, 1, 2, 3, 4],
          "Schedulability vs number of edge servers",
          {"ccr": 0.05, "u_norm": 0.2,
           "accesses_per_resource": 30, "csp": 0.15}),

    Sweep("cores_per_edge_server", "cores per edge server", [8, 16, 32, 64],
          "Schedulability vs cores per edge server",
          {"ccr": 0.05, "u_norm": 0.3, "num_edge_servers": 1,
           "accesses_per_resource": 30, "csp": 0.15}),
]


@dataclass
class Point:
    """One point on one line."""

    value: float
    schedulability: float
    quality_of_service: float
    samples: int


def run_point(base: Config, sweep: Sweep, scenario: Scenario, value,
              count: int) -> Point:
    """Average over ``count`` task sets at one setting.

    A task set that cannot even be generated -- the utilization does not
    fit in the nodes, say -- is skipped rather than counted as a failure,
    because that is a limit of the generator, not of the scheduling.
    """
    scheduled = 0
    qos = 0.0
    samples = 0

    for i in range(count):
        cfg = sweep.apply(base, scenario, value).copy_with(seed=base.seed + i)
        try:
            result = evaluate(cfg, scenario.protocol)
        except ValueError:
            continue                      # this setting cannot be generated
        scheduled += result.schedulable
        qos += result.quality_of_service
        samples += 1

    if not samples:
        return Point(value, 0.0, 0.0, 0)
    return Point(value, scheduled / samples, qos / samples, samples)


def run_sweep(base: Config, sweep: Sweep, count: int,
              scenarios: Optional[List[Scenario]] = None,
              progress: Optional[Callable] = None) -> Dict[str, List[Point]]:
    """Run one sweep under every implemented scenario."""
    scenarios = scenarios or SCENARIOS
    results: Dict[str, List[Point]] = {}

    for scenario in scenarios:
        if not scenario.implemented:
            continue

        # A scenario that pins a parameter cannot appear on the chart that
        # varies it: the sweep would overwrite the pin and the line would
        # be labelled "local" while running on edge cores.
        if sweep.key in scenario.overrides:
            continue

        # Nor can a scenario with no edge servers appear on a chart about
        # the edge platform -- the parameter does nothing for it, and the
        # line would be flat for a reason that has nothing to say.
        if (sweep.key in EDGE_PARAMETERS
                and scenario.overrides.get("num_edge_servers") == 0):
            continue
        points = []
        for value in sweep.values:
            points.append(run_point(base, sweep, scenario, value, count))
            if progress:
                progress(sweep, scenario, points[-1])
        results[scenario.name] = points

    return results


def run_all(base: Config, count: int = 100,
            sweeps: Optional[List[Sweep]] = None,
            progress: Optional[Callable] = None) -> Dict[str, Dict]:
    """Run every sweep. This is the whole experiment."""
    sweeps = sweeps or SWEEPS
    return {sweep.title: {"sweep": sweep,
                          "lines": run_sweep(base, sweep, count, progress=progress)}
            for sweep in sweeps}


def to_csv(results: Dict[str, Dict]) -> str:
    """Every number behind the charts, so the report can quote them."""
    rows = ["chart,parameter,value,scenario,schedulability,qos,samples,conditions"]
    for title, block in results.items():
        sweep = block["sweep"]
        for name, points in block["lines"].items():
            for p in points:
                rows.append(f'"{title}",{sweep.key},{p.value},"{name}",'
                            f"{p.schedulability:.4f},{p.quality_of_service:.4f},"
                            f'{p.samples},"{sweep.conditions}"')
    return "\n".join(rows)

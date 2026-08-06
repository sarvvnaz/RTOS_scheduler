"""All settings of the system, in one place.

Nothing about the setup is hard-coded anywhere else. To change an
experiment you only touch this file (or call a setter).

  * a value  -> change it directly, or use ``set_...``
  * a rule   -> override a ``pick_...`` method
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Config:
    """One experiment setting."""

    seed: int = 1                     # same seed -> same task set

    # platform: the local machine
    num_cores: int = 8                # m, the spec uses 2, 4, 8, 16 or 32
    local_speed: float = 1.0          # all local cores run at the same speed

    # platform: the edge servers (0 servers = no offloading at all)
    num_edge_servers: int = 2         # es, the spec uses 1, 2, 3 or 4
    cores_per_edge_server: int = 8    # the spec uses 8, 16, 32 or 64
    edge_speed: float = 2.0           # edge cores are faster than local ones

    # communication between nodes of the same task (CCR)
    ccr: float = 0.5                  # spec: 0.25 .. 1.75
    comm_factors: tuple = (0.5, 1.5)  # comm = U(0.5 * average, 1.5 * average)

    # load
    u_norm: float = 0.5               # utilization per core, 0.1 .. 1

    # tasks
    num_tasks: int = 100                # n
    nodes_per_task: tuple = (20, 50)  # |V_i|
    edge_prob: float = 0.1            # p of the Erdos-Renyi graph
    periods: tuple = (2000, 4000, 6000)   # T_i is picked from this list

    # OC-HEFT cost weights: Cost = w1 * Exec + w2 * Comm + w3 * RC
    #
    # The spec gives the formula but fixes no values, so they are settings.
    # The three terms are on very different scales -- Comm is driven by
    # CCR * C_i and is usually far larger than Exec -- so these weights are
    # what balances them.
    cost_weights: tuple = (1.0, 1.0, 1.0)   # w1, w2, w3

    # federated scheduling: heavy tasks (u > 1) get their own dedicated
    # cluster of cores, light tasks share what is left. Turning this off
    # lets every task use every core, which is plain partitioning.
    federated: bool = True

    # protocol overhead: one context switch, paid twice per suspension
    context_switch: float = 1.0       # only LPP-P pays this

    # shared resources
    num_resources: int = 4            # n_r, the spec allows 2 .. 8
    accesses_per_resource: int = 30   # N_q, the spec uses 10/30/50/80/150
    csp: float = 0.5                  # share of a node spent in critical sections

    # ---------------------------------------------------------------
    @property
    def total_utilization(self) -> float:
        """U = m * U_norm.

        Only the local cores count here: U_norm is defined as the load of
        the local machine. The edge servers are extra capacity that the
        mapping step may use, not part of this formula.
        """
        return self.num_cores * self.u_norm

    @property
    def num_edge_cores(self) -> int:
        return self.num_edge_servers * self.cores_per_edge_server

    @property
    def total_cores(self) -> int:
        """Local cores plus every edge core."""
        return self.num_cores + self.num_edge_cores

    @property
    def offloading_enabled(self) -> bool:
        """False when there are no edge servers (the local-only scenario)."""
        return self.num_edge_servers > 0

    # -- setters: change a value ------------------------------------
    def set_num_cores(self, m):
        self.num_cores = m
        return self

    def set_num_edge_servers(self, es):
        self.num_edge_servers = es
        return self

    def set_cores_per_edge_server(self, m):
        self.cores_per_edge_server = m
        return self

    def set_edge_speed(self, speed):
        self.edge_speed = speed
        return self

    def set_ccr(self, ccr):
        self.ccr = ccr
        return self

    def set_u_norm(self, u):
        self.u_norm = u
        return self

    def set_num_tasks(self, n):
        self.num_tasks = n
        return self

    def set_nodes_per_task(self, low, high):
        self.nodes_per_task = (low, high)
        return self

    def set_periods(self, periods):
        self.periods = tuple(periods)
        return self

    def set_num_resources(self, n_r):
        self.num_resources = n_r
        return self

    def set_accesses_per_resource(self, n_q):
        self.accesses_per_resource = n_q
        return self

    def set_csp(self, csp):
        self.csp = csp
        return self

    def copy_with(self, **changes):
        """A copy with some values changed (used for parameter sweeps)."""
        return replace(self, **changes)

    # -- pickers: change a rule -------------------------------------
    def pick_period(self, rng: np.random.Generator) -> float:
        """T_i. Override this to use a formula instead of a list."""
        return float(rng.choice(self.periods))

    def pick_num_nodes(self, rng: np.random.Generator) -> int:
        """|V_i|, not counting the source and the sink."""
        low, high = self.nodes_per_task
        return int(rng.integers(low, high + 1))

    def pick_comm_cost(self, average: float, rng: np.random.Generator) -> float:
        """Cost of one edge: uniform between 0.5 and 1.5 of the average."""
        low, high = self.comm_factors
        return float(rng.uniform(low * average, high * average))

    # ---------------------------------------------------------------
    def check(self) -> None:
        if self.num_cores < 1:
            raise ValueError("num_cores must be >= 1")
        if not 0.1 <= self.u_norm <= 1.0:
            raise ValueError("u_norm must be between 0.1 and 1")
        if self.num_tasks < 1:
            raise ValueError("num_tasks must be >= 1")
        if not 0.1 <= self.csp <= 1.0:
            raise ValueError("csp must be between 0.1 and 1")
        if self.num_edge_servers < 0:
            raise ValueError("num_edge_servers must be >= 0")
        if self.local_speed <= 0 or self.edge_speed <= 0:
            raise ValueError("core speeds must be > 0")
        if self.ccr <= 0:
            raise ValueError("ccr must be > 0")
        if len(self.cost_weights) != 3:
            raise ValueError("cost_weights must be (w1, w2, w3)")
        if any(w < 0 for w in self.cost_weights):
            raise ValueError("cost weights must be >= 0")

    def describe(self) -> str:
        if self.offloading_enabled:
            platform = (f"m={self.num_cores} local + {self.num_edge_servers} edge"
                        f"x{self.cores_per_edge_server} "
                        f"(speed {self.local_speed}/{self.edge_speed})")
        else:
            platform = f"m={self.num_cores} local only (no offloading)"
        return (f"{platform}, U_norm={self.u_norm}, "
                f"U={self.total_utilization:.2f}, n={self.num_tasks} tasks, "
                f"n_r={self.num_resources}, CSP={self.csp}, CCR={self.ccr}, "
                f"seed={self.seed}")

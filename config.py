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

    # platform
    num_cores: int = 8                # m, the spec uses 4, 8, 16 or 32

    # load
    u_norm: float = 0.5               # utilization per core, 0.1 .. 1

    # tasks
    num_tasks: int = 4                # n
    nodes_per_task: tuple = (20, 50)  # |V_i|
    edge_prob: float = 0.1            # p of the Erdos-Renyi graph
    periods: tuple = (2000, 4000, 6000)   # T_i is picked from this list

    # shared resources
    num_resources: int = 4            # n_r, the spec allows 2 .. 8
    accesses_per_resource: int = 30   # N_q, the spec uses 10/30/50/80/150
    csp: float = 0.5                  # share of a node spent in critical sections

    # ---------------------------------------------------------------
    @property
    def total_utilization(self) -> float:
        """U = m * U_norm."""
        return self.num_cores * self.u_norm

    # -- setters: change a value ------------------------------------
    def set_num_cores(self, m):
        self.num_cores = m
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

    def describe(self) -> str:
        return (f"m={self.num_cores} cores, U_norm={self.u_norm}, "
                f"U={self.total_utilization:.2f}, n={self.num_tasks} tasks, "
                f"n_r={self.num_resources}, CSP={self.csp}, seed={self.seed}")

"""Drawing the experiment results.

Every sweep becomes one figure with two panels side by side: average
schedulability on the left, quality of service on the right, one line per
scenario so the scenarios can be read against each other.
"""

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")           # write files, never open a window
import matplotlib.pyplot as plt

STYLES = {
    "local only (no offload)": dict(color="#888888", marker="s", linestyle="--"),
    "spin (MSRP)":             dict(color="#1f77b4", marker="o"),
    "suspension (POMIP)":      dict(color="#d62728", marker="^"),
    "adaptive":                dict(color="#2ca02c", marker="d"),
}


def _panel(axis, sweep, lines: Dict, field: str, ylabel: str) -> None:
    """One panel: the same sweep drawn for every scenario."""
    for name, points in lines.items():
        style = STYLES.get(name, {})
        axis.plot([p.value for p in points],
                  [getattr(p, field) for p in points],
                  label=name, linewidth=1.8, markersize=5, **style)

    axis.set_xlabel(sweep.label)
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.3)


def draw_sweep(title: str, block: Dict, path: str) -> str:
    """Draw one sweep to a file and return the path."""
    sweep, lines = block["sweep"], block["lines"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    _panel(left, sweep, lines, "schedulability", "average schedulability")
    _panel(right, sweep, lines, "quality_of_service", "quality of service")

    left.legend(fontsize=8, loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def draw_all(results: Dict[str, Dict], directory: str = "charts") -> list:
    """Draw every sweep. Returns the files written."""
    os.makedirs(directory, exist_ok=True)
    written = []

    for index, (title, block) in enumerate(results.items(), start=1):
        name = f"{index:02d}_{block['sweep'].key}.png"
        written.append(draw_sweep(title, block, os.path.join(directory, name)))

    return written

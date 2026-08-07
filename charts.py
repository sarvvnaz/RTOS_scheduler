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

# Colour says which protocol, line style says which platform, so the two
# comparisons can be read off the same chart independently.
STYLES = {
    "local, spin (MSRP)":        dict(color="#1f77b4", marker="o", linestyle="--"),
    "local, suspension (POMIP)": dict(color="#d62728", marker="^", linestyle="--"),
    "local, hybrid (H2LP)":      dict(color="#2ca02c", marker="s", linestyle="--"),
    "edge, spin (MSRP)":         dict(color="#1f77b4", marker="o"),
    "edge, suspension (POMIP)":  dict(color="#d62728", marker="^"),
    "edge, hybrid (H2LP)":       dict(color="#2ca02c", marker="s"),
    "local, adaptive":           dict(color="#9467bd", marker="d", linestyle="--"),
    "edge, adaptive":            dict(color="#9467bd", marker="d"),

    # the clustering figures: solid keeps light tasks together, dashed
    # gives every task its own cluster
    "H2LP, heavy + light":       dict(color="#2ca02c", marker="s"),
    "H2LP-H, every task heavy":  dict(color="#2ca02c", marker="s", linestyle="--"),
    "MSRP, heavy + light":       dict(color="#1f77b4", marker="o"),
    "MSRP-H, every task heavy":  dict(color="#1f77b4", marker="o", linestyle="--"),
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
    # the chart must say what it was measured under, or the reader cannot
    # tell why two charts disagree about the same scenario
    figure.suptitle(title + (f"\n({sweep.caption})" if sweep.caption else ""),
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def draw_all(results: Dict[str, Dict], directory: str = "charts") -> list:
    """Draw every sweep. Returns the files written."""
    os.makedirs(directory, exist_ok=True)
    written = []

    for index, (title, block) in enumerate(results.items(), start=1):
        # two charts can sweep the same parameter and differ only in what
        # they hold fixed, so the title has to reach the filename
        slug = "".join(c if c.isalnum() else "_" for c in title.lower())
        slug = "_".join(part for part in slug.split("_") if part)[:48]
        name = f"{index:02d}_{slug}.png"
        written.append(draw_sweep(title, block, os.path.join(directory, name)))

    return written

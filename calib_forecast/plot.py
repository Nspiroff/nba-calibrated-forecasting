"""Calibration figure: PIT histograms (assets/calibration.png).

A calibrated model's PIT values are uniform -> a FLAT histogram. Mis-calibration
shows as shape: a U / peak means intervals are too narrow or too wide. This plots
the three models side by side for points so the calibration story is visible at a
glance.

    python -m calib_forecast.plot
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .forecast import load_features, fit_score  # noqa: E402


def render(stat="pts"):
    df = load_features()
    res = fit_score(df, stat)
    pits = res["_pit"]
    models = list(pits)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), sharey=True)
    for ax, m in zip(axes, models):
        p = pits[m][np.isfinite(pits[m])]
        ax.hist(p, bins=20, range=(0, 1), color="#3b78c3",
                edgecolor="white", weights=np.ones(len(p)) / len(p) * 20)
        ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="calibrated (flat)")
        cov = res["models"][m]["cov80"]
        ax.set_title(f"{m}\ncov80={cov:.2f}  (target 0.80)", fontsize=10)
        ax.set_xlabel("PIT"); ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper center")
    axes[0].set_ylabel("density (1.0 = uniform)")
    fig.suptitle(f"Calibration of {C.STATS[stat]} forecasts — flat is good "
                 f"(Poisson is not; the learned models are)", fontsize=12)
    fig.tight_layout()
    out = C.ASSETS / "calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"WROTE {out}")
    return str(out)


if __name__ == "__main__":
    render()

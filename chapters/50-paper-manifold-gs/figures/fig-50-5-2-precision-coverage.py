"""Figure 50.5.2: [ManifoldGS26] Table 4 as points on the precision-coverage plane.

x = Cov. % (share of the DTU ground truth within the 1% bbox tolerance of the collision candidate),
y = 100 - Float. % (share of the candidate within the tolerance of the ground truth). Colour = method, marker = scene
(circle 24, square 65, triangle 105). Dotted grey curves: equal F1 = 2 p r / (p + r) (0.2, 0.4, 0.6, 0.8), our summary.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-5-2-precision-coverage.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import mgs50 as g  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
mcol = {"ours": C["tangent"], "2DGS": C["third"], "SuGaR": C["accent"], "Poisson": C["aux"]}
mark = {24: "o", 65: "s", 105: "^"}
# self-checks: ours has the lowest floater and the lowest coverage on every scene; by F1, 2DGS beats ours on every scene
for scan, d in g.TABLE4.items():
    assert min(d, key=lambda m: d[m][0]) == "ours" and min(d, key=lambda m: d[m][1]) == "ours"
    assert g.f1(*d["2DGS"]) > g.f1(*d["ours"])
fig, ax = plt.subplots(figsize=(4.8, 4.3))
r = np.linspace(0.01, 1, 400)
for F in (0.2, 0.4, 0.6, 0.8):
    p = F * r / (2 * r - F)
    ok = (p > 0) & (p <= 1.0) & (2 * r - F > 0)
    ax.plot(100 * r[ok], 100 * p[ok], ":", color=C["aux"], lw=0.8)
    j = np.argmin(np.abs(r - 0.97))
    if ok[j]:
        ax.text(98, 100 * p[j] + 1.5, f"F1 {F}", fontsize=8, color=C["aux"], ha="right")
for scan, d in g.TABLE4.items():
    for m, (fl, cov) in d.items():
        ax.plot(cov, 100 - fl, mark[scan], color=mcol[m], ms=7, mec="white", mew=0.5)
for m, c in mcol.items():
    ax.plot([], [], "o", color=c, label=m)
for scan, mk in mark.items():
    ax.plot([], [], mk, color=C["main"], mfc="none", label=f"scan {scan}")
ax.set_xlim(0, 100)
ax.set_ylim(0, 103)
ax.set_xlabel("coverage Cov. (%)")
ax.set_ylabel("precision 100 - Float. (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), fontsize=8, frameon=False, ncol=4, columnspacing=0.8,
          handletextpad=0.3)
fig.tight_layout()
dgfig.save(fig, __file__)
for scan, d in g.TABLE4.items():
    print(scan, {m: round(g.f1(*v), 3) for m, v in d.items()})

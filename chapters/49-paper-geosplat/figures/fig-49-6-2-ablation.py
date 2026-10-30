"""Figure 49.6.2: the ablation of [GeoSplat25, Table 3] as PSNR drops (Section 49.6).

No new experiment: the numbers are the printed table values (lib/geo49.py, TABLE3; GeoGaussian from TABLE1/TABLE2).
For each removed component, the bar is (full model) - (model without it) in PSNR (dB), on Replica R0 (a) and on
ICL Office-3 (b), for both GeoSplat variants. The vertical lines mark the gain of the full model over GeoGaussian on
the same scene (Tables 1-2).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-6-2-ablation.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
labels = ["warm-up (5)", "upsampling (6)", "MAC → mean curv.", "truncation (7)", "shape reg. (8)-(9)",
          "densification (10)-(11)"]
drop, ref = {}, {}
for v in ("Manifold", "Varifold"):
    T = np.array(geo49.TABLE3[v])
    drop[v] = {"R0": T[0, 0] - T[1:, 0], "Office-3": T[0, 3] - T[1:, 3]}
    ref[v] = {"R0": T[0, 0] - geo49.TABLE1["GeoGaussian"]["PSNR"][0], "Office-3": T[0, 3] - geo49.TABLE2["GeoGaussian"]["PSNR"][3]}
    print(v, {k: np.round(d, 2) for k, d in drop[v].items()}, {k: round(r, 2) for k, r in ref[v].items()})
    for sc in ("R0", "Office-3"):
        print("  sum of drops without the MAC row", sc, round(drop[v][sc][[0, 1, 3, 4, 5]].sum(), 2))
assert all(np.all(drop[v][s] > 0) for v in drop for s in drop[v])
assert np.argmax(drop["Manifold"]["R0"]) == 4 and np.argmax(drop["Varifold"]["Office-3"]) == 4
assert all(drop[v][s].max() < ref[v][s] for v in drop for s in drop[v])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
y = np.arange(len(labels))[::-1]
h = 0.38
styles = {"Manifold": (C["main"], "manifold-based", "--"), "Varifold": (C["tangent"], "varifold-based", "-.")}
for k, sc in enumerate(["R0", "Office-3"]):
    ax = axes[k]
    for j, (v, (col, lab, ls)) in enumerate(styles.items()):
        ax.barh(y + (0.5 - j) * h, drop[v][sc], height=h, color=col, alpha=0.85 if j == 0 else 0.75, label=lab)
        ax.axvline(ref[v][sc], color=col, lw=1.0, ls=ls)
    ax.set_title(f"({'ab'[k]}) {'Replica R0' if sc == 'R0' else 'ICL Office-3'}", fontsize=10)
    ax.set_xlabel("PSNR drop when removed (dB)")
    top = max(ref["Manifold"][sc], ref["Varifold"][sc])
    ax.set_xlim(0, 1.3 * top)
    ax.text(top * 1.02, y[-1] - 0.45, "gain over\nGeoGaussian", fontsize=7.5, color=C["aux"], va="bottom")
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels, fontsize=8.5)
axes[1].legend(loc="upper right", fontsize=8, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)

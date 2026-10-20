"""Figure 41.4.2: on the subdivided torus the pointwise error stays at fixed families of vertices; integrated errors converge.

Subdivided torus meshes of Figure 41.4.1(b) (levels 1-4), K_i = delta_i / A_i (mixed area).
(a) max |K_i - K| over three classes of vertices: the 128 vertices of the original torus_mesh(2, 0.8, 8, 16) (class 1),
    vertices on its edges (class 2), vertices inside its faces (class 3); for comparison the regular grids of Figure 41.4.1(a).
(b) three error measures on the subdivided meshes: maximum, area-weighted L2, and the weak error
    |sum_i delta_i rho_i^2 - int K rho^2 dA| with rho^2 = x^2 + y^2 (int K rho^2 dA = 4 pi^2 R r); guide lines of slope 1, 2.
Also writes an interactive version: the level-3 mesh colored by |K_i - K|.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-4-2-where-the-error-lives.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8
weak_exact = 4 * np.pi ** 2 * R * r
rows, keep = [], None
for k in (1, 2, 3, 4):
    V, F, cls = ddg41.subdivided_torus(k)
    u, K, H, k1, k2 = ddg41.torus_exact(V)
    A = dm.mass_voronoi(V, F)
    d = dm.angle_defect(V, F)
    e = np.abs(d / A - K)
    rows.append([ddg41.mean_edge_length(V, F)] + [e[cls == c].max() if np.any(cls == c) else np.nan for c in (1, 2, 3)]
                + [np.sqrt(np.sum(A * e ** 2) / A.sum()), abs(np.sum(d * np.sum(V[:, :2] ** 2, 1)) - weak_exact)])
    if k == 3:
        keep = (V, F, e)
rows = np.array(rows)
reg = []
for n in (12, 24, 48, 96):
    V, F = dm.torus_mesh(R, r, n, 2 * n)
    u, K, *_ = ddg41.torus_exact(V)
    reg.append([ddg41.mean_edge_length(V, F), np.abs(dm.angle_defect(V, F) / dm.mass_voronoi(V, F) - K).max()])
reg = np.array(reg)
assert abs(rows[-1, 1] - 0.0267) < 5e-4 and rows[-1, 1] > rows[-1, 2] > rows[-1, 3]
assert 1.9 < np.log2(rows[-2, 5] / rows[-1, 5]) < 2.1 and 0.9 < np.log2(rows[-2, 4] / rows[-1, 4]) < 1.1

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
ax = axes[0]
for c, (lab, col, st) in enumerate([("original vertices", C["normal"], "s-"), ("on original edges", C["accent"], "^-"),
                                    ("inside original faces", C["tangent"], "o-")]):
    ax.loglog(rows[:, 0], rows[:, c + 1], st, color=col, lw=1.3, ms=4, label=lab)
ax.loglog(reg[:, 0], reg[:, 1], "x--", color=C["aux"], lw=1.0, ms=4, label="regular grid")
ax.set_xlabel(r"mean edge length $h$")
ax.set_ylabel(r"$\max|K_i - K|$")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_title("(a)", loc="left", fontsize=12)
ax = axes[1]
ax.loglog(rows[:, 0], np.nanmax(rows[:, 1:4], 1), "s-", color=C["normal"], lw=1.3, ms=4, label="max")
ax.loglog(rows[:, 0], rows[:, 4], "o-", color=C["tangent"], lw=1.3, ms=4, label=r"$L^2$")
ax.loglog(rows[:, 0], rows[:, 5] / weak_exact, "^-", color=C["third"], lw=1.3, ms=4, label="weak (relative)")
hh = np.array([0.05, 0.14])
ax.loglog(hh, 0.009 * hh, color=C["aux"], lw=0.8)
ax.loglog(hh, 0.016 * hh ** 2, color=C["aux"], lw=0.8)
ax.text(0.15, 1.15e-3, "slope 1", color=C["aux"], fontsize=9.5, va="center")
ax.text(0.15, 3.1e-4, "slope 2", color=C["aux"], fontsize=9.5, va="center")
ax.set_xlabel(r"mean edge length $h$")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_title("(b)", loc="left", fontsize=12)
for a_ in axes:
    a_.set_xticks([0.05, 0.1, 0.2, 0.4])
    a_.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    a_.minorticks_off()
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.9, wspace=0.3)
dgfig.save(fig, __file__)

V, F, e = keep
tr = [dm.plotly_mesh(V, F, intensity=e, colorscale="Viridis", cmin=0, cmax=0.03, showscale=True,
                     name="|K_i - K|", colorbar_title="|K_i - K|")]
dm.plotly_save(tr, __file__, "Subdivided torus (level 3): pointwise error of the angle defect curvature", "fig-41-4-2")
print(np.round(rows, 5).tolist())

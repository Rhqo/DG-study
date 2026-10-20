"""Figure 40.1.3: the pointwise discrete Gaussian curvature K_i = delta_i / A_i depends on the dual area A_i.

(a) icosphere(level), level 1-5 (unit sphere, K = 1): the largest error max |K_i - 1| over the 12 valence-5 vertices
(circles) and over the valence-6 vertices (squares), with the barycentric dual area (dashed) and the mixed Voronoi area
(solid). With the barycentric area the error at the valence-5 vertices stays near 0.146.
(b) the saddle z = x^2 - y^2 on grid_mesh over [-0.5, 0.5]^2 with alternating diagonals (n x n vertices,
n = 9, 17, 33, 65, 129): K_i at the origin (a valence-8 vertex). Mixed: -3.82 -> -4 (the exact K = -4); barycentric:
-> -3, because there the barycentric area is 4/3 of the Voronoi area.

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-1-3-dual-area-choice.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402

C = dgfig.COLORS
levels = np.arange(1, 6)
err = {(k, v): [] for k in ("bary", "mixed") for v in (5, 6)}
for lev in levels:
    V, F = dm.icosphere(lev)
    d = dm.angle_defect(V, F)
    assert abs(d.sum() - 4 * np.pi) < 1e-10
    deg = np.array([len(n) for n in dm.neighbors(F, len(V))])
    for k, A in (("bary", dm.mass_barycentric(V, F)), ("mixed", dm.mass_voronoi(V, F))):
        for v in (5, 6):
            err[(k, v)].append(np.abs(d[deg == v] / A[deg == v] - 1).max())
assert 0.145 < err[("bary", 5)][-1] < 0.147 and err[("mixed", 5)][-1] < 3e-4 and err[("mixed", 6)][-1] < 4e-4
ns = [9, 17, 33, 65, 129]
Kb, Km = [], []
for n in ns:
    V, F = ddg40.saddle_grid(n, 0.5)
    o = int(np.argmin(np.linalg.norm(V, axis=1)))
    assert np.allclose(V[o], 0) and len(dm.neighbors(F, len(V))[o]) == 8
    d = dm.angle_defect(V, F)[o]
    Kb.append(d / dm.mass_barycentric(V, F)[o])
    Km.append(d / dm.mass_voronoi(V, F)[o])
assert abs(Km[-1] + 4) < 2e-3 and abs(Kb[-1] + 3) < 2e-3

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1))
for (k, v), vals in err.items():
    ax.semilogy(levels, vals, ls="--" if k == "bary" else "-", marker="o" if v == 5 else "s", ms=4.5,
                color=C["normal"] if v == 5 else C["tangent"], lw=1.4,
                label=("barycentric" if k == "bary" else "mixed") + f", valence {v}")
ax.set_xticks(levels)
ax.set_xlabel("icosphere level")
ax.set_ylabel(r"$\max\,|K_i - 1|$")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("(a)", loc="left", fontsize=12)
ax2.semilogx(ns, Km, "-o", color=C["tangent"], ms=4.5, lw=1.4, label="mixed")
ax2.semilogx(ns, Kb, "--o", color=C["normal"], ms=4.5, lw=1.4, label="barycentric")
ax2.axhline(-4, color=C["aux"], lw=0.8, ls=(0, (3, 2)))
ax2.text(30, -4.12, r"$K = -4$", color=C["aux"], fontsize=10)
ax2.set_xticks(ns)
ax2.set_xticklabels([str(n) for n in ns])
ax2.set_xlabel(r"grid size $n$")
ax2.set_ylabel(r"$K_i$ at the origin")
ax2.set_ylim(-4.2, -2.7)
ax2.legend(fontsize=8.5, frameon=False, loc="center right")
ax2.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.9, wspace=0.36)
dgfig.save(fig, __file__)
print("icosphere errors", {k: np.round(v, 5).tolist() for k, v in err.items()})
print("saddle origin mixed", np.round(Km, 4), "bary", np.round(Kb, 4))

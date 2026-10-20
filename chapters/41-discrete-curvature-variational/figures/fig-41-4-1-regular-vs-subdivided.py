"""Figure 41.4.1: the same surface, the same valence 6, two refinement sequences.

Torus of revolution T_{2,0.8}, outward normal. Maximum errors over all vertices of four discrete curvatures:
K_i = delta_i / A_i, H_i = <(L x)_i, N_i> / (2 A_i) (dgdec.mean_curvature), the Steiner H_i = -(1/4) sum l theta / A_i,
and kappa_{1,i} (Definition 41.3.3); A_i is the mixed area. h = mean edge length.
(a) regular grids torus_mesh(2, 0.8, n, 2n), n = 12, 24, 48, 96 (gray guide line: slope 2);
(b) subdivided meshes: torus_mesh(2, 0.8, 8, 16) refined 1-4 times by midpoint subdivision with every new vertex projected to
    the torus (all vertices still have valence 6).

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-4-1-regular-vs-subdivided.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS


def errors(V, F):
    u, K, H, k1, k2 = ddg41.torus_exact(V)
    A = dm.mass_voronoi(V, F)
    Kd, Hd = dm.angle_defect(V, F) / A, dd.mean_curvature(V, F)
    a, b, cl = ddg41.principal_curvatures(Hd, Kd)
    assert cl.sum() == 0
    assert np.all(np.array([len(s) for s in dm.neighbors(F, len(V))]) == 6)
    return [ddg41.mean_edge_length(V, F), np.abs(Kd - K).max(), np.abs(Hd - H).max(),
            np.abs(dm.mean_curvature_dihedral(V, F, mass=A) - H).max(), np.abs(a - k1).max()]


reg = np.array([errors(*dm.torus_mesh(2, 0.8, n, 2 * n)) for n in (12, 24, 48, 96)])
sub = np.array([errors(*ddg41.subdivided_torus(k)[:2]) for k in (1, 2, 3, 4)])
order_reg = np.log2(reg[:-1, 1:] / reg[1:, 1:])
assert np.all(order_reg[-1] > 1.9)
assert np.all(sub[-1, 1:] > 0.009) and sub[-1, 1] > 0.02

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
styles = [(r"$\delta_i/A_i$", C["tangent"], "o-"), (r"$H_i$ (cotan)", C["normal"], "s-"),
          (r"$H_i$ (Steiner)", C["accent"], "^--"), (r"$\kappa_{1,i}$", C["third"], "d:")]
for ax, data, title in ((axes[0], reg, "(a) regular grids"), (axes[1], sub, "(b) subdivided")):
    for c, (lab, col, st) in enumerate(styles):
        ax.loglog(data[:, 0], data[:, c + 1], st, color=col, lw=1.3, ms=4, label=lab)
    hh = np.array([0.6, 0.06])
    ax.loglog(hh, 0.08 * hh ** 2, color=C["aux"], lw=0.8)
    ax.set_xlabel(r"mean edge length $h$")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(0.04, 0.7)
    ax.set_ylim(5e-5, 0.08)
axes[0].text(0.17, 6e-4, "slope 2", color=C["aux"], fontsize=9.5)
axes[0].set_ylabel("max error over vertices")
axes[1].legend(fontsize=9, frameon=False, loc="lower right")
for a_ in axes:
    a_.set_xticks([0.05, 0.1, 0.2, 0.4])
    a_.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    a_.minorticks_off()
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.9, wspace=0.08)
dgfig.save(fig, __file__)
print("regular orders", np.round(order_reg, 2).tolist(), "subdivided last", np.round(sub[-1, 1:], 4).tolist())

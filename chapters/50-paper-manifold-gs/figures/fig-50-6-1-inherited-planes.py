"""Figure 50.6.1: a conservative split that keeps the parent's plane cannot bring the varifold closer to the surface.

Start: the n_theta = 4 tiling of the unit sphere (32 atoms, Section 50.2). Level k splits every cell into 2^k x 2^k
sub-cells with one child each (q_j = A(C_j)). Inherit: the child sits at the projection of its sub-cell centre to the
parent's tangent plane, the children are recentred so that sum_j q_j mu_j = q_i mu_i, and each child keeps the parent's
plane (Proposition 50.3.1 with the tangent moment). Re-estimate: the child sits at its sub-cell centre on the sphere with
the true tangent plane. Plotted: the cell-integral upper bound (46.7.6) of Delta^{1,1} to the sphere (Frobenius metric,
markers) and the lower bound by the test function min(1, distance to the atoms) (bars below the markers; for the
re-estimated levels they coincide). Dashed: sum_i int_{C_i} ||P(x) - P_i||_F dA, the plane error of level 0 that the
inherited children keep.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-6-1-inherited-planes.py
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
K = (0, 1, 2, 3)
inh = [g.refined_tiling(4, k, "inherit") for k in K]
ree = [g.refined_tiling(4, k, "reestimate", lower=(k < 3)) for k in K]
pt = g.plane_term(4)
# self-checks: level 0 is the same in both models; re-estimating halves the distance per level; inheriting keeps the
# lower bound above the level-0 plane error
assert abs(inh[0][1] - ree[0][1]) < 1e-12
for a, b in zip(ree[:-1], ree[1:]):
    assert abs(b[1] / a[1] - 0.5) < 0.03
for n, up, lo in inh:
    assert lo > pt and lo <= up + 1e-9
fig, ax = plt.subplots(figsize=(4.8, 3.3))
x = np.array([n for n, _, _ in inh])
ax.semilogy(x, [u for _, u, _ in inh], "-o", color=C["normal"], ms=5, label="inherit the plane")
ax.semilogy(x, [u for _, u, _ in ree], "-s", color=C["tangent"], ms=5, label="re-estimate the plane")
for (n, up, lo) in inh:
    ax.plot([n, n], [lo, up], color=C["normal"], lw=3, alpha=0.5)
ax.axhline(pt, color=C["normal"], ls="--", lw=0.9)
ax.text(2048, pt * 0.92, f"plane error of level 0: {pt:.2f}", fontsize=8.5, color=C["normal"], va="top", ha="right")
ax.set_xscale("log", base=2)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in x])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("number of atoms")
ax.set_ylabel(r"$\Delta^{1,1}$ to the sphere")
ax.set_ylim(0.7, 12)
ax.legend(loc="lower left", fontsize=8.5, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
print("plane term of level 0: %.4f" % pt)
for k, a, b in zip(K, inh, ree):
    print(f"level {k}: atoms {a[0]}, inherit upper {a[1]:.4f} lower {a[2]:.4f}; re-estimate upper {b[1]:.4f} lower {b[2]:.4f}")

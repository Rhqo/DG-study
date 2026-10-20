"""Figure 40.3.2: on the torus the mean curvature normal never flips, the Gauss curvature normal does.

torus_mesh(2, 0.8, 24, 48) (faces outward). The 24 vertices of the meridian v = 0 lie in the xz-plane, and by symmetry
their curvature normals lie in that plane. (a) the pointwise mean curvature normals (L x)_i / A_i (orange, drawn at a
quarter of their length; A_i the mixed area); (b) the pointwise Gauss curvature normals (K N)_i / A_i (green, drawn at half
length). Black: the meridian polygon; gray dot: the center (2, 0) of the tube; gray circles: the points u = +-pi/2 where K = 0.
Exact values: 2 H N with H = -(R + 2 r cos u) / (2 r (R + r cos u)) (outward N) and K N with K = cos u / (r (R + r cos u)).
Also writes an interactive version of the whole torus with both vector fields.

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-3-2-torus-normals.py
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
R, r = 2.0, 0.8
V, F = dm.torus_mesh(R, r, 24, 48)
AN, Lx, KN = ddg40.curvature_normals(V, F)
A = dm.mass_voronoi(V, F)
ring = np.where((np.abs(V[:, 1]) < 1e-12) & (V[:, 0] > 0))[0]
u = np.arctan2(V[ring, 2], V[ring, 0] - R)
order = np.argsort(u)
ring, u = ring[order], u[order]
Nout = np.c_[np.cos(u), np.zeros_like(u), np.sin(u)]
H = -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))
K = np.cos(u) / (r * (R + r * np.cos(u)))
mH = Lx[ring] / A[ring, None]
mK = KN[ring] / A[ring, None]
assert np.allclose(mH[:, 1], 0, atol=1e-12) and np.allclose(mK[:, 1], 0, atol=1e-12)
eH = np.abs(mH - 2 * H[:, None] * Nout).max()
eK = np.abs(mK - K[:, None] * Nout).max()
assert eH < 0.02 and eK < 0.02
assert np.all(np.sum(mH * Nout, 1) < 0)                                   # always toward the tube center
assert np.all(np.sign(np.round(np.sum(mK * Nout, 1), 3)) == np.sign(np.round(K, 3)))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
t = np.linspace(0, 2 * np.pi, 400)
for ax, vec, s, col, title in [(axes[0], mH, 0.25, C["accent"], "(a)"), (axes[1], mK, 0.5, C["third"], "(b)")]:
    ax.plot(V[ring, 0].tolist() + [V[ring[0], 0]], V[ring, 2].tolist() + [V[ring[0], 2]], color=C["main"], lw=1.2)
    ax.plot(R, 0, "o", color=C["aux"], ms=4)
    ax.plot([R, R], [r, -r], "o", ms=7, mfc="none", mec=C["aux"], mew=1.0)
    for p, w in zip(V[ring], vec):
        q = np.array([p[0], p[2]])
        dq = s * np.array([w[0], w[2]])
        if np.linalg.norm(dq) > 1e-3:
            ax.annotate("", q + dq, q, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=8))
    ax.plot(V[ring, 0], V[ring, 2], ".", color=C["main"], ms=3)
    ax.set_aspect("equal")
    ax.set_xlim(R - r - 0.75, R + r + 0.75)
    ax.set_ylim(-r - 0.6, r + 0.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$z$")
    ax.set_title(title, loc="left", fontsize=12)
axes[0].text(0.03, 0.9, r"$\frac{1}{4}(L\mathbf{x})_i/A_i$", color=C["accent"], fontsize=11, transform=axes[0].transAxes)
axes[1].text(0.03, 0.9, r"$\frac{1}{2}(K\mathbf{N})_i/A_i$", color=C["third"], fontsize=11, transform=axes[1].transAxes)
for ax in axes:
    ax.text(R + 0.12, r + 0.12, r"$K=0$", color=C["aux"], fontsize=9.5)
axes[1].text(R - r - 0.7, -0.08, r"$K<0$", color=C["aux"], fontsize=10)
axes[1].text(R + r + 0.3, -0.08, r"$K>0$", color=C["aux"], fontsize=10)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.9, wspace=0.3)
dgfig.save(fig, __file__)

# interactive version: the whole torus with both fields (every second vertex)
sel = np.arange(0, len(V), 2)
tr = [dm.plotly_mesh(V, F, color="#D0D0D0", name="torus_mesh(2, 0.8, 24, 48)"), dm.plotly_edges(V, F, width=1)]
tr += dm.plotly_arrows(V[sel], 0.25 * Lx[sel] / A[sel, None], color="#E69F00", cone=0.04, name="(Lx)_i / 4 A_i")
tr += dm.plotly_arrows(V[sel], 0.5 * KN[sel] / A[sel, None], color="#009E73", cone=0.04, name="(KN)_i / 2 A_i")
dm.plotly_save(tr, __file__, "Torus: mean curvature normals (orange) and Gauss curvature normals (green)", "fig-40-3-2")
print(f"max error against 2 H N: {eH:.4f}, against K N: {eK:.4f}; u = 0: {mH[np.argmin(np.abs(u))]}, {mK[np.argmin(np.abs(u))]}")
print(f"u = pi: {mH[np.argmin(np.abs(np.abs(u) - np.pi))]}, {mK[np.argmin(np.abs(np.abs(u) - np.pi))]}")

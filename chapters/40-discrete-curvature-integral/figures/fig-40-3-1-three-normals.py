"""Figure 40.3.1: the three discrete curvature normals at a convex vertex and at a saddle vertex.

(a) the convex star of Figure 40.1.1 (ddg40.example_star): at x_i the discrete vector area (1/3) sum A_f N_f
(vermilion), the discrete mean curvature normal (L x)_i (orange, drawn at half length) and the discrete Gauss curvature
normal (K N)_i = (1/2) sum theta_ij (x_i - x_j) / l_ij (green); thin blue arrows: the six terms w_ij (x_j - x_i),
w_ij = (cot a + cot b) / 2, whose sum is (L x)_i (drawn at full length). The vectors A_i and (K N)_i are drawn
from points shifted by -+(0.07, 0.07, 0) so that they do not overlap.
(b) the saddle star ddg40.saddle_star(0.35): the same three vectors; here (L x)_i = 0 by symmetry and (K N)_i points
downward (delta_i = -1.084 < 0).

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-3-1-three-normals.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402

C = dgfig.COLORS
fig = plt.figure(figsize=(7.0, 3.6))
for p, (V, F, title) in enumerate([(*ddg40.example_star(), "(a)"), (*ddg40.saddle_star(0.35), "(b)")]):
    AN, Lx, KN = ddg40.curvature_normals(V, F)
    w = dm.cotan_weights(V, F)
    terms = [w[(0, j)] * (V[j] - V[0]) for j in range(1, len(V))]
    assert np.allclose(np.sum(terms, 0), Lx[0], atol=1e-12)
    assert np.allclose(-ddg40.boundary_integral_N_cross_dx(V, F, 0), Lx[0], atol=1e-12)
    assert np.allclose(0.5 * ddg40.boundary_integral_N_cross_dN(V, F, 0), KN[0], atol=1e-12)
    if p == 0:
        assert AN[0, 2] > 0 and Lx[0, 2] < 0 and KN[0, 2] > 0
    else:
        assert np.allclose(Lx[0], 0, atol=1e-12) and KN[0, 2] < 0 < AN[0, 2]
    ax = dgfig.axes3d(fig, 121 + p, elev=16, azim=-62)
    ax.set_position([0.5 * p, 0.0, 0.5, 1.0])
    col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.6, alpha=0.45, zorder=1))
    for t in terms:
        dgfig.arrow3d(ax, V[0], t, role="tangent", zorder=6, head=6, lw=1.0)
    sh = np.array([0.07, 0.07, 0.0])                       # side offset so that A_i and (K N)_i do not overlap
    dgfig.arrow3d(ax, V[0] - sh, AN[0], role="normal", zorder=9, head=9)
    if np.linalg.norm(Lx[0]) > 1e-9:
        dgfig.arrow3d(ax, V[0], 0.5 * Lx[0], role="accent", zorder=8, head=9)
    dgfig.arrow3d(ax, V[0] + sh, KN[0], role="third", zorder=10, head=9, lw=1.6)
    dgfig.point3d(ax, V[0], None, size=14)
    ax.text(*(V[0] - sh + AN[0] + np.array([-0.2, 0.0, 0.03])), r"$\mathbf{A}_i$", color=C["normal"], fontsize=11)
    ax.text(*(V[0] + sh + KN[0] + np.array([0.06, 0.0, 0.0 if p == 0 else -0.08])), r"$(K\mathbf{N})_i$", color=C["third"],
            fontsize=11)
    if p == 0:
        ax.text(*(V[0] + 0.5 * Lx[0] + np.array([0.06, 0.0, -0.05])), r"$\frac{1}{2}(L\mathbf{x})_i$", color=C["accent"],
                fontsize=11)
    else:
        ax.text2D(0.05, 0.08, r"$(L\mathbf{x})_i = 0$", color=C["accent"], transform=ax.transAxes, fontsize=11)
    pts = np.vstack([V, V[0] + AN[0], V[0] + 0.5 * Lx[0], V[0] + KN[0]])
    dgfig.equal_aspect(ax, pts, zoom=1.45)
    ax.set_title(title, loc="left", fontsize=12, y=0.93)
    print(title, "A_i", np.round(AN[0], 4), "(Lx)_i", np.round(Lx[0], 4), "(KN)_i", np.round(KN[0], 4))
dgfig.save(fig, __file__)

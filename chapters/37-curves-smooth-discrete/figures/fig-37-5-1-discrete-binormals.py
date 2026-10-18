"""Figure 37.5.1: discrete binormals and torsion angles of a sampled helix.

(a) the helix (a, b) = (1, 0.3) sampled with 8 vertices per turn (t_j = 2 pi j / 8, 1.5 turns): the planes through three
consecutive vertices (shaded triangles) and their unit normals, the discrete binormals b_i; (b) the plane orthogonal
to the edge tangent t_4, seen from the tip of t_4: b_4 and b_5 both lie in it and the torsion angle psi_4 is the
signed angle between them (right-hand rule about t_4).

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-5-1-discrete-binormals.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
n_turn = 8
tj = 2 * np.pi * np.arange(13) / n_turn
X = ddg37.helix_point(tj)
B = dd.vertex_binormals(X, False)
psi = dd.torsion_angles(X, False)
T = dd.curve_tangents(X, False)
ell, th_c, psi_c = ddg37.helix_discrete_angles(2 * np.pi / n_turn)
assert np.allclose(psi[1:-1], psi_c) and psi_c > 0                  # right-handed: psi > 0
assert np.allclose(np.sum(B[1:-1] * T[:-1][0:len(B) - 2], 1), 0)     # b_i is orthogonal to t_{i-1}
i0 = 2

fig = plt.figure(figsize=(7.0, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
ax = dgfig.axes3d(fig, gs[0], elev=18, azim=120)
dgfig.curve3d(ax, X, lw=1.5)
ax.scatter(*X.T, s=8, color=C["main"], depthshade=False, zorder=6)
for i in range(1, len(X) - 1):
    if i in (i0, i0 + 1, 7):
        tri = Poly3DCollection([X[i - 1:i + 2]], facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=2)
        ax.add_collection3d(tri)
    dgfig.arrow3d(ax, X[i], 0.55 * B[i], role="third", head=7, lw=1.2,
                  label=r"$\mathbf{b}_{%d}$" % i if i in (i0, i0 + 1) else None, fontsize=10.5)
dgfig.arrow3d(ax, X[i0] + 0.5 * (X[i0 + 1] - X[i0]) - 0.25 * T[i0] - 0.22 * B[i0], 0.5 * T[i0], role="tangent", head=7, lw=1.3,
              label=r"$\mathbf{t}_{%d}$" % i0, fontsize=10.5)
dgfig.equal_aspect(ax, np.vstack([X + 0.55, X - 0.55]), zoom=1.75)
ax.set_title("(a)", fontsize=12, loc="left")

# (b) the normal plane of edge i0, seen from the tip of t_{i0}
bx = fig.add_subplot(gs[1])
xh = B[i0]
yh = np.cross(T[i0], B[i0])
b2 = np.array([np.dot(B[i0 + 1], xh), np.dot(B[i0 + 1], yh)])
assert abs(np.arctan2(b2[1], b2[0]) - psi[i0]) < 1e-12
for v, lab in ((np.array([1.0, 0.0]), r"$\mathbf{b}_{%d}$" % i0), (b2, r"$\mathbf{b}_{%d}$" % (i0 + 1))):
    bx.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["third"], lw=1.6,
                                                          mutation_scale=12))
    bx.text(1.08 * v[0] + 0.03, 1.08 * v[1] + 0.02, lab, color=C["third"], fontsize=12)
bx.add_patch(Arc((0, 0), 1.4, 1.4, theta1=0, theta2=np.degrees(psi[i0]), color=C["accent"], lw=1.6))
bx.text(0.3, -0.32, r"$\psi_{%d} = %.3f$" % (i0, psi[i0]), color=C["accent"], fontsize=11)
bx.plot([0], [0], "o", ms=9, mfc="white", mec=C["tangent"], mew=1.3)
bx.plot([0], [0], "o", ms=2.5, color=C["tangent"])
bx.text(-0.35, -0.2, r"$\mathbf{t}_{%d}$" % i0, color=C["tangent"], fontsize=12)
circ = np.linspace(0, 2 * np.pi, 200)
bx.plot(np.cos(circ), np.sin(circ), color=C["aux"], lw=0.6, ls=":")
bx.set_aspect("equal")
bx.set_xlim(-1.2, 1.5)
bx.set_ylim(-1.2, 1.2)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")
fig.subplots_adjust(left=0, right=0.98, bottom=0.02, top=0.93, wspace=0.05)
dgfig.save(fig, __file__)

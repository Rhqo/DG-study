"""Figure 41.1.2: the four gradients of the bending angle of a hinge (Proposition 41.1.8, Example 41.1.9).

x_i = (0, 0, 0), x_j = (2, 0, 0), x_k = (0.6, 1.2, 0), x_l = (1.3, -1, -0.7); f = (i, j, k), g = (j, i, l);
theta = 0.6107 (35.0 degrees). Arrows (vermilion: g_k, g_l, along the face normals; blue: g_i, g_j) at full length.
Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-1-2-hinge.py
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
import ddg41  # noqa: E402

C = dgfig.COLORS
X = np.array([[0, 0, 0.0], [2, 0, 0], [0.6, 1.2, 0], [1.3, -1.0, -0.7]])
th = ddg41.bending_angle(*X)
g = np.array(ddg41.bending_angle_gradients(*X))
num = ddg41.central_gradient(lambda Y: ddg41.bending_angle(*Y), X)
assert np.abs(num - g).max() < 1e-8
assert abs(np.degrees(th) - 34.992) < 1e-2
assert np.allclose(g.sum(0), 0) and np.allclose(np.cross(X, g).sum(0), 0) and abs(np.sum(X * g)) < 1e-12

fig = plt.figure(figsize=(5.6, 4.2))
ax = dgfig.axes3d(fig, elev=18, azim=-75)
for tri, col in [([0, 1, 2], "#D0D0D0"), ([1, 0, 3], "#BDBDBD")]:
    ax.add_collection3d(Poly3DCollection([X[tri]], facecolor=col, edgecolor=C["main"], lw=1.0, alpha=0.75, zorder=1))
ax.plot(*X[[0, 1]].T, color=C["main"], lw=2.2, zorder=2)
labels = [r"$\mathbf{x}_i$", r"$\mathbf{x}_j$", r"$\mathbf{x}_k$", r"$\mathbf{x}_l$"]
offs = [(-0.22, 0, 0.05), (0.2, 0.0, 0.05), (-0.05, 0.18, 0.08), (0.05, -0.2, -0.1)]
for p, lab, o in zip(X, labels, offs):
    ax.scatter(*p, color=C["main"], s=10, zorder=6)
    ax.text(*(p + np.array(o)), lab, fontsize=12, zorder=7)
roles = ["tangent", "tangent", "normal", "normal"]
names = [r"$\nabla_{\mathbf{x}_i}\theta$", r"$\nabla_{\mathbf{x}_j}\theta$", r"$\nabla_{\mathbf{x}_k}\theta$",
         r"$\nabla_{\mathbf{x}_l}\theta$"]
loff = [(-0.25, 0, 0.05), (0.3, 0, 0.0), (0.12, 0, -0.05), (0.15, 0.1, -0.1)]
for p, v, r, n, o in zip(X, g, roles, names, loff):
    dgfig.arrow3d(ax, p, v, role=r, label=n, label_offset=o, fontsize=11, zorder=8)
dgfig.equal_aspect(ax, X, X + g, zoom=0.95)
dgfig.save(fig, __file__)

tr = [dm.plotly_mesh(X, np.array([[0, 1, 2], [1, 0, 3]]), name="hinge"),
      dm.plotly_edges(X, np.array([[0, 1, 2], [1, 0, 3]]), width=3)]
tr += dm.plotly_arrows(X[2:], g[2:], color="#D55E00", cone=0.08, name="grad theta at x_k, x_l")
tr += dm.plotly_arrows(X[:2], g[:2], color="#0072B2", cone=0.08, name="grad theta at x_i, x_j")
dm.plotly_save(tr, __file__, "Gradients of the bending angle of a hinge (vermilion: x_k, x_l; blue: x_i, x_j)",
               "fig-41-1-2")
print("theta", th, "gradients", np.round(g, 4).tolist())

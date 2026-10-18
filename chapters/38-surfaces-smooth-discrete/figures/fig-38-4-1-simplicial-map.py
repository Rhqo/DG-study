"""Figure 38.4.1: a simplicial surface (combinatorics only) and a simplicial map that gives it a shape.

(a) the octahedron's abstract simplicial surface drawn as a planar graph (schematic): the outer face (2, 0, 5) is
fixed on a large triangle and the other vertices are placed by uniform convex combinations (a Tutte drawing). The face
(0, 2, 4) is shaded and contains the point p with barycentric coordinates (0.5, 0.3, 0.2).
(b) the simplicial map x with x_i = the six unit vectors (ddg38.octahedron): x(p) = 0.5 x_0 + 0.3 x_2 + 0.2 x_4,
and the discrete differential (dx)_{4j} = x_j - x_4 on the four edges at vertex 4.
Also writes an interactive plotly version.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-4-1-simplicial-map.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402

C = dgfig.COLORS
V, F = ddg38.octahedron()
E, index = dd.edge_list(F)
lam = {0: 0.5, 2: 0.3, 4: 0.2}

# planar drawing: outer triangle (2, 0, 5) fixed, vertices 1, 3, 4 at the average of their neighbors
P = np.zeros((6, 2))
outer = {2: (-1.9, -1.2), 0: (1.9, -1.2), 5: (0.0, 2.1)}
for k, xy in outer.items():
    P[k] = xy
nb = dm.neighbors(F)
inner = [1, 3, 4]
A = np.zeros((3, 3))
b = np.zeros((3, 2))
for r, i in enumerate(inner):
    A[r, r] = len(nb[i])
    for j in nb[i]:
        if j in inner:
            A[r, inner.index(j)] -= 1
        else:
            b[r] += P[j]
P[inner] = np.linalg.solve(A, b)

# self-checks: closed oriented surface, x(p) on the face, edge vectors are d0 V, d1 d0 = 0, Tutte drawing has no fold
assert dm.is_manifold(F) and dm.oriented_consistently(F) and dm.euler_characteristic(F) == 2
xp = sum(w * V[k] for k, w in lam.items())
nf = np.cross(V[2] - V[0], V[4] - V[0])
assert abs((xp - V[0]) @ nf) < 1e-12
dX = dd.d0(F, 6) @ V
for j in nb[4]:
    e = index[(min(4, j), max(4, j))]
    assert np.allclose(dX[e], (V[j] - V[4]) * (1 if 4 < j else -1))
assert np.abs(dd.d1(F) @ dX).max() < 1e-14
det2 = lambda a, c: a[0] * c[1] - a[1] * c[0]  # noqa: E731
signs = [np.sign(det2(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])) for t in F if not set(t) == {0, 2, 5}]
assert len(set(signs)) == 1                                   # every inner face has the same orientation in the drawing

fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0.01, 0.04, 0.42, 0.9])
for f in F:
    if set(f) == {0, 2, 4}:
        ax.add_patch(Polygon(P[f], closed=True, fc=C["region"], alpha=0.35, ec="none"))
for i, j in E:
    ax.plot(*P[[i, j]].T, color=C["main"], lw=1.1)
for i in range(6):
    ax.plot(*P[i], "o", color=C["main"], ms=5)
    off = P[i] - P[inner].mean(0)
    off = 0.2 * off / max(np.linalg.norm(off), 1e-9)
    ax.text(*(P[i] + off + np.array([-0.05, -0.04])), str(i), fontsize=12)
pp = sum(w * P[k] for k, w in lam.items())
ax.plot(*pp, "o", color=C["accent"], ms=6)
ax.text(pp[0] + 0.08, pp[1] - 0.18, r"$p$", fontsize=12, color=C["accent"])
dgfig.schematic_axes(ax)
ax.set_title("(a)", loc="left", fontsize=12)

ax3 = dgfig.axes3d(fig, 111, elev=22, azim=-55)
ax3.set_position([0.43, 0.0, 0.57, 1.0])
col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
ax3.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.6, alpha=0.55))
for j in nb[4]:
    dgfig.arrow3d(ax3, V[4], V[j] - V[4], role="tangent", zorder=9, head=9, lw=1.4)
for i in range(6):
    dgfig.point3d(ax3, V[i], None, size=10)
lab = {0: (0.08, -0.05, -0.1), 1: (-0.15, 0.05, 0.05), 2: (0.05, 0.08, 0.05), 3: (-0.05, -0.2, -0.05),
       4: (0.05, 0.05, 0.1), 5: (0.05, 0.0, -0.18)}
for i, o in lab.items():
    ax3.text(*(V[i] + np.array(o)), rf"$\mathbf{{x}}_{i}$", fontsize=12)
dgfig.point3d(ax3, xp, None, role="accent", size=22)
ax3.text(*(xp + np.array([0.12, -0.05, -0.02])), r"$\mathbf{x}(p)$", color=C["accent"], fontsize=12)
ax3.text(*(0.5 * (V[4] + V[3]) + np.array([0.05, -0.1, 0.05])), r"$(d\mathbf{x})_{43}$", color=C["tangent"], fontsize=11)
dgfig.equal_aspect(ax3, V, zoom=1.25)
ax3.set_title("(b)", loc="left", fontsize=12, y=0.93)
dgfig.map_arrow(fig, ax, ax3, r"$\mathbf{x}$", xy_from=(0.92, 0.82), xy_to=(0.2, 0.86), rad=-0.3)
dgfig.save(fig, __file__)

tr = [dm.plotly_mesh(V, F, opacity=0.6, name="octahedron"), dm.plotly_edges(V, F)]
tr += dm.plotly_arrows(np.repeat(V[4:5], 4, 0), np.array([V[j] - V[4] for j in nb[4]]), color=C["tangent"], cone=0.15,
                       name="(dx)_4j")
dm.plotly_save(tr, __file__, "Octahedron: simplicial map x with x_i = +-e_k; blue arrows: (dx)_4j = x_j - x_4",
               "fig-38-4-1")

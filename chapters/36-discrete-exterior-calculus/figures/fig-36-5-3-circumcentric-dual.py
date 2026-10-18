"""Figure 36.5.3: the circumcentric dual of a planar mesh, and a dual edge of negative length (Section 36.5).

(a) the triangular lattice patch lib/ch36.equilateral_patch(2) (19 vertices, 24 triangles, spacing 1) with its
    vertices moved by small seeded random offsets (all triangles stay acute): primal edges gray, circumcenters and
    circumcentric dual edges orange (a dual edge of a boundary edge stops at the edge midpoint, dashed); the dual
    cell 0* of the center vertex is filled. Every dual edge is perpendicular to its primal edge.
(b) the non-Delaunay pair lib/ch36.non_delaunay_pair (x_0 = (0, 0), x_1 = (2, 0), x_2 = (1, 0.3), x_3 = (1, -0.25),
    faces f = (0, 1, 2) above and g = (1, 0, 3) below the shared edge e = [x_0, x_1]). Both angles opposite e are
    obtuse (alpha + beta = 298.5 deg); the circumcenter c_f lies below e and c_g above e, so the dual edge c_g -> c_f
    (vermillion) points opposite to e turned by +90 degrees (gray dashed): its signed length is -3.392 and
    star1_e = -1.696.
Self-check: perpendicularity, all star1 > 0 in (a), the values in (b), dual cell areas sum to the total area in (a).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-5-3-circumcentric-dual.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

rng = np.random.default_rng(365)
V, F = ch36.equilateral_patch(2)
V = V + 0.09 * rng.uniform(-1, 1, size=V.shape)
V3 = np.c_[V, np.zeros(len(V))]
ang = dm.corner_angles(V3, F)
assert ang.max() < np.pi / 2
E, index = dd.edge_list(F)
s1 = dd.hodge1(V, F)
assert s1.min() > 0
C = dd.circumcenters(V, F)
polys = ch36.dual_edge_polylines(V, F)
lr = ch36.left_right_faces(F)
for e, (i, j) in enumerate(E):
    t = V[j] - V[i]
    assert all(abs((q - polys[e][1]) @ t) < 1e-12 for q in (polys[e][0], polys[e][2]))
he = dd.HalfedgeMesh(F)
ring = he.vertex_faces(0)
cell = []
for a, b in zip(ring, ring[1:] + ring[:1]):
    shared = sorted(set(F[a]) & set(F[b]))
    cell += [C[a], 0.5 * (V[shared[0]] + V[shared[1]])]
cell = np.array(cell)
area0 = ch36.signed_polygon_area(cell)
assert abs(area0 - dd.dual_areas(V, F, "circumcentric")[0]) < 1e-12
assert abs(dd.dual_areas(V, F, "circumcentric").sum() - dm.face_areas(V3, F).sum()) < 1e-12

Vn, Fn = ch36.non_delaunay_pair()
En, idxn = dd.edge_list(Fn)
e01 = idxn[(0, 1)]
angn = dm.corner_angles(np.c_[Vn, np.zeros(4)], Fn)
ab = np.degrees(angn[0, 2] + angn[1, 2])
Cn = dd.circumcenters(Vn, Fn)
st, mid, en = ch36.dual_edge_polylines(Vn, Fn)[e01]
signed = (en - st) @ np.array([0.0, 1.0])                 # e = +x, so J t = +y
s1n = dd.hodge1(Vn, Fn)[e01]
assert abs(ab - 298.5) < 0.05 and abs(signed + 3.392) < 1e-3 and abs(s1n + 1.696) < 1e-3
assert abs(signed / 2.0 - s1n) < 1e-12 and Cn[0][1] < 0 < Cn[1][1]

col = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
ch36.draw_mesh2d(axa, V, F, face="#F6F6F6", edge="#9A9A9A", lw=0.8)
axa.add_patch(Polygon(cell, closed=True, facecolor=col["accent"], alpha=0.3, edgecolor="none", zorder=1))
for e in range(len(E)):
    P = polys[e]
    boundary = lr[e].min() < 0
    axa.plot(*P.T, color=col["accent"], lw=1.4, ls=(0, (3, 2)) if boundary else "-", zorder=3)
axa.plot(*C.T, "o", color=col["accent"], ms=3, zorder=4)
axa.plot(*V[0], "o", color="black", ms=4, zorder=5)
axa.text(*(V[0] + np.array([0.12, -0.2])), r"$0^\star$", fontsize=11, color="#8a5a00", zorder=6)
for e in (index[(0, 1)], index[(0, 2)]):
    m = polys[e][1]
    i, j = E[e]
    t = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i])
    n = np.array([-t[1], t[0]])
    s = 0.09
    axa.plot(*np.array([m + s * t, m + s * t + s * n, m + s * n]).T, color="#555555", lw=0.8, zorder=6)
axa.set_title("(a) circumcentric dual (Delaunay)", fontsize=11)
axa.set_aspect("equal")
axa.set_axis_off()
axa.autoscale_view()

ch36.draw_mesh2d(axb, Vn, Fn, face="#F6F6F6", edge="#777777", lw=1.0)
axb.plot(*Vn[[0, 1]].T, color=col["tangent"], lw=2.4, zorder=3)
axb.add_patch(FancyArrowPatch(st, en, arrowstyle="-|>", mutation_scale=14, color=col["normal"], lw=2.2, zorder=5))
axb.add_patch(FancyArrowPatch(mid + np.array([0.45, 0.0]), mid + np.array([0.45, 1.0]), arrowstyle="-|>",
                              mutation_scale=12, color="#888888", lw=1.4, ls=(0, (3, 2)), zorder=4))
axb.text(*(mid + np.array([0.55, 0.8])), r"$Jt$", fontsize=11, color="#666666")
for c, name in ((Cn[0], r"$c_f$"), (Cn[1], r"$c_g$")):
    axb.plot(*c, "o", color=col["accent"], ms=6, zorder=6)
    axb.text(*(c + np.array([0.12, -0.05])), name, fontsize=12, color="#8a5a00")
for v, off in zip(range(4), ([-0.2, 0.0], [0.08, 0.05], [0.05, 0.12], [0.05, -0.28])):
    axb.text(*(Vn[v] + np.array(off)), rf"$x_{v}$", fontsize=11)
axb.text(1.95, 0.35, r"$f$", fontsize=12)
axb.text(1.95, -0.6, r"$g$", fontsize=12)
axb.text(1.0, -2.45, r"$\alpha + \beta = 298.5^\circ$" + "\n" + r"$|e^\star| = -3.392$,  $\star_1 = -1.696$",
         fontsize=10.5, ha="center", va="top")
axb.set_title(r"(b) obtuse pair: $|e^\star| < 0$", fontsize=11)
axb.set_xlim(-0.5, 2.6)
axb.set_ylim(-3.2, 2.3)
axb.set_aspect("equal")
axb.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

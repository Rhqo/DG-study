"""Figure 33.5.2: halfedges, twin and next, and the two rotations around a vertex (Section 33.5).

The mesh of Figure 33.5.1 (a closed fan of five triangles (0, t+1, t+2) around vertex 0, counterclockwise rows),
stored as dgdec.HalfedgeMesh (one boundary halfedge per rim edge).
(a) every interior halfedge drawn as an arrow pushed toward the centroid of its triangle; boundary halfedges outside
    the rim, gray dashed. Highlighted: h = 0 -> 1, its twin eta(h) = 1 -> 0 and its next rho(h) = 1 -> 2.
(b) the five halfedges leaving vertex 0, numbered in the order of eta o rho^{-1} (twin o prev, counterclockwise) from
    h; the arcs show the two rotations eta o rho^{-1} (counterclockwise) and rho o eta (next o twin, clockwise).
Self-check: next has order 3 on interior halfedges, twin is a fixed-point-free involution with tail(twin h) = head(h),
outgoing(0) visits 1, 2, 3, 4, 5 with increasing angle, and rho o eta visits them in the reverse order.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-5-2-halfedges.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
ang = np.deg2rad(90 + 72 * np.arange(5))
X = np.vstack([[0, 0], np.c_[np.cos(ang), np.sin(ang)]])
F = np.array([(0, 1 + t, 1 + (t + 1) % 5) for t in range(5)])
he = dd.HalfedgeMesh(F)
ni = 3 * len(F)
assert np.all(he.next[he.next[he.next[:ni]]] == np.arange(ni))
assert np.all(he.twin[he.twin] == np.arange(he.nh)) and np.all(he.twin != np.arange(he.nh))
assert np.all(he.vertex[he.twin] == he.head)
out = he.outgoing(0)
assert [int(he.head[h]) for h in out] == [1, 2, 3, 4, 5]
crane = [out[0]]
for _ in range(4):
    crane.append(int(he.next[he.twin[crane[-1]]]))
assert [int(he.head[h]) for h in crane] == [1, 5, 4, 3, 2]          # next o twin turns the other way


def arrow(ax, p, q, color, lw=1.5, ls="-", ms=11, z=5, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                                 alpha=alpha, shrinkA=0, shrinkB=0))


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.5))
for ax in (axa, axb):
    for f in F:
        ax.add_patch(Polygon(X[f], closed=True, facecolor="#EEEEEE", edgecolor="#9A9A9A", lw=0.8, zorder=0))
    for v in range(6):
        ax.plot(*X[v], "o", color="black", ms=4, zorder=6)
        q = X[v] * 1.16 if v else np.array([-0.12, -0.13])
        ax.text(*q, str(v), fontsize=12, ha="center", va="center", zorder=7)

# ---------------------------------------------------------------- (a) halfedges
h0 = out[0]                                                     # 0 -> 1
special = {h0: (C["accent"], r"$h$"), int(he.twin[h0]): (C["normal"], r"$\eta(h)$"),
           int(he.next[h0]): (C["third"], r"$\rho(h)$")}
# label position along the arrow (0 = tail, 1 = head): h and rho(h) lie in the same triangle, so the label of h
# sits low on its arrow and does not collide with the label of rho(h)
along = {h0: 0.25}
for h in range(he.nh):
    a, b = X[he.vertex[h]], X[he.head[h]]
    if he.face[h] >= 0:
        c = X[F[he.face[h]]].mean(0)
        p, q = a + 0.2 * (c - a), b + 0.2 * (c - b)
        p, q = p + 0.12 * (q - p), q - 0.12 * (q - p)
        col, lab = special.get(h, (C["tangent"], None))
        arrow(axa, p, q, col, lw=2.0 if lab else 1.2, z=6 if lab else 5)
        if lab:
            t = along.get(h, 0.5)
            mid = (1 - t) * p + t * q + 0.16 * (c - (a + b) / 2) / np.linalg.norm(c - (a + b) / 2)
            axa.text(*mid, lab, color=col, fontsize=11.5, ha="center", va="center", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    else:
        nrm = (a + b) / 2 / np.linalg.norm((a + b) / 2)
        p, q = a + 0.12 * nrm, b + 0.12 * nrm
        p, q = p + 0.12 * (q - p), q - 0.12 * (q - p)
        arrow(axa, p, q, C["aux"], lw=0.9, ls=(0, (3, 2)), ms=8, alpha=0.8)
axa.set_title("(a) halfedges", fontsize=12)
dgfig.schematic_axes(axa, xlim=(-1.35, 1.35), ylim=(-1.42, 1.4))

# ---------------------------------------------------------------- (b) rotations around vertex 0
for k, h in enumerate(out):
    d = X[he.head[h]]
    arrow(axb, X[0], d * 0.8, C["tangent"], lw=1.6)
    side = np.array([-d[1], d[0]])
    axb.text(*(d * 0.68 + 0.12 * side), rf"${k}$", fontsize=10.5, color=C["tangent"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.05", fc="#EEEEEE", ec="none"))
th = np.linspace(np.deg2rad(110), np.deg2rad(380), 80)
arc = 0.22 * np.c_[np.cos(th), np.sin(th)]
axb.plot(*arc[:-1].T, color=C["accent"], lw=1.8, zorder=8)
arrow(axb, arc[-3], arc[-1], C["accent"], lw=1.8, z=8)
th2 = np.linspace(np.deg2rad(70), np.deg2rad(-200), 80)
arc2 = 0.4 * np.c_[np.cos(th2), np.sin(th2)]
axb.plot(*arc2[:-1].T, color=C["normal"], lw=1.4, ls=(0, (3, 2)), zorder=8)
arrow(axb, arc2[-3], arc2[-1], C["normal"], lw=1.4, z=8)
axb.text(-1.3, -1.12, r"$\eta\circ\rho^{-1}$: counterclockwise", color="#8a5a00", fontsize=11, ha="left")
axb.text(-1.3, -1.32, r"$\rho\circ\eta$: clockwise", color=C["normal"], fontsize=11, ha="left")
axb.set_title("(b) around vertex 0", fontsize=12)
dgfig.schematic_axes(axb, xlim=(-1.35, 1.35), ylim=(-1.42, 1.4))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)

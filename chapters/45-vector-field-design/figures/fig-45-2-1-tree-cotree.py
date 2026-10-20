"""Figure 45.2.1: a tree-cotree decomposition of a torus mesh and its two generators (Section 45.2, Theorem 45.2.11).

Mesh: dgmesh.torus_mesh(2, 0.8, 8, 16) (128 vertices, 384 edges, 256 faces), drawn in the parameter rectangle:
vertex (i, j) (index 16 i + j, u = 2 pi i / 8 around the tube, v = 2 pi j / 16 around the z-axis) is drawn at
(j, i); the left and right sides (v = 0 and 2 pi) and the bottom and top sides (u = 0 and 2 pi) are identified, so
edges on those sides are drawn twice. ch45.tree_cotree with root vertex 0 and root face 0 (BFS).
(a) primal spanning tree T (blue, 127 edges), dual spanning tree T* (orange, 255 dual edges, drawn through the edge
    midpoints between face centroids), the two leftover edges (vermilion, thick), all other edges light gray.
(b) the two primal loops c_1, c_2 (blue and green, solid) and the two dual loops c*_1, c*_2 (same colors, dashed);
    c*_a crosses c_b only for a = b, at the leftover edge (black ring).
Self-checks: |T| = n0 - 1, |T*| = n2 - 1, two leftover edges, d0^T c_a = 0, d1 c*_a = 0, c*_a . c_b = delta_ab, and the
periods of R_1(du), R_1(dv) on c_1, c_2 form an invertible matrix (so the loops are generators).

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-2-1-tree-cotree.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
nu, nv = 8, 16
V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
tc, C, Cs = ch45.generators(F)
E, left, right = tc["E"], tc["left"], tc["right"]
n0, n1, n2 = dm.counts(F)
D0, D1 = dd.d0(F, n0), dd.d1(F)
assert len(tc["T"]) == n0 - 1 and len(tc["Ts"]) == n2 - 1 and len(tc["rest"]) == 2
assert np.abs(D0.T @ C.T).max() == 0 and np.abs(D1 @ Cs.T).max() == 0 and np.array_equal(Cs @ C.T, np.eye(2))
u = 2 * np.pi * (np.arange(n0) // nv) / nu
v = 2 * np.pi * (np.arange(n0) % nv) / nv
per = np.array([[c @ ch45.wrap(D0 @ u), c @ ch45.wrap(D0 @ v)] for c in C])
assert abs(np.linalg.det(per)) > 1.0
print("periods of (du, dv) on c_1, c_2 / 2 pi:\n", np.round(per / (2 * np.pi), 6))

# unwrapped face corners: face 2 (i nv + j) + t of quad (i, j); corners in the order of F
Cf = np.zeros((n2, 3, 2))
for f in range(n2):
    q, t = divmod(f, 2)
    i, j = divmod(q, nv)
    pos = {(i % nu) * nv + (j % nv): (j, i), (i % nu) * nv + (j + 1) % nv: (j + 1, i),
           ((i + 1) % nu) * nv + j % nv: (j, i + 1), ((i + 1) % nu) * nv + (j + 1) % nv: (j + 1, i + 1)}
    Cf[f] = [pos[int(a)] for a in F[f]]
cen = Cf.mean(1)


def edge_segments(e):
    """the copies of edge e in the drawing: its position in each of its two faces (deduplicated)."""
    out = []
    for f in (left[e], right[e]):
        k = [list(F[f]).index(E[e, 0]), list(F[f]).index(E[e, 1])]
        seg = (tuple(Cf[f, k[0]]), tuple(Cf[f, k[1]]))
        if seg not in out and seg[::-1] not in out:
            out.append(seg)
    return out


def dual_segments(e):
    """the dual edge of e: centroid of the right face -> edge midpoint -> centroid of the left face (per face)."""
    out = []
    for f in (right[e], left[e]):
        k = [list(F[f]).index(E[e, 0]), list(F[f]).index(E[e, 1])]
        out.append([cen[f], Cf[f, k].mean(0)])
    return out


def draw_base(ax):
    ax.add_collection(LineCollection([s for e in range(n1) for s in edge_segments(e)], colors="#D6D6D6",
                                     linewidths=0.5, zorder=0))
    ax.plot([0, nv, nv, 0, 0], [0, 0, nu, nu, 0], color=col["aux"], lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-0.4, nv + 0.4)
    ax.set_ylim(-0.4, nu + 0.4)
    ax.set_aspect("equal")
    ax.set_axis_off()


fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 7.0))
draw_base(axa)
axa.add_collection(LineCollection([s for e in tc["T"] for s in edge_segments(e)], colors=col["tangent"],
                                  linewidths=1.6, zorder=2))
axa.add_collection(LineCollection([s for e in tc["Ts"] for s in dual_segments(e)], colors=col["accent"],
                                  linewidths=1.2, zorder=3))
axa.add_collection(LineCollection([s for e in tc["rest"] for s in edge_segments(e)], colors=col["normal"],
                                  linewidths=3.2, zorder=4))
axa.set_title(r"(a) tree $T$ (blue), cotree $T^\star$ (orange), leftover edges (vermilion)", fontsize=11)
draw_base(axb)
for a, cc in enumerate((col["tangent"], col["third"])):
    segs = [s for e in np.where(C[a] != 0)[0] for s in edge_segments(e)]
    axb.add_collection(LineCollection(segs, colors=cc, linewidths=2.2, zorder=3))
    dsegs = [s for e in np.where(Cs[a] != 0)[0] for s in dual_segments(e)]
    axb.add_collection(LineCollection(dsegs, colors=cc, linewidths=1.4, linestyles=(0, (3, 2)), zorder=4))
    e0 = tc["rest"][a]
    for s in edge_segments(e0):
        m = np.mean(s, 0)
        axb.plot(*m, "o", ms=9, mfc="none", mec=col["main"], mew=1.2, zorder=5)
    s = edge_segments(e0)[0]
    axb.text(*(np.mean(s, 0) + [(0.3, -0.75), (0.3, 0.3)][a]), rf"$c_{a + 1},\ c^\star_{a + 1}$", fontsize=11, color=cc,
             zorder=6, bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
axb.set_title(r"(b) loops $c_a$ (solid) and dual loops $c^\star_a$ (dashed)", fontsize=11)
axb.text(nv / 2, -0.95, r"$v \to$ (around the $z$-axis)", ha="center", fontsize=10)
axb.text(-0.9, nu / 2, r"$u \to$", rotation=90, va="center", fontsize=10)
fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, hspace=0.18)
dgfig.save(fig, __file__)

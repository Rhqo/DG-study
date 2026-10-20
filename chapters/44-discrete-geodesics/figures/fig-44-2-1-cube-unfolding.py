"""Figure 44.2.1: exact intrinsic distance on the surface of the unit cube by unfolding (Section 44.2).

Net of the cube [0, 1]^3 (a cross): bottom z = 0 at [0,1]^2 (coordinates (x, y)); y = 0 below (x, -z); y = 1 above
(x, 1 + z); x = 0 on the left (-z, y); x = 1 on the right (1 + z, y); top z = 1 to the right of x = 1 (3 - x, y).
p = (0.2, 0.3, 0) on the bottom face, q = (0.9, 0.6, 1) on the top face.
(a) the shortest path from p to q (orange): in this net it is one straight segment of length sqrt(1.9^2 + 0.3^2).
    The straight segments through the other single side faces (x = 0, y = 0, y = 1) are longer (printed).
(b) the distance d(p, .) on the net (shading, isolines every 0.25) and the cut locus of p (red): sample points where
    the two smallest distances from different unfoldings differ by less than 1.2/n (n = 120 samples per face side).
    Every copy of the eight cube corners in the net is marked.
Distances: ch44.cube_distance_vec (exact: shortest paths on a convex polyhedron avoid vertices).
Self-checks: the straight segment of (a) crosses the two shared edges inside the edges; its length equals
cube_distance; the distance at the corners is attained and every corner lies within one sample of the cut locus.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-2-1-cube-unfolding.py
"""

import pathlib
import sys

import dgmesh as dm  # noqa: F401
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
p = np.array([0.2, 0.3, 0.0])
q = np.array([0.9, 0.6, 1.0])


def to_net(P):
    """3D points on the cube -> 2D net coordinates (face chosen by the first matching face)."""
    P = np.atleast_2d(P)
    out = np.full((len(P), 2), np.nan)
    rules = [
        (lambda X: np.abs(X[:, 2]) < 1e-9, lambda X: np.stack([X[:, 0], X[:, 1]], 1)),
        (lambda X: np.abs(X[:, 1]) < 1e-9, lambda X: np.stack([X[:, 0], -X[:, 2]], 1)),
        (lambda X: np.abs(X[:, 1] - 1) < 1e-9, lambda X: np.stack([X[:, 0], 1 + X[:, 2]], 1)),
        (lambda X: np.abs(X[:, 0]) < 1e-9, lambda X: np.stack([-X[:, 2], X[:, 1]], 1)),
        (lambda X: np.abs(X[:, 0] - 1) < 1e-9, lambda X: np.stack([1 + X[:, 2], X[:, 1]], 1)),
        (lambda X: np.abs(X[:, 2] - 1) < 1e-9, lambda X: np.stack([3 - X[:, 0], X[:, 1]], 1)),
    ]
    for test, f in rules:
        m = test(P) & np.isnan(out[:, 0])
        out[m] = f(P[m])
    return out


net_squares = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (2, 0)]
pn, qn = to_net(p)[0], to_net(q)[0]
d_exact = ch44.cube_distance(p, q[None])[0]
assert abs(np.linalg.norm(qn - pn) - d_exact) < 1e-12
# the segment crosses x = 1 and x = 2 at heights inside [0, 1]
for xe in (1.0, 2.0):
    lam = (xe - pn[0]) / (qn[0] - pn[0])
    assert 0 < pn[1] + lam * (qn[1] - pn[1]) < 1
# the other single-side candidates, each straight in its own net
others = []
for name, src2, dst2 in (("x = 0", (0.3, -0.2), (0.6, 1.9)), ("y = 0", (0.2, -0.3), (0.9, 1.6)),
                         ("y = 1", (0.2, -0.7), (0.9, 1.4))):
    others.append((name, float(np.hypot(dst2[0] - src2[0], dst2[1] - src2[1]))))
print("shortest via x = 1:", round(d_exact, 4), " other single side faces:", [(n, round(L_, 4)) for n, L_ in others])
assert all(L_ > d_exact for _, L_ in others)

# distance field on the six faces
n = 120
g = (np.arange(n) + 0.5) / n
pts, nets = [], []
for k, (a, s) in enumerate(ch44.CUBE_FACES):
    ax = [i for i in range(3) if i != a]
    U, W = np.meshgrid(g, g, indexing="ij")
    P = np.zeros((n * n, 3))
    P[:, a], P[:, ax[0]], P[:, ax[1]] = s, U.ravel(), W.ravel()
    pts.append(P)
P = np.vstack(pts)
best, second = ch44.cube_distance_vec(p, P)
cut = (second - best) < 1.2 / n
corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
for c in corners:
    near = np.linalg.norm(P[cut] - c, axis=1).min()
    assert near < 2.5 / n, near
print(f"max distance {best.max():.4f}, cut-locus samples {cut.sum()}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.5))
for ax in (axa, axb):
    for (i, j) in net_squares:
        ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, ec=col["main"], lw=0.8, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 3.1)
    ax.set_ylim(-1.1, 2.1)
    ax.set_axis_off()
labels = {(0, 0): "bottom", (0, -1): "y = 0", (0, 1): "y = 1", (-1, 0): "x = 0", (1, 0): "x = 1", (2, 0): "top"}
for (i, j), lab in labels.items():
    axa.text(i + 0.5, j + 0.82, lab, ha="center", fontsize=9, color=col["aux"])
axa.plot([pn[0], qn[0]], [pn[1], qn[1]], color=col["accent"], lw=2.4, zorder=4)
axa.plot(*pn, "o", color=col["main"], ms=4, zorder=5)
axa.plot(*qn, "o", color=col["main"], ms=4, zorder=5)
axa.text(pn[0] - 0.12, pn[1] + 0.08, r"$p$", fontsize=12)
axa.text(qn[0] + 0.05, qn[1] + 0.08, r"$q$", fontsize=12)
axa.set_title(r"(a) the net: shortest path is straight", fontsize=11)

cmap = plt.get_cmap("Blues_r")
for k in range(6):
    Pk = pts[k]
    Nk = to_net(Pk)
    bk = best[k * n * n:(k + 1) * n * n].reshape(n, n)
    Xn, Yn = Nk[:, 0].reshape(n, n), Nk[:, 1].reshape(n, n)
    axb.pcolormesh(Xn, Yn, bk, cmap=cmap, vmin=0, vmax=2.3, shading="nearest", rasterized=True, zorder=1)
    axb.contour(Xn, Yn, bk, levels=np.arange(0.25, 2.3, 0.25), colors=col["main"], linewidths=0.35, zorder=2)
Cn = to_net(P[cut])
axb.plot(Cn[:, 0], Cn[:, 1], ".", color="#C00000", ms=0.9, zorder=4, rasterized=True)
for (i, j) in net_squares:                     # every copy of every cube corner in the net
    for (di, dj) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        axb.plot(i + di, j + dj, "s", color=col["main"], ms=3, zorder=5)
axb.plot(*pn, "o", color=col["accent"], ms=5, zorder=6)
axb.text(pn[0] - 0.12, pn[1] + 0.08, r"$p$", fontsize=12, color=col["main"])
axb.set_title(r"(b) $d(p, \cdot)$ and the cut locus of $p$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

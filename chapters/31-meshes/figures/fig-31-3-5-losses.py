"""Figure 31.3.5: what the PyTorch3D mesh regularizers do on their own (Section 31.3).

Start: unit icosphere, level 3 (642 vertices), every vertex scaled by 1 + 0.03 n, n ~ N(0, 1) (seed 0).
Each loss alone is minimized by normalized gradient descent x <- x - 0.0005 g / max|g| (the path does not depend
on the step size when it is small). The three losses follow PyTorch3D (Section 31.3):
  Laplacian (uniform):    mean_i |(L_u x)_i|, L_u treated as a constant matrix (as in mesh_laplacian_smoothing)
  normal consistency:     mean over pairs of faces sharing an edge of 1 - cos(n_0, n_1)
  edge length (target 0): mean over edges of |x_i - x_j|^2
Roughness = std / mean of the distances |x_i - c| to the centroid c; size = their mean.
(a) Size against roughness (relative to the start); every curve starts at (1, 1) on the right.
(b) The three meshes at the end (roughness 25% of the start, or 6000 steps for the Laplacian loss), drawn with
    the same scale; dashed circle: radius 1.
Self-check: gradients agree with finite differences; normal consistency keeps the size (scale invariant).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-3-5-losses.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

C = dgfig.COLORS
V0, F = dm.icosphere(3)
rng = np.random.default_rng(0)
V0 = V0 * (1 + 0.03 * rng.normal(size=(len(V0), 1)))
E = dm.edges(F)
Lu = dm.uniform_laplacian(F)
pairs = []
for (i, j), fs in dm.edge_faces(F).items():
    a = [x for x in F[fs[0]] if x not in (i, j)][0]
    b = [x for x in F[fs[1]] if x not in (i, j)][0]
    pairs.append((i, j, a, b))
pairs = np.array(pairs)


def lap_loss(X):
    R = Lu @ X
    nr = np.linalg.norm(R, axis=1)
    return nr.mean(), Lu.T @ (R / nr[:, None]) / len(X)


def edge_loss(X):
    d = X[E[:, 0]] - X[E[:, 1]]
    g = np.zeros_like(X)
    np.add.at(g, E[:, 0], 2 * d / len(E))
    np.add.at(g, E[:, 1], -2 * d / len(E))
    return (d ** 2).sum(1).mean(), g


def nc_loss(X):
    v0, v1, a, b = (X[pairs[:, k]] for k in range(4))
    p, q = v1 - v0, a - v0
    n0 = np.cross(p, q)
    p2, q2 = b - v0, v1 - v0
    n1 = np.cross(p2, q2)
    l0 = np.linalg.norm(n0, axis=1, keepdims=True)
    l1 = np.linalg.norm(n1, axis=1, keepdims=True)
    c = (n0 * n1).sum(1, keepdims=True) / (l0 * l1)
    gn0 = -(n1 / l1 - c * n0 / l0) / l0 / len(pairs)
    gn1 = -(n0 / l0 - c * n1 / l1) / l1 / len(pairs)
    g = np.zeros_like(X)
    gp, gq = np.cross(q, gn0), np.cross(gn0, p)
    np.add.at(g, pairs[:, 1], gp)
    np.add.at(g, pairs[:, 2], gq)
    np.add.at(g, pairs[:, 0], -gp - gq)
    gp2, gq2 = np.cross(q2, gn1), np.cross(gn1, p2)
    np.add.at(g, pairs[:, 3], gp2)
    np.add.at(g, pairs[:, 1], gq2)
    np.add.at(g, pairs[:, 0], -gp2 - gq2)
    return (1 - c).mean(), g


for fn in (lap_loss, edge_loss, nc_loss):  # finite-difference check
    _, g = fn(V0)
    for (i, k) in ((5, 1), (300, 2)):
        Xp, Xm = V0.copy(), V0.copy()
        Xp[i, k] += 1e-6
        Xm[i, k] -= 1e-6
        assert abs((fn(Xp)[0] - fn(Xm)[0]) / 2e-6 - g[i, k]) < 1e-6 * max(1, abs(g[i, k]))


def stats(X):
    r = np.linalg.norm(X - X.mean(0), axis=1)
    return r.std() / r.mean(), r.mean()


r0, s0 = stats(V0)
LOSSES = [("Laplacian (uniform)", lap_loss, C["tangent"]), ("edge length (target 0)", edge_loss, C["third"]),
          ("normal consistency", nc_loss, C["accent"])]
results = {}
for name, fn, col in LOSSES:
    X = V0.copy()
    traj = [stats(X)]
    for it in range(6000):
        _, g = fn(X)
        X = X - 0.0005 * g / np.abs(g).max()
        traj.append(stats(X))
        if traj[-1][0] < 0.25 * r0:
            break
    results[name] = (np.array(traj), X)

tl = results["Laplacian (uniform)"][0]
te = results["edge length (target 0)"][0]
tn = results["normal consistency"][0]
assert tn[-1, 0] < 0.25 * r0 and abs(tn[-1, 1] / s0 - 1) < 0.002          # smooth, same size
assert te[-1, 0] < 0.25 * r0 and te[-1, 1] / s0 < 0.97                     # smooth, shrunk
assert tl[:, 0].min() > 0.5 * r0 and tl[-1, 1] / s0 < 0.95                  # stalls, keeps shrinking

fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.15))
for name, fn, col in LOSSES:
    t = results[name][0]
    ax.plot(t[:, 0] / r0, t[:, 1] / s0, color=col, lw=2.0, label=name)
    ax.plot(t[-1, 0] / r0, t[-1, 1] / s0, "o", color=col, ms=5)
ax.plot(1, 1, "ko", ms=5)
ax.text(0.97, 0.9965, "start", ha="left", fontsize=10.5)
ax.invert_xaxis()
ax.set_xlabel(r"roughness / start   $\rightarrow$ smoother")
ax.set_ylabel("size / start")
# direct labels instead of a legend (k = degree of homogeneity under scaling, see the insight box in Section 31.3)
ax.text(0.62, 1.0035, r"normal consistency ($k = 0$)", color=C["accent"], fontsize=10.5, ha="left", va="bottom")
ax.text(0.42, 0.961, "edge length,\ntarget 0 ($k = 2$)", color=C["third"], fontsize=10.5, ha="center", va="top")
ax.text(tl[-1, 0] / r0 + 0.04, tl[-1, 1] / s0 - 0.001, "Laplacian,\nuniform ($k = 1$)", color=C["tangent"],
        fontsize=10.5, ha="right", va="bottom")
ax.set_ylim(0.928, 1.013)
ax.grid(True, lw=0.3, alpha=0.5)
ax.set_title("(a) shrinkage against smoothing", fontsize=11.5)

cam = dm.Ortho(elev=15, azim=-60)
base = dm.hex_to_rgb(C["surface"])
for k, (name, fn, col) in enumerate(LOSSES):
    X = results[name][1]
    off = np.array([2.35 * k, 0, 0])
    Xs = X - X.mean(0)
    P = cam(Xs)
    shift = np.array([2.35 * k, 0.0])
    order = np.argsort(cam.depth(Xs[F].mean(1)))
    n = dm.face_cross(Xs, F)
    order = order[(n[order] @ cam.view) > 0]
    cols = dm.shade(Xs, F, dm.hex_to_rgb(col) * 0.5 + 0.5, light=(0.3, -0.5, 0.9), ambient=0.5)
    from matplotlib.collections import PolyCollection
    pc = PolyCollection(P[F[order]] + shift, facecolors=cols[order], edgecolors="none")
    pc.set_rasterized(True)
    axb.add_collection(pc)
    axb.add_patch(Circle(shift, 1.0, fill=False, ec=C["main"], lw=0.9, ls=(0, (4, 3)), zorder=5))
    axb.text(shift[0], -1.35, f"size {results[name][0][-1, 1] / s0:.3f}", ha="center", fontsize=10.5, color=col)
axb.set_xlim(-1.2, 2.35 * 2 + 1.2)
axb.set_ylim(-1.55, 1.2)
axb.set_aspect("equal")
axb.set_axis_off()
axb.set_title("(b) end states (dashed: radius 1)", fontsize=11.5)

fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)

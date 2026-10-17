"""Animation 31.3.4: the three PyTorch3D mesh regularizers, each minimized on its own (Section 31.3, Figure 31.3.5,
the "What geometry explains" box on scale homogeneity).

The setting of Figure 31.3.5 (same code): unit icosphere of level 3 (642 vertices), every vertex scaled by
1 + 0.03 n, n ~ N(0, 1) (seed 0). Each loss alone is minimized by normalized gradient descent
x <- x - 0.0005 g / max|g|:
  Laplacian (uniform):    mean_i |(L_u x)_i|                     (homogeneous of degree k = 1)
  edge length (target 0): mean over edges of |x_i - x_j|^2        (k = 2)
  normal consistency:     mean over edge-sharing face pairs of 1 - cos(n_0, n_1)   (k = 0)
until the roughness (std / mean of the distances to the centroid) is 25% of the start, or 6000 steps.
Normal consistency stops after 218 steps with the size unchanged (1.000), edge length after 203 steps at 0.965;
the uniform Laplacian loss never gets there: its roughness stalls near 0.56 and the mesh keeps shrinking (0.936 after
6000 steps). Left: size against roughness, both relative to the start (as Figure 31.3.5(a)); right: the three meshes
at the same scale with the dashed unit circle. Steps 0-250 are shown at one speed, 250-6000 fast-forwarded.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-3-4-loss-descent.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (Circle, DashedVMobject, VGroup)  # noqa: E402

# ---------------------------------------------------------------- data and losses (as in fig-31-3-5-losses.py)
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


for fn in (lap_loss, edge_loss, nc_loss):  # finite-difference check of the gradients
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
LOSSES = [("Laplacian (uniform)", "Laplacian", lap_loss, "tangent", 1),
          ("edge length (target 0)", "edge length", edge_loss, "third", 2),
          ("normal consistency", "normal consistency", nc_loss, "accent", 0)]
# steps shown: every 5th step up to 250, then geometrically to 6000
SHOW = sorted(set(list(range(0, 251, 5)) + [int(round(v)) for v in np.geomspace(250, 6000, 36)]))
RES = {}
for name, short, fn, col, k in LOSSES:
    X = V0.copy()
    traj, snaps = [stats(X)], {0: X.copy()}
    for it in range(1, 6001):
        _, g = fn(X)
        X = X - 0.0005 * g / np.abs(g).max()
        traj.append(stats(X))
        if it in SHOW:
            snaps[it] = X.copy()
        if traj[-1][0] < 0.25 * r0:
            snaps[it] = X.copy()
            break
    RES[name] = (np.array(traj), snaps)
tl, te, tn = (RES[n][0] for n, *_ in LOSSES)
assert len(tn) - 1 == 218 and tn[-1, 0] < 0.25 * r0 and abs(tn[-1, 1] / s0 - 1) < 0.002     # same size
assert len(te) - 1 == 203 and te[-1, 0] < 0.25 * r0 and te[-1, 1] / s0 < 0.97                # shrunk
assert len(tl) - 1 == 6000 and tl[:, 0].min() > 0.5 * r0 and tl[-1, 1] / s0 < 0.95            # stalls, shrinks
assert abs(tl[:, 0].min() / r0 - 0.56) < 0.01 and abs(tl[-1, 1] / s0 - 0.936) < 0.001


def state(name, it):
    """(roughness/start, size/start, mesh) of a run at step it (the run stays at its last state after stopping)."""
    traj, snaps = RES[name]
    j = min(it, len(traj) - 1)
    keys = [s for s in snaps if s <= j]
    return traj[:j + 1] / np.array([r0, s0]), snaps[max(keys)]


KOFF = {0: np.array([0.0, 0.32, 0]), 1: np.array([0.7, 0.0, 0]), 2: np.array([0.0, -0.38, 0])}


class Anim(hc.Scene2D):
    def construct(self):
        # layout: every label stays above the bottom 12% of the frame (paused control bar, GUIDELINES §12.8)
        pl = hc.Plot((1.0, 0.25), (0.928, 1.013), ((-5.75, -1.6), (-1.05, 2.4)))
        ax = pl.axes(xtick=(1.0, 0.75, 0.5, 0.25), ytick=(0.94, 0.97, 1.0), size=30,
                     xlabels=["1", "0.75", "0.5", "0.25"])
        self.add(ax, hc.text("roughness / start  → smoother", size=30).move_to([-3.4, -2.55, 0]),
                 hc.text("size / start", size=30).move_to(pl(1.0, 1.013) + np.array([0.85, 0.35, 0])))
        self.add(hc.dot(pl(1.0, 1.0), r=0.07))
        pos = {"Laplacian (uniform)": (1.1, 1.85), "edge length (target 0)": (4.8, 1.85),
               "normal consistency": (2.95, -1.5)}
        SC = 1.0
        views = {}
        for name, short, fn, col, k in LOSSES:
            cam = hc.Cam(elev=15, azim=-60, scale=SC, origin=pos[name])
            base = hc.rgb(hc.HEX[col]) * 0.5 + 0.5
            mv = hc.MeshView(V0 - V0.mean(0), F, cam, color=base, edge_color="fill", edge_width=0.5, cull=True,
                             sort=True, ambient=0.45)   # the noisy sphere is not convex: cull and sort
            views[name] = mv
            ring = DashedVMobject(Circle(radius=SC, color=hc.BLACK, stroke_width=2).move_to(cam([0, 0, 0])),
                                  num_dashes=48)
            self.add(mv, ring)
            self.add(hc.text(short, size=30, color=hc.C[col]).move_to(
                [pos[name][0], pos[name][1] + SC + 0.3, 0]))
        info = VGroup()
        curves = VGroup()
        self.add(curves, info)

        def show(it, fast=False):
            curves.submobjects, info.submobjects = [], []
            for name, short, fn, col, k in LOSSES:
                tr, X = state(name, it)
                views[name].set_mesh(X - X.mean(0))
                end = pl(tr[-1, 0], tr[-1, 1])
                curves.add(pl.curve(tr[:, 0], tr[:, 1], color=hc.C[col], width=5),
                           hc.dot(end, color=hc.C[col], r=0.08))
                # k labels once the curves have separated; the Laplacian curve runs on the k = 0 line until it
                # bends down, so its label waits until then (it would sit on the orange line)
                if it >= 40 and (k != 1 or tr[-1, 1] < 0.975):
                    curves.add(hc.text(f"k = {k}", size=30, color=hc.C[col]).move_to(end + KOFF[k]))
                p = pos[name]
                info.add(hc.text(f"size {tr[-1, 1]:.3f}", size=30, color=hc.C[col]).move_to([p[0], p[1] - SC - 0.28, 0]))
            info.add(hc.text(f"step {it}" + ("  (fast-forward)" if fast else ""), size=30).move_to([-3.4, 3.45, 0]))
        show(0)
        self.wait(1.5)
        for it in SHOW[1:]:
            fast = it > 250
            show(it, fast)
            self.wait(0.2 if not fast else 0.22)
        msg = hc.math(r"$\frac{d}{dt}\,\Sigma_i\,|x_i - c|^2 = -2k\,\mathcal{L}$", size=36).move_to([-3.4, 3.4, 0])
        step_label = info.submobjects[-1]
        info.submobjects = info.submobjects[:-1]
        self.add(step_label)
        self.play(*hc.swap(step_label, msg), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.0)

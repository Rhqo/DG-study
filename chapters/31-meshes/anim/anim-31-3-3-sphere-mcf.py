"""Animation 31.3.3: mean curvature flow of a sphere mesh against r(t)^2 = r_0^2 - 4t (Section 31.3, (31.3.7),
Example 31.3.6, Exercise 0.4.4).

The unit icosphere of level 2 (162 vertices, 320 faces) is moved by the implicit cotan flow of Section 31.3,
    (M - tau L) x_new = M x,   tau = 0.0005,   M = mixed Voronoi mass, L and M recomputed every step,
i.e. dx/dt = M^{-1} L x = 2 H N (31.3.7). On a sphere with outward N, H = -1/r, so every vertex moves inward with
speed 2/r and d(r^2)/dt = -4: r(t)^2 = r_0^2 - 4t, extinction at t = r_0^2/4 = 0.25. Left: the mesh at a fixed zoom
(dashed circle: the start, radius 1). Right: r^2 against t; black dashed line = 1 - 4t, orange = the mesh (mean
distance of the vertices from their centroid). Shown: every 8th step, until r < 0.3 (t = 0.23).
Self-check: |r^2 - (1 - 4t)| < 0.005 along the whole run; the mesh stays round (std/mean of the radii < 1e-4).

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-3-3-sphere-mcf.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (Circle, DashedVMobject, FadeIn, VGroup)  # noqa: E402

V0, F = dm.icosphere(2)
TAU = 0.0005
EVERY = 8
X = V0.copy()
t = 0.0
FRAMES = [(0.0, V0.copy(), 1.0)]
k = 0
while True:
    m = dm.mass_voronoi(X, F)
    L = dm.cotan_laplacian(X, F)
    X = np.linalg.solve(np.diag(m) - TAU * L, m[:, None] * X)
    t += TAU
    k += 1
    c = X.mean(0)
    rad = np.linalg.norm(X - c, axis=1)
    assert rad.std() / rad.mean() < 1e-4                      # stays a round sphere mesh
    assert abs(rad.mean() ** 2 - (1 - 4 * t)) < 0.005         # r^2 = r_0^2 - 4t
    if k % EVERY == 0:
        FRAMES.append((t, X - c, rad.mean()))
    if rad.mean() < 0.3:
        break
T_END = t
assert 0.22 < T_END < 0.235 and len(FRAMES) < 70
# the speed at the start: (M^{-1} L x)_i . N_i = -2 / r exactly for the Voronoi mass (Example 31.3.6)
m0 = dm.mass_voronoi(V0, F)
Hn = (dm.cotan_laplacian(V0, F) @ V0) / m0[:, None]
assert np.allclose(np.einsum("ij,ij->i", Hn, V0), -2.0, atol=1e-10)


class Anim(hc.Scene2D):
    def construct(self):
        # layout: the equation and the speed stay above the bottom 12% of the frame (paused control bar, §12.8)
        cam = hc.Cam(elev=20, azim=-60, scale=2.3, origin=(-3.7, -0.05))
        mv = hc.MeshView(V0, F, cam, edge_color="#707070", edge_width=0.8, cull=True, sort=False, ambient=0.5)
        start = DashedVMobject(Circle(radius=cam.scale, color=hc.C["aux"], stroke_width=2.5).move_to(cam([0, 0, 0])),
                               num_dashes=70)
        self.add(start, mv)
        title = hc.text("mean curvature flow of a sphere mesh", size=34).move_to([0, 3.5, 0])
        eq = hc.math(r"$\partial_t x = M^{-1}Lx = 2H\mathbf{N},\quad H = -1/r$", size=31).move_to([-3.45, -2.75, 0])
        self.add(title, eq)
        # plot of r^2 against t
        pl = hc.Plot((0, 0.25), (0, 1.0), ((1.4, -1.75), (6.4, 2.4)))
        ax = pl.axes(xtick=(0, 0.1, 0.2), ytick=(0, 0.5, 1.0), size=30)
        tl = hc.math(r"$t$", size=36).move_to(pl(0.25, 0) + np.array([0.0, -0.45, 0]))
        rl = hc.math(r"$r^2$", size=36).move_to(pl(0, 1.0) + np.array([0.15, 0.45, 0]))
        tt = np.array([0, 0.25])
        theory = hc.seg(pl(0, 1.0), pl(0.25, 0.0), color=hc.BLACK, width=3, dashed=True)
        tlab = hc.math(r"$r_0^2 - 4t$", size=34).move_to(pl(0.175, 0.55))
        self.add(ax, tl, rl, theory, tlab)
        info = VGroup()
        curve = VGroup()
        self.add(info, curve)

        def show(j):
            tj, Xj, rj = FRAMES[j]
            mv.set_mesh(Xj)
            info.submobjects = [hc.text(f"t = {tj:.3f}   r = {rj:.3f}", size=32).move_to([-3.7, 2.85, 0]),
                                hc.text(f"inward speed 2/r = {2 / rj:.2f}", size=30, color=hc.C["normal"])
                                .move_to([3.9, -2.75, 0])]
            ts = np.array([f[0] for f in FRAMES[:j + 1]])
            r2 = np.array([f[2] ** 2 for f in FRAMES[:j + 1]])
            curve.submobjects = [pl.curve(ts, r2, color=hc.C["accent"], width=6),
                                 hc.dot(pl(ts[-1], r2[-1]), color=hc.C["accent"], r=0.08)] if j > 0 else []
        show(0)
        self.wait(2.0)
        for j in range(1, len(FRAMES)):
            show(j)
            self.wait(0.2)
        end = VGroup(hc.text("extinction at", size=30), hc.text("t = r₀²/4 = 0.25", size=30)).arrange(
            np.array([0, -1.0, 0]), buff=0.12).move_to(pl(0.09, 0.17))
        self.play(FadeIn(end), run_time=0.8)
        self.wait(3.0)
        del tt


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)

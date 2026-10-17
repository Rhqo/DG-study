"""Animation 31.3.2: uniform smoothing slides vertices along the surface, cotan smoothing does not (Section 31.3,
Proposition 31.3.4(d), Figure 31.3.4).

The data of Figure 31.3.4: a planar 17 x 13 grid mesh on [0, 2] x [0, 1.4], x-spacing growing to the right
(x_k = 2 (k/16)^1.7), alternating diagonals; interior vertices get z-noise N(0, 0.04^2) (seed 3), boundary fixed.
40 implicit steps of
  uniform: (I - 0.5 L_u) x_new = x,   cotan: (M - 0.005 L) x_new = M x   (L, M recomputed at every step).
Top: the two meshes seen from above, one update per step (steps 1-10, then every second step). Face colour = mean height z of its vertices (RdBu_r,
white = 0, so the speckle is the noise), grey = the initial grid, arrows = horizontal displacement of each vertex
from its start, magnified 10 times. Bottom: mean horizontal drift against the step. The noise disappears in both;
uniform keeps pushing vertices from the dense side to the sparse side (0.048 after 40 steps), cotan stops at
0.0063 (the tilt of the initial noise) because it does not move a flat mesh at all (linear precision).

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-3-2-uniform-vs-cotan.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, VGroup)  # noqa: E402

# ---------------------------------------------------------------- data (identical to fig-31-3-4-smoothing.py)
nx, ny = 17, 13
xs = (np.linspace(0, 1, nx) ** 1.7) * 2.0
ys = np.linspace(0, 1.4, ny)
V, F = dm.grid_mesh(xs, ys, pattern="alternate")
bnd = np.array([i in (0, nx - 1) or j in (0, ny - 1) for i in range(nx) for j in range(ny)])
I = ~bnd
rng = np.random.default_rng(3)
V0 = V.copy()
V0[I, 2] += 0.04 * rng.normal(size=I.sum())
Lu = dm.uniform_laplacian(F)


def step(X, kind, tau):
    if kind == "uniform":
        A, rhs = np.eye(len(X)) - tau * Lu, X
    else:
        m = dm.mass_barycentric(X, F)
        A, rhs = np.diag(m) - tau * dm.cotan_laplacian(X, F), m[:, None] * X
    Xn = X.copy()
    Xn[I] = np.linalg.solve(A[np.ix_(I, I)], rhs[I] - A[np.ix_(I, ~I)] @ X[~I])
    return Xn


NSTEP = 40
RUNS = {}
for kind, tau in (("uniform", 0.5), ("cotan", 0.005)):
    X = V0.copy()
    traj = [X]
    for s in range(NSTEP):
        X = step(X, kind, tau)
        traj.append(X)
    RUNS[kind] = np.array(traj)
DRIFT = {k: np.array([np.linalg.norm((T - V0)[I, :2], axis=1).mean() for T in R]) for k, R in RUNS.items()}
NOISE = {k: np.array([np.sqrt((T[I, 2] ** 2).mean()) for T in R]) for k, R in RUNS.items()}
assert DRIFT["uniform"][-1] > 0.04 and DRIFT["cotan"][-1] < 0.007                       # Figure 31.3.4(c)
assert abs(DRIFT["cotan"][-1] - DRIFT["cotan"][11]) < 1e-4                              # cotan drift saturates
assert NOISE["uniform"][-1] < 0.25 * NOISE["uniform"][0] and NOISE["cotan"][-1] < 0.25 * NOISE["cotan"][0]
Xc = V.copy()                                                                           # linear precision
for s in range(5):
    Xc = step(Xc, "cotan", 0.005)
assert np.abs(Xc - V).max() < 1e-12
ZLIM = 0.1
MAG = 10.0


class Anim(hc.Scene2D):
    def construct(self):
        # layout: every label stays above the bottom 12% of the frame (paused control bar, GUIDELINES §12.8)
        SCM = 2.5
        cams = {"uniform": hc.Cam(elev=90, azim=-90, scale=SCM, origin=(-6.35, -0.8)),
                "cotan": hc.Cam(elev=90, azim=-90, scale=SCM, origin=(0.9, -0.8))}
        names = {"uniform": ("(a) uniform", hc.C["tangent"]), "cotan": ("(b) cotan", hc.C["accent"])}
        views, grids = {}, VGroup()
        E = dm.edges(F)
        for kind, cam in cams.items():
            for a, b in E:
                grids.add(hc.seg(cam(V0[a] * [1, 1, 0]), cam(V0[b] * [1, 1, 0]), color="#BBBBBB", width=1.0))
            mv = hc.MeshView(V0 * [1, 1, 0], F, cam, colors=self.zcol(V0), edge_color="#666666", edge_width=0.8,
                             sort=False, shade=False, opacity=0.9)
            views[kind] = mv
            t, c = names[kind]
            self.add(hc.text(t, size=34, color=c).move_to([cam.origin[0] + SCM, 3.5, 0]))
        self.add(grids, *views.values())
        # plot of the drift
        pl = hc.Plot((0, NSTEP), (0, 0.05), ((-5.6, -2.5), (0.6, -1.4)))
        ax = pl.axes(xtick=(0, 10, 20, 30, 40), ytick=(0, 0.05), size=30, ylabels=["0", "0.05"])
        self.add(ax, hc.text("step", size=30).move_to(pl(NSTEP, 0) + np.array([0.75, 0.0, 0])),
                 hc.text("mean drift", size=30).move_to(pl(0, 0.05) + np.array([1.25, 0.05, 0])))
        info = VGroup()
        arrows = VGroup()
        curves = VGroup()
        self.add(info, arrows, curves)
        self.wait(1.0)

        def show(k):
            info.submobjects = []
            arrows.submobjects = []
            curves.submobjects = []
            for kind, cam in cams.items():
                X = RUNS[kind][k]
                views[kind].set_mesh(X * [1, 1, 0], colors=self.zcol(X))
                c = names[kind][1]
                for p, q in zip(V0[I], X[I]):
                    d = (q - p)[:2]
                    if np.linalg.norm(d) * MAG > 0.012:
                        arrows.add(hc.arrow(cam(p * [1, 1, 0]), cam(p * [1, 1, 0] + MAG * np.r_[d, 0]), color=c,
                                            width=3.0, tip=0.09).set_z_index(3))
                info.add(hc.text(f"drift {DRIFT[kind][k]:.4f}   noise {NOISE[kind][k]:.3f}", size=30, color=c)
                         .move_to([cam.origin[0] + SCM, 3.02, 0]))
                curves.add(pl.curve(np.arange(k + 1), DRIFT[kind][:k + 1], color=c, width=4))
            info.add(hc.text(f"step {k}", size=34).move_to([4.3, -2.6, 0]))
        show(0)
        self.wait(1.5)
        for k in list(range(1, 11)) + list(range(12, NSTEP + 1, 2)):
            show(k)
            self.wait(0.7 if k <= 10 else 0.45)
        note = VGroup(hc.text("uniform: slides sideways", size=30, color=hc.C["tangent"]),
                      hc.text("cotan: removes only noise", size=30, color=hc.C["accent"])).arrange(
            np.array([0, -1.0, 0]), buff=0.18).move_to([4.15, -1.65, 0])
        self.play(FadeIn(note), run_time=0.8)
        self.wait(3.0)

    @staticmethod
    def zcol(X):
        return hc.cmap_rgb(X[F, 2].mean(1), -ZLIM, ZLIM, center=0.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=18.5)

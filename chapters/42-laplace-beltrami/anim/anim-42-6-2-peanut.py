"""Animation 42.6.2: recomputing L every step degenerates a peanut mesh; keeping the first L rounds it
(Section 42.6, Example 42.6.9, (42.6.3), (42.6.4), Figure 42.6.4, [Kazhdan12]).

Data (as Example 42.6.9): icosphere(3) mapped by (x, y, z) -> (x s(z), y s(z), 1.5 z), s(z) = 1 - 0.5 exp(-(z/0.4)^2)
(1280 faces). Semi-implicit steps with h = 0.02 and the mixed Voronoi M of the current mesh,
  left:  (M^k - h L^k) x^{k+1} = M^k x^k   (L recomputed every step, (42.6.3)),
  right: (M^k - h L^0) x^{k+1} = M^k x^k   (L of the first mesh kept, (42.6.4)),
after every step the centroid is kept and the mesh is scaled back to the first area (only the shape is compared).
Video: steps 0 -> 40 for both in 8 s, then the left stops (its smallest angle is 0 degrees at step 40; at step 54 the
cotan weights are no longer finite) and the right goes on to step 120 in 6 s; frames between two steps interpolate
linearly. Colour: the smallest angle of each face, viridis on [0, 40] degrees (dark = sliver), Lambert shading; view
elevation 10, azimuth -60 (the long axis z is vertical), the same scale in both panels.
Readouts: the step k, the smallest angle of the mesh, the length of the mesh along z.
Self-check (Example 42.6.9, Figure 42.6.4): left: z length 3.0, 3.86, 5.1 at steps 0, 20, 35, smallest angle 19.1,
1.9, 0.0 degrees at steps 0, 35, 40, not finite at step 54; right: smallest angle never below 10.6 degrees, z length
3.0, 2.88, 2.32 at steps 0, 40, 120, std/mean of the radii 0.054 at step 120; every step keeps the first area.

Render: bash tools/render_anims.sh chapters/42-laplace-beltrami/anim/anim-42-6-2-peanut.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _helpers as hp  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
_V, F = dm.icosphere(3)
_s = 1 - 0.5 * np.exp(-(_V[:, 2] / 0.4) ** 2)
PEANUT = np.c_[_V[:, 0] * _s, _V[:, 1] * _s, 1.5 * _V[:, 2]]
A0 = dm.face_areas(PEANUT, F).sum()
L0 = dm.cotan_laplacian(PEANUT, F)


def flow(fixed, steps, h=0.02):
    X = [PEANUT.copy()]
    for k in range(1, steps + 1):
        L = L0 if fixed else dm.cotan_laplacian(X[-1], F)
        m = dm.mass_voronoi(X[-1], F)
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(m)) and m.min() > 0):
            return np.array(X), k
        Y = np.linalg.solve(np.diag(m) - h * L, m[:, None] * X[-1])
        c = Y.mean(0)
        X.append(c + (Y - c) * np.sqrt(A0 / dm.face_areas(Y, F).sum()))
    return np.array(X), None


def min_angle(X):
    return float(np.degrees(dm.corner_angles(X, F).min()))


XA, broke = flow(False, 60)
XB, _ = flow(True, 120)
assert broke == 54
assert np.allclose([np.ptp(XA[k][:, 2]) for k in (0, 20, 35)], [3.0, 3.86, 5.1], atol=0.02)
assert np.allclose([min_angle(XA[k]) for k in (0, 35, 40)], [19.1, 1.9, 0.0], atol=0.05)
assert abs(min(min_angle(X) for X in XB) - 10.6) < 0.05
assert np.allclose([np.ptp(XB[k][:, 2]) for k in (0, 40, 120)], [3.0, 2.88, 2.32], atol=0.01)
_r = np.linalg.norm(XB[120] - XB[120].mean(0), axis=1)
assert abs(_r.std() / _r.mean() - 0.054) < 5e-4
assert all(abs(dm.face_areas(X, F).sum() - A0) < 1e-9 for X in (XA[35], XB[120]))
KA, KB = 40, 120
XA = XA[:KA + 1]
ANG_A = np.array([min_angle(X) for X in XA])
ANG_B = np.array([min_angle(X) for X in XB])
ZA = np.array([np.ptp(X[:, 2]) for X in XA])
ZB = np.array([np.ptp(X[:, 2]) for X in XB])
assert abs(ZA[KA] - 6.5255) < 5e-4                                    # z length at step 40 (caption)
AMAX = 40.0


def at(X, s):
    s = float(np.clip(s, 0, len(X) - 1))
    j = min(int(np.floor(s)), len(X) - 2)
    w = s - j
    return (1 - w) * X[j] + w * X[j + 1]


def face_min_angle(V):
    return np.degrees(dm.corner_angles(V, F).min(1))


class Anim(hp.Scene2D):
    def construct(self):
        k = ValueTracker(0.0)
        cams = [hp.Cam(elev=10, azim=-60, scale=0.75, origin=(x, -1.05)) for x in (-3.7, 3.7)]
        views = []
        for X, kmax, cam in ((XA, KA, cams[0]), (XB, KB, cams[1])):
            mv = hp.MeshView(PEANUT, F, cam, colors=hp.cmap_rgb(face_min_angle(PEANUT), 0, AMAX, "viridis"),
                             edge_width=0.3, edge_color="#333333", cull=True, sort=True, ambient=0.6)

            def upd(m, X=X, kmax=kmax, cam=cam):
                V = at(X, min(k.get_value(), kmax))
                cam.pivot = V.mean(0)
                m.set_mesh(V, colors=hp.cmap_rgb(face_min_angle(V), 0, AMAX, "viridis"))
            mv.add_updater(upd)
            mv.update()
            views.append(mv)
        heads = VGroup(hp.math(r"recompute $L^k$ every step", 30).move_to([-3.7, 3.45, 0]),
                       hp.math(r"keep the first $L^0$", 30).move_to([3.7, 3.45, 0]))

        def readout(A, Z, kmax, x):
            def f():
                s = min(k.get_value(), kmax)
                return VGroup(hp.text(f"smallest angle {np.interp(s, np.arange(len(A)), A):.1f}°", 30)
                              .move_to([x, 2.9, 0]),
                              hp.text(f"z length {np.interp(s, np.arange(len(Z)), Z):.2f}", 30).move_to([x, 2.4, 0]))
            return always_redraw(f)
        ra, rb = readout(ANG_A, ZA, KA, -3.7), readout(ANG_B, ZB, KB, 3.7)
        kl = always_redraw(lambda: hp.text(f"step {int(round(k.get_value()))}", 30).move_to([0, 1.25, 0]))

        # colour bar (middle)
        cb = VGroup()
        n = 40
        rgbs = hp.cmap_rgb(np.linspace(0, AMAX, n), 0, AMAX, "viridis")
        y0, hh = -2.5, 2.4
        for j in range(n):
            cb.add(Rectangle(width=0.28, height=hh / n + 0.005, stroke_width=0,
                             fill_color=hp.ManimColor(tuple(rgbs[j])), fill_opacity=1)
                   .move_to([0, y0 + (j + 0.5) * hh / n, 0]))
        cb.add(Rectangle(width=0.28, height=hh, stroke_width=1.5, stroke_color=hp.C["main"]).move_to([0, y0 + hh / 2, 0]))
        for v in (0, 20, 40):
            cb.add(hp.text(f"{v}°", 30).next_to([0.14, y0 + hh * v / AMAX, 0], np.array([1.0, 0, 0]), buff=0.12))
        cb.add(hp.text("face angle", 30).move_to([0.3, y0 + hh + 0.4, 0]))

        self.add(*views, heads, ra, rb, kl, cb)
        for m in (ra, rb, kl):
            m.update()
        self.wait(1.5)
        self.play(k.animate.set_value(float(KA)), run_time=8.0, rate_func=linear)
        na = hp.text("slivers: the flow must stop", 30, color=hp.C["normal"]).move_to([-3.7, 1.9, 0])
        self.play(FadeIn(na), run_time=0.5)
        self.wait(1.5)
        self.play(k.animate.set_value(float(KB)), run_time=6.0, rate_func=linear)
        nb = hp.text("round, no sliver", 30, color=hp.C["tangent"]).move_to([3.7, 1.9, 0])
        self.play(FadeIn(nb), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.5)

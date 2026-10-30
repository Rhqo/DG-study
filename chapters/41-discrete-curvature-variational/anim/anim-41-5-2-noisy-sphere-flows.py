"""Animation 41.5.2: mean curvature flow smooths a noisy sphere, Gauss curvature flow sinks its pits (Section 41.5,
Non-example 41.5.10, Remark 41.5.11, Figure 41.5.3).

Data (as Non-example 41.5.10): lib/ddg41.py noisy_sphere(3, 0.05, 5) (icosphere(3), every vertex scaled by
1 + 0.05 N(0, 1), seed 5; 373 of the 642 vertices have a negative angle defect). Explicit steps x <- x + h v(x) with
h = 0.1 h_c (h_c = 2 / lambda_max of the first mesh) until t = 0.033 (51 steps, t = 0.0335),
  left:  v = (L x)_i / A_i     (discrete mean curvature flow),
  right: v = -(K N)_i / A_i    (discrete Gauss curvature flow), A_i the mixed area.
The video runs steps 0 -> 10 (t = 0.0066, the state of Figure 41.5.3(b)) in 6 s, stops 3 s (with the label changes),
then steps 10 -> 51 in 5 s; frames between two steps interpolate linearly.
Colour: the radial deviation r_i / mean(r) - 1 of the vertices (r = distance to the centroid), averaged over the face,
RdBu_r on [-0.15, 0.15] (red outward, blue inward, white 0, clipped), two-sided Lambert shading, painter's algorithm
(the right mesh is no longer convex). View elevation 15, azimuth -60.
Readouts: t and the roughness std(r) / mean(r) (ddg41.roughness).
Self-check (Non-example 41.5.10): roughness 0.0497 at t = 0; at 10 steps (t = 0.0066) 0.0250 and 0.164; at t = 0.033
0.0101 and 0.329; MCF roughness decreases at every step; after 10 GCF steps the 96 pits moved 0.275 inward on average.

Render: bash tools/render_anims.sh chapters/41-discrete-curvature-variational/anim/anim-41-5-2-noisy-sphere-flows.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg41  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
V0, F = ddg41.noisy_sphere(3, 0.05, 5)
D0 = dm.angle_defect(V0, F)
assert len(V0) == 642 and (D0 < 0).sum() == 373
assert abs(ddg41.roughness(V0) - 0.0497) < 5e-5
H_STEP = 0.1 * 2 / ddg41.largest_eigenvalue(V0, F)
STEPS = int(np.ceil(0.033 / H_STEP))
assert STEPS == 51 and abs(10 * H_STEP - 0.0066) < 5e-5


def run(velocity):
    X = []
    ddg41.explicit_flow(V0, F, velocity, H_STEP, STEPS, record=lambda k, V: X.append(V.copy()))
    return np.array(X)


XM, XG = run(ddg41.mcf_velocity), run(ddg41.gcf_velocity)
RM = np.array([ddg41.roughness(V) for V in XM])
RG = np.array([ddg41.roughness(V) for V in XG])
assert np.allclose([RM[10], RM[-1]], [0.0250, 0.0101], atol=5e-5) and np.all(np.diff(RM) < 0)
assert np.allclose([RG[10], RG[-1]], [0.164, 0.329], atol=5e-4)
_H0 = dd.mean_curvature(V0, F)
_pit = (D0 > 0) & (_H0 > 0)
_rad = lambda V: np.linalg.norm(V - V.mean(0), axis=1)  # noqa: E731
assert _pit.sum() == 96 and abs((_rad(XG[10]) - _rad(V0))[_pit].mean() + 0.275) < 5e-4
DLIM = 0.15


def at(X, s):
    k = min(int(np.floor(s)), STEPS - 1)
    w = s - k
    return (1 - w) * X[k] + w * X[k + 1]


def deviation_colors(V):
    r = np.linalg.norm(V - V.mean(0), axis=1)
    dev = r / r.mean() - 1
    return hp.cmap_rgb(dev[F].mean(1), -DLIM, DLIM, "RdBu_r", center=0.0)


class Anim(hp.Scene2D):
    def construct(self):
        tau = ValueTracker(0.0)              # the (fractional) step k
        cams = [hp.Cam(elev=15, azim=-60, scale=1.6, origin=(x, -0.7)) for x in (-3.75, 3.75)]
        views = []
        for X, cam in ((XM, cams[0]), (XG, cams[1])):
            mv = hp.MeshView(V0, F, cam, colors=deviation_colors(V0), edge_width=0.3, edge_color="#555555", cull=False,
                             sort=True, ambient=0.55)

            def upd(m, X=X, cam=cam):
                V = at(X, tau.get_value())
                cam.pivot = V.mean(0)
                m.set_mesh(V, colors=deviation_colors(V))
            mv.add_updater(upd)
            mv.update()
            views.append(mv)

        t1 = hp.math(r"mean curvature flow: $2H\mathbf{N}$", 32).move_to([-3.6, 3.4, 0])
        t2 = hp.math(r"Gauss curvature flow: $-K\mathbf{N}$", 32).move_to([3.6, 3.4, 0])

        def rough(k):
            s = tau.get_value()
            j = min(int(np.floor(s)), STEPS - 1)
            w = s - j
            R = (RM, RG)[k]
            return (1 - w) * R[j] + w * R[j + 1]
        r1 = always_redraw(lambda: hp.text(f"roughness {rough(0):.4f}", 30).move_to([-3.55, 2.8, 0]))
        r2 = always_redraw(lambda: hp.text(f"roughness {rough(1):.4f}", 30).move_to([3.55, 2.8, 0]))
        tl = hp.math(r"$t =$", 32).move_to([-0.45, -2.4, 0])
        tn = always_redraw(lambda: hp.text(f"{tau.get_value() * H_STEP:.4f}", 32)
                           .next_to(tl, np.array([1.0, 0, 0]), buff=0.15))

        # colour bar (between the panels)
        cb = VGroup()
        n = 48
        vals = np.linspace(-DLIM, DLIM, n)
        rgbs = hp.cmap_rgb(vals, -DLIM, DLIM, "RdBu_r", center=0.0)
        y0, h = -1.5, 2.5
        for k in range(n):
            cb.add(Rectangle(width=0.28, height=h / n + 0.005, stroke_width=0,
                             fill_color=hp.ManimColor(tuple(rgbs[k])), fill_opacity=1)
                   .move_to([0, y0 + (k + 0.5) * h / n, 0]))
        cb.add(Rectangle(width=0.28, height=h, stroke_width=1.5, stroke_color=hp.C["main"]).move_to([0, y0 + h / 2, 0]))
        for v, lab in ((-DLIM, "−0.15"), (0, "0"), (DLIM, "0.15")):
            cb.add(hp.text(lab, 30).next_to([0.14, y0 + h * (v + DLIM) / (2 * DLIM), 0], np.array([1.0, 0, 0]),
                                            buff=0.12))
        cb.add(hp.math(r"$r_i/\bar r - 1$", 30).move_to([0.2, y0 + h + 0.45, 0]))

        self.add(*views, t1, t2, r1, r2, tl, tn, cb)
        for m in (r1, r2, tn):
            m.update()
        self.wait(2.0)
        self.play(tau.animate.set_value(10.0), run_time=6.0, rate_func=linear)
        n1 = hp.text("bumps go in, pits come out", 30, color=hp.C["tangent"]).move_to([-3.6, 2.25, 0])
        n2 = hp.text("pits sink (blue)", 30, color=hp.C["normal"]).move_to([3.6, 2.25, 0])
        self.play(FadeIn(n1), FadeIn(n2), run_time=0.5)
        self.wait(2.0)
        e1 = hp.text("smoother at every step", 30, color=hp.C["tangent"]).move_to([-3.6, 2.25, 0])
        e2 = hp.text("the noise grows to mesh size", 30, color=hp.C["normal"]).move_to([3.6, 2.25, 0])
        self.play(*hp.swap(n1, e1), *hp.swap(n2, e2), run_time=0.5)
        self.play(tau.animate.set_value(float(STEPS)), run_time=5.0, rate_func=linear)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.5)

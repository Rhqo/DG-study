"""Animation 42.6.1: the explicit step's limit narrows while the sphere shrinks; the semi-implicit step has none
(Section 42.6, Proposition 42.6.4(d), Theorem 42.6.6, Example 42.6.8, Figure 42.6.3).

Data (as Example 42.6.8): icosphere(3) with every vertex scaled by 1 + 0.05 N(0, 1) (seed 5); L and M (mixed
Voronoi) of the current mesh at every step; h_c = 2 / lambda_max(M^{-1}(-L)) of the first mesh = 0.00656.
  left:  explicit  x' = x + h M^{-1} L x,   h = 0.9 h_c, 16 steps (t = 0.094);
  right: semi-implicit  (M - h L) x' = M x,  h = 0.02 (about 3 h_c), 5 steps (t = 0.1).
The video time is linear in t (12 s for t from 0 to 0.1); frames between two steps interpolate the positions
linearly. Colour: the radial deviation r_i / mean(r) - 1 (r = distance to the centroid), face average, RdBu_r on
[-0.05, 0.05] (colour bar between the meshes and the plots), two-sided Lambert shading; view elevation 15,
azimuth -60.
Right top: h / h_c(t) of the explicit run, h_c(t) = 2 / lambda_max of the current mesh; dashed = 1 (above it the
explicit step is unstable for the current mesh, Proposition 42.6.4(d)). Right bottom: roughness std(r) / mean(r) (log)
of the two runs (orange explicit, blue semi-implicit).
Self-check (Example 42.6.8): h_c = 0.00656; explicit: h / h_c(t) passes 1 at step 2 (t = 0.012) and is 1.5 at the end,
roughness 0.0497 -> minimum 0.0080 (t = 0.059) -> 0.0188; semi-implicit: roughness and area decrease at every step,
ending at 0.0070.

Render: bash tools/render_anims.sh chapters/42-laplace-beltrami/anim/anim-42-6-1-explicit-limit.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _helpers as hp  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import DashedVMobject, FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
_rng = np.random.default_rng(5)
_V, F = dm.icosphere(3)
V0 = _V * (1 + 0.05 * _rng.normal(size=(len(_V), 1)))


def lam_max(V):
    L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
    return float(np.linalg.eigvalsh(-(L / np.sqrt(m)[:, None] / np.sqrt(m)[None, :]))[-1])


def rough(V):
    d = np.linalg.norm(V - V.mean(0), axis=1)
    return float(d.std() / d.mean())


HC = 2 / lam_max(V0)
assert abs(HC - 0.00656) < 5e-6
HE, HI = 0.9 * HC, 0.02
assert abs(HI / HC - 3) < 0.1
XE, XI = [V0.copy()], [V0.copy()]
for k in range(int(0.1 / HE)):
    L, m = dm.cotan_laplacian(XE[-1], F), dm.mass_voronoi(XE[-1], F)
    XE.append(XE[-1] + HE * (L @ XE[-1]) / m[:, None])
for k in range(int(round(0.1 / HI))):
    L, m = dm.cotan_laplacian(XI[-1], F), dm.mass_voronoi(XI[-1], F)
    XI.append(np.linalg.solve(np.diag(m) - HI * L, m[:, None] * XI[-1]))
XE, XI = np.array(XE), np.array(XI)
TE, TI = HE * np.arange(len(XE)), HI * np.arange(len(XI))
RE, RI = np.array([rough(V) for V in XE]), np.array([rough(V) for V in XI])
RATIO = np.array([HE * lam_max(V) / 2 for V in XE])                 # h / h_c(t)
AI = np.array([dm.face_areas(V, F).sum() for V in XI])
assert len(XE) == 17 and abs(TE[-1] - 0.094) < 1e-3 and len(XI) == 6
assert abs(RE.min() - 0.0080) < 5e-5 and abs(RE[-1] - 0.0188) < 5e-5 and abs(TE[np.argmin(RE)] - 0.059) < 1e-3
assert int(np.argmax(RATIO > 1)) == 2 and abs(TE[2] - 0.012) < 5e-4 and abs(RATIO[-1] - 1.5) < 0.02
assert np.all(np.diff(RI) < 0) and np.all(np.diff(AI) <= 1e-12) and abs(RI[-1] - 0.0070) < 5e-5
DLIM = 0.05
T_END = 0.1


def at(X, T, t):
    t = min(t, T[-1])
    j = int(np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2))
    w = float(np.clip((t - T[j]) / (T[j + 1] - T[j]), 0, 1))
    return (1 - w) * X[j] + w * X[j + 1]


def deviation_colors(V):
    r = np.linalg.norm(V - V.mean(0), axis=1)
    return hp.cmap_rgb((r / r.mean() - 1)[F].mean(1), -DLIM, DLIM, "RdBu_r", center=0.0)


class Anim(hp.Scene2D):
    def construct(self):
        tt = ValueTracker(0.0)
        cams = [hp.Cam(elev=15, azim=-60, scale=1.45, origin=(x, 0.25)) for x in (-5.0, -1.75)]
        views = []
        for X, T, cam in ((XE, TE, cams[0]), (XI, TI, cams[1])):
            mv = hp.MeshView(V0, F, cam, colors=deviation_colors(V0), edge_width=0.3, edge_color="#555555",
                             cull=False, sort=True, ambient=0.55)

            def upd(m, X=X, T=T, cam=cam):
                V = at(X, T, tt.get_value())
                cam.pivot = V.mean(0)
                m.set_mesh(V, colors=deviation_colors(V))
            mv.add_updater(upd)
            mv.update()
            views.append(mv)
        heads = VGroup(hp.text("explicit", 32).move_to([-5.0, 3.4, 0]),
                       hp.math(r"$h = 0.9\,h_c$", 32).move_to([-5.0, 2.8, 0]),
                       hp.text("semi-implicit", 32).move_to([-1.75, 3.4, 0]),
                       hp.math(r"$h = 0.02 \approx 3\,h_c$", 32).move_to([-1.75, 2.8, 0]))

        def rlab(R, T, x, color):
            return always_redraw(lambda: hp.text(f"{np.interp(min(tt.get_value(), T[-1]), T, R):.4f}", 30, color=color)
                                 .move_to([x, -1.95, 0]))
        rl = VGroup(hp.text("roughness", 30).move_to([-5.0, -1.45, 0]), hp.text("roughness", 30).move_to([-1.75, -1.45, 0]))
        r1, r2 = rlab(RE, TE, -5.0, hp.C["accent"]), rlab(RI, TI, -1.75, hp.C["tangent"])

        # ---- right top: h / h_c(t)
        PA = hp.Plot(2.35, 0.55, 4.0, 2.0, (0, T_END), (0.8, 1.6))
        axA = PA.axes(xticks=[(0, ""), (0.05, ""), (0.1, "")], yticks=[(1.0, "1"), (1.5, "1.5")])
        ylA = hp.math(r"$h / h_c(t)$", 30).next_to(PA.p(0, 1.6), np.array([0, 1.0, 0]), buff=0.12) \
            .shift(np.array([0.6, 0, 0]))
        one = DashedVMobject(hp.seg(PA.p(0, 1.0), PA.p(T_END, 1.0), color=hp.C["main"], width=2.5), num_dashes=30)
        unst = hp.text("unstable", 30, color=hp.C["normal"]).move_to(PA.p(0.075, 1.12))

        def traceA():
            t = min(tt.get_value(), TE[-1])
            k = int(np.searchsorted(TE, t, side="right"))
            xs, ys = np.r_[TE[:k], t], np.r_[RATIO[:k], np.interp(t, TE, RATIO)]
            return VGroup(PA.curve(xs, ys, color=hp.C["accent"], width=5),
                          hp.dot(PA.p(xs[-1], ys[-1]), color=hp.C["accent"], r=0.065))
        trA = always_redraw(traceA)

        # ---- right bottom: roughness (log)
        PB = hp.Plot(2.35, -2.25, 4.0, 2.2, (0, T_END), (0.005, 0.06), ylog=True)
        axB = PB.axes(xticks=[(0, "0"), (0.05, "0.05"), (0.1, "0.1")], yticks=[(0.01, "0.01"), (0.05, "0.05")])
        ylB = hp.text("roughness (log)", 30).move_to(PB.p(0.06, 0.045))
        xlB = hp.math(r"$t$", 32).next_to(PB.p(T_END, 0.005), np.array([1.0, 0, 0]), buff=0.15) \
            .shift(np.array([0, 0.15, 0]))                             # clear of the tick label 0.1

        # ---- colour bar of the two meshes (between the meshes and the plots)
        cb = VGroup()
        n = 40
        rgbs = hp.cmap_rgb(np.linspace(-DLIM, DLIM, n), -DLIM, DLIM, "RdBu_r", center=0.0)
        cx, y0, hh = 0.3, 0.35, 1.4
        for j in range(n):
            cb.add(Rectangle(width=0.24, height=hh / n + 0.005, stroke_width=0,
                             fill_color=hp.ManimColor(tuple(rgbs[j])), fill_opacity=1)
                   .move_to([cx, y0 + (j + 0.5) * hh / n, 0]))
        cb.add(Rectangle(width=0.24, height=hh, stroke_width=1.5, stroke_color=hp.C["main"])
               .move_to([cx, y0 + hh / 2, 0]))
        for v, lab in ((-DLIM, "−0.05"), (0, "0"), (DLIM, "0.05")):
            cb.add(hp.text(lab, 30).next_to([cx + 0.12, y0 + hh * (v + DLIM) / (2 * DLIM), 0], np.array([1.0, 0, 0]),
                                            buff=0.1))
        cb.add(hp.math(r"$r_i/\bar r - 1$", 30).move_to([cx + 0.2, y0 + hh + 0.38, 0]))

        def traceB():
            t = tt.get_value()
            g = VGroup()
            for T, R, c in ((TE, RE, hp.C["accent"]), (TI, RI, hp.C["tangent"])):
                s = min(t, T[-1])
                k = int(np.searchsorted(T, s, side="right"))
                xs, ys = np.r_[T[:k], s], np.r_[R[:k], np.interp(s, T, R)]
                g.add(PB.curve(xs, ys, color=c, width=5), hp.dot(PB.p(xs[-1], ys[-1]), color=c, r=0.065))
            return g
        trB = always_redraw(traceB)
        tl = hp.math(r"$t =$", 32).move_to([4.55, 3.4, 0])
        tn = always_redraw(lambda: hp.text(f"{tt.get_value():.3f}", 32).next_to(tl, np.array([1.0, 0, 0]), buff=0.15))

        self.add(*views, heads, rl, r1, r2, axA, ylA, one, unst, trA, axB, ylB, xlB, trB, tl, tn, cb)
        for m in (r1, r2, trA, trB, tn):
            m.update()
        self.wait(2.0)
        self.play(tt.animate.set_value(T_END), run_time=12.0, rate_func=linear)
        e1 = hp.text("the zigzag grows back", 30, color=hp.C["accent"]).move_to([-3.4, -2.5, 0])
        self.play(FadeIn(e1), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)

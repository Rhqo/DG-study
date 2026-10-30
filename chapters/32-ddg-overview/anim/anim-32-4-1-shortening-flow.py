"""Animation 32.4.1: curve shortening flow of the flower curve -- concave parts move out, the curve becomes convex,
then a round point, while the enclosed area falls at the constant rate 2 pi (Section 32.4, Definition 32.4.1,
Proposition 32.4.2, Exercise 32.4.2, Figure 32.4.1).

Flower curve r(t) = 1 + 0.3 cos 5t (counterclockwise), the flow computed as in Figure 32.4.1: ddg32.smooth_flow with
400 points equally spaced in arc length (pointwise -grad L / lbar, forward Euler with time step 0.2 min lbar^2).
Left: the curve at time t; black where kappa_s > 0, orange where kappa_s < 0 (sign of the turning angle, as in
Figure 32.2.1). In the first part vermillion arrows show the velocity kappa_s n_s at the five tips (inward) and the
five waists (outward), drawn at about 0.027 x their length (0.06 scene units per unit of |kappa_s|). Right: the enclosed area A(t) (blue dot, trace) on the dashed
line A(0) - 2 pi t, A(0) = pi (1 + 0.3^2/2) = 3.2830; the inset is the curve rescaled to area pi (shape only).
Self-check: |A(t) - (A(0) - 2 pi t)| < 1e-3 on all snapshots; the snapshot curve is convex (all turning angles > 0) from
t ~ 0.088 on and stays convex; L(t) decreases; the last curve (t = 0.48) is round within 2 percent.

Render: ``bash tools/render_anims.sh chapters/32-ddg-overview/anim/anim-32-4-1-shortening-flow.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ddg32  # noqa: E402
import dgdec as dd  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, ValueTracker, VGroup, always_redraw, linear)  # noqa: E402

A0 = np.pi * (1 + 0.3 ** 2 / 2)
T_END = 0.48
K = 140
TS = T_END * (np.arange(K + 1) / K) ** 1.7              # denser at the start, where the shape changes fast
X0 = ddg32.sample("flower", 4000)[0]
assert abs(ddg32.polygon_area(X0) - A0) < 1e-5
CURVES = ddg32.smooth_flow(X0, TS, n=400)
AREAS = np.array([ddg32.polygon_area(X) for X in CURVES])
LENS = np.array([dd.curve_length(X, True) for X in CURVES])
assert np.max(np.abs(AREAS - (A0 - 2 * np.pi * TS))) < 1e-3          # dA/dt = -2 pi (Exercise 32.4.2)
assert np.all(np.diff(LENS) < 0)                                     # Proposition 32.4.2
CONVEX = np.array([np.all(dd.turning_angles(X, True) > 0) for X in CURVES])
K_CONVEX = int(np.argmax(CONVEX))
assert CONVEX[K_CONVEX:].all() and 0.08 < TS[K_CONVEX] < 0.1                     # Figure 32.4.1: convex by t = 0.1
_c, _r = ddg32.fit_circle(CURVES[-1])
assert ddg32.roundness_defect(CURVES[-1]) < 0.02 and np.linalg.norm(_c) < 1e-6    # symmetric: centre stays at 0
TIPS = [0, 80, 160, 240, 320]                  # t = 0, 2pi/5, ... (400 points, equal arc length, start at a tip)
WAISTS = [40, 120, 200, 280, 360]
_kv = dd.curve_vertex_normals(CURVES[0], True) * (dd.DISCRETE_CURVATURES["B"](CURVES[0], True)
                                                   / dd.curve_dual_lengths(CURVES[0], True))[:, None]
assert np.all(np.einsum("ij,ij->i", _kv[TIPS], CURVES[0][TIPS]) < 0)           # tips move inward
assert np.all(np.einsum("ij,ij->i", _kv[WAISTS], CURVES[0][WAISTS]) > 0)       # waists move outward

CEN = np.array([-3.55, 0.0, 0.0])
SC = 2.25
ARROW = 0.06                                     # arrow length per unit of |kappa_s| (scene units)
PLOT = hp.Plot((0.0, 0.53), (0.0, 3.6), ((1.35, -2.25), (6.6, 2.35)))


def P(X):
    return CEN + SC * hp.to3(X)


def colored_curve(X, width=4.5):
    """black where the turning angle is >= 0, orange (thicker) where it is < 0."""
    th = dd.turning_angles(X, True)
    neg = th < 0
    Q = P(X)
    g = VGroup(hp.polyline(Q, width=width, closed=True))
    n = len(X)
    i = 0
    while i < n:                                 # orange runs of negative curvature
        if neg[i] and not neg[i - 1]:
            j = i
            while neg[(j + 1) % n] and j - i < n:
                j += 1
            idx = [(k % n) for k in range(i - 1, j + 2)]
            g.add(hp.polyline(Q[idx], color=hp.C["accent"], width=width + 3.5))
        i += 1
    return g


class Anim(hp.Scene2D):
    def construct(self):
        k = ValueTracker(0.0)
        show_arrows = ValueTracker(1.0)

        def state():
            x = k.get_value()
            j = int(np.clip(np.floor(x), 0, K - 1))
            s = x - j
            return j, s

        def curve_now():
            j, s = state()
            X = (1 - s) * CURVES[j] + s * CURVES[j + 1]
            return X, (1 - s) * TS[j] + s * TS[j + 1]

        def left():
            X, t = curve_now()
            g = VGroup(colored_curve(X))
            op = show_arrows.get_value()
            if op > 0.01:
                kv = dd.curve_vertex_normals(X, True) * (dd.DISCRETE_CURVATURES["B"](X, True)
                                                         / dd.curve_dual_lengths(X, True))[:, None]
                for i in TIPS + WAISTS:
                    p0 = P(X[i])
                    g.add(hp.arrow(p0, p0 + ARROW * hp.to3(kv[i]), color=hp.C["normal"], width=5,
                                   tip=0.16).set_opacity(op))
            return g
        self.add(always_redraw(left), hp.dot(CEN, color=hp.C["aux"], r=0.04))
        self.add(hp.text("curve shortening flow", size=32).move_to([CEN[0], 3.45, 0]))
        self.add(hp.math(r"$\partial_t\gamma = \kappa_s\,\mathbf{n}_s$", size=34).move_to([4.0, 3.45, 0]))

        # ---- right: area against time
        ax = PLOT.axes(xtick=(0, 0.1, 0.2, 0.3, 0.4, 0.5), ytick=(0, 1, 2, 3), size=30)
        tt = np.linspace(0, 0.5225, 2)
        line = hp.dashed(PLOT(tt, A0 - 2 * np.pi * tt)[:, :2], color=hp.C["aux"], width=2.5, closed=False)
        llab = hp.math(r"$A(0) - 2\pi t$", size=30, color=hp.C["aux"]).move_to(PLOT(0.2, 3.27))   # above the line
        xl = hp.text("t", size=30).move_to(PLOT(0.53, 0) + np.array([0.15, -0.6, 0]))
        yl = hp.text("enclosed area A", size=30).move_to(PLOT(0.0, 3.6) + np.array([1.75, 0.45, 0]))
        self.add(ax, line, llab, xl, yl)

        def trace():
            j, s = state()
            X, t = curve_now()
            a = ddg32.polygon_area(X)
            g = VGroup()
            if j >= 1 or s > 0:
                g.add(hp.Plot.curve(PLOT, np.r_[TS[:j + 1], t], np.r_[AREAS[:j + 1], a], color=hp.C["tangent"],
                                    width=4))
            g.add(hp.dot(PLOT(t, a), color=hp.C["tangent"], r=0.09))
            return g
        self.add(always_redraw(trace))
        ro = always_redraw(lambda: VGroup(
            hp.text(f"t = {curve_now()[1]:.3f}", size=30).move_to(PLOT(0.085, 1.05), aligned_edge=np.array([-1, 0, 0])),
            hp.text(f"A = {ddg32.polygon_area(curve_now()[0]):.3f}", size=30)
            .move_to(PLOT(0.085, 0.5), aligned_edge=np.array([-1, 0, 0]))))
        self.add(ro)
        # inset: the same curve rescaled to area pi (shape only)
        ICEN = PLOT(0.43, 2.6)
        IR = 0.62

        def inset():
            X, _ = curve_now()
            c = X.mean(0)
            s = np.sqrt(np.pi / ddg32.polygon_area(X))
            return VGroup(hp.ring(ICEN, IR, color=hp.C["aux"], width=1.5, dashed_=True),
                          hp.polyline(ICEN + IR * hp.to3((X - c) * s), color=hp.C["tangent"], width=3, closed=True))
        ins = always_redraw(inset)
        ilab = hp.text("rescaled", size=30, color=hp.C["tangent"]).move_to(ICEN + np.array([0, -0.98, 0]))
        self.add(ins, ilab)
        self.wait(1.5)

        # ---- the flow: slowly until the curve is convex, then faster
        self.play(k.animate.set_value(K_CONVEX), run_time=8.0, rate_func=linear)
        cv = hp.text("convex", size=32, color=hp.C["third"]).move_to([CEN[0], -2.5, 0])
        self.play(FadeIn(cv), show_arrows.animate.set_value(0.0), run_time=0.8)
        self.wait(1.2)
        self.play(k.animate.set_value(K), run_time=8.0, rate_func=linear)
        end = hp.math(r"$A \to 0$ at $t = A(0)/2\pi = 0.5225$", size=30).move_to(CEN + np.array([0, -1.45, 0]))
        self.play(FadeOut(cv), run_time=0.4)
        self.play(FadeIn(end), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=4.0)

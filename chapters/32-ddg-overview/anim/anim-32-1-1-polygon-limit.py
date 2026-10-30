"""Animation 32.1.1: inscribed regular polygons -- the length converges, the total turning is exact
(Section 32.1, Proposition 32.1.1, Example 32.1.2, Figure 32.1.1).

Regular n-gon P_n inscribed in the circle of radius r = 1 (vertices r(cos 2 pi i/n, sin 2 pi i/n), counterclockwise).
Part 1 (n = 3): the unit tangent t (blue) walks once around the triangle. At every vertex it turns by the exterior
angle 2 pi/3 (orange arc between the extended incoming edge, grey dashed, and the outgoing edge); the dial at the
centre adds the turns up: 2 pi/3, 4 pi/3, 2 pi.
Part 2: n = 3, 4, ..., 96. Right: L_n / 2 pi r (black dots) creeps up to 1 but never reaches it (0.827, 0.955, 0.989
at n = 3, 6, 12), while the total exterior angle / 2 pi (orange squares) is exactly 1 for every n.
Self-check: perimeter of the vertex list = 2 n r sin(pi/n); dgdec.turning_angles sums to 2 pi (1e-12) for every n;
0 < 2 pi r - L_n <= pi^3 r / (3 n^2) (32.1.1); (2 pi - L_n) n^2 -> pi^3/3 (Example 32.1.2).

Render: ``bash tools/render_anims.sh chapters/32-ddg-overview/anim/anim-32-1-1-polygon-limit.py``
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
from manim import (Create, FadeIn, FadeOut, Square, ValueTracker, VGroup, always_redraw, linear, smooth)  # noqa: E402

R_MATH = 1.0
NS = [3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96]
HOLD = {6: 1.6, 12: 1.6}          # the two polygons of Figure 32.1.1(a)
DATA = {}
for n in NS:
    X = ddg32.regular_ngon(n, R_MATH)
    L = dd.curve_length(X, True)
    th = dd.turning_angles(X, True)
    assert abs(L - 2 * n * R_MATH * np.sin(np.pi / n)) < 1e-12
    assert np.allclose(th, 2 * np.pi / n, atol=1e-12) and abs(th.sum() - 2 * np.pi) < 1e-12   # Prop 32.1.1(b)
    err = 2 * np.pi * R_MATH - L
    assert 0 < err <= np.pi ** 3 * R_MATH / (3 * n ** 2)                                     # (32.1.1)
    DATA[n] = dict(X=X, ratio=L / (2 * np.pi * R_MATH), turn=th.sum() / (2 * np.pi))
assert [round(DATA[n]["ratio"], 3) for n in (3, 6, 12)] == [0.827, 0.955, 0.989]            # Figure 32.1.1(b)
assert abs((2 * np.pi - DATA[96]["ratio"] * 2 * np.pi) * 96 ** 2 - np.pi ** 3 / 3) < 0.01    # Example 32.1.2

# layout (scene units)
CEN = np.array([-3.7, -0.15, 0.0])
SC = 2.45                                       # scene units per unit of length
assert CEN[1] - SC > -2.75 and CEN[1] + SC < 2.6


def P(x):
    """math point(s) (.., 2) -> scene point(s)."""
    x = np.asarray(x, float)
    return CEN + SC * (np.c_[x, np.zeros(len(x))] if x.ndim == 2 else np.r_[x, 0.0])


PLOT = hp.Plot((2.6, 120.0), (0.80, 1.03), ((1.15, -2.05), (6.65, 2.05)), logx=True)


DIAL = np.array([3.9, 0.15, 0.0])          # part 1: the direction dial (the plot takes its place in part 2)
DR = 1.25


class Anim(hp.Scene2D):
    def construct(self):
        circle = hp.ring(CEN, SC * R_MATH, color=hp.C["aux"], width=2.5, dashed_=True)
        self.add(circle, hp.dot(CEN, color=hp.C["aux"], r=0.035))
        ntrack = ValueTracker(3)

        def poly():
            n = int(round(ntrack.get_value()))
            X = P(DATA[n]["X"])
            return VGroup(hp.polyline(X, width=3.5, closed=True),
                          *[hp.dot(p, r=0.055 if n <= 24 else 0.035) for p in X])
        pg = always_redraw(poly)
        title = always_redraw(lambda: hp.text(f"regular n-gon, n = {int(round(ntrack.get_value()))}",
                                              size=32).move_to([CEN[0], 3.45, 0]))
        self.add(pg, title)

        # ---- part 1: walk once around the triangle; the dial on the right adds up the turns
        X3 = DATA[3]["X"]
        T3 = dd.curve_tangents(X3, True)
        ALEN = 0.85                                  # tangent arrow length (scene units)
        dial = hp.ring(DIAL, DR, color=hp.C["aux"], width=2)
        dtitle = hp.text("direction of t", size=32, color=hp.C["tangent"]).move_to(DIAL + np.array([0, 1.85, 0]))
        phi0 = np.arctan2(T3[0, 1], T3[0, 0])
        turned = ValueTracker(0.0)
        pointer = always_redraw(lambda: hp.arrow(DIAL, DIAL + 0.95 * DR * np.array(
            [np.cos(phi0 + turned.get_value()), np.sin(phi0 + turned.get_value()), 0]), color=hp.C["tangent"],
            width=6, tip=0.24))

        def turned_arc():
            a = turned.get_value()
            if a < 1e-3:
                return VGroup()
            s = phi0 + np.linspace(0, a, max(2, int(60 * a / np.pi)))
            pts = DIAL + DR * np.stack([np.cos(s), np.sin(s), 0 * s], 1)
            return hp.polyline(pts, color=hp.C["accent"], width=9)
        tarc = always_redraw(turned_arc)
        tlabel = hp.text("turned 0", size=32).move_to(DIAL + np.array([0, -1.85, 0]))
        self.wait(1.0)
        self.play(Create(dial), FadeIn(dtitle), FadeIn(pointer), FadeIn(tarc), FadeIn(tlabel), run_time=0.8)

        u = ValueTracker(0.0)                        # position along the current edge, 0..1
        edge = [0]
        spin = ValueTracker(0.0)                     # rotation at the current vertex, 0..1

        def tangent_arrow():
            k = edge[0]
            a, b = X3[k], X3[(k + 1) % 3]
            if spin.get_value() > 0:                 # at vertex k+1: rotate from t_k to t_{k+1}
                base = b
                ang = np.arctan2(T3[k, 1], T3[k, 0]) + spin.get_value() * 2 * np.pi / 3
                d = np.array([np.cos(ang), np.sin(ang)])
            else:
                base = a + u.get_value() * (b - a) * 0.999
                d = T3[k]
            p0 = P(base)
            return hp.arrow(p0, p0 + ALEN * np.r_[d, 0], color=hp.C["tangent"], width=6, tip=0.22)
        tang = always_redraw(tangent_arrow)
        self.play(FadeIn(tang), run_time=0.5)
        labels = ["turned 2π/3", "turned 4π/3", "turned 2π"]
        wedges = VGroup()
        self.add(wedges)
        for k in range(3):
            edge[0] = k
            u.set_value(0.0)
            spin.set_value(0.0)
            self.play(u.animate.set_value(1.0), run_time=1.3, rate_func=linear)
            # exterior angle at vertex k+1: grey dashed extension of edge k, orange arc of radius 0.5
            v = X3[(k + 1) % 3]
            ext = hp.seg(P(v), P(v) + 1.0 * np.r_[T3[k], 0], color=hp.C["aux"], width=2.5, dashed=True)
            a0 = np.arctan2(T3[k, 1], T3[k, 0])
            s = a0 + np.linspace(0, 2 * np.pi / 3, 40)
            arc = hp.polyline(P(v) + 0.5 * np.stack([np.cos(s), np.sin(s), 0 * s], 1), color=hp.C["accent"], width=5)
            self.play(FadeIn(ext), run_time=0.3)
            spin.set_value(1e-4)
            new = hp.text(labels[k], size=32).move_to(tlabel)
            self.play(spin.animate.set_value(1.0), turned.animate.set_value((k + 1) * 2 * np.pi / 3), Create(arc),
                      *hp.swap(tlabel, new), run_time=1.1)
            tlabel = new
            wedges.add(ext, arc)
        self.wait(1.5)

        # ---- part 2: n = 3 .. 96, with the plot on the right
        self.play(*[FadeOut(m) for m in (tang, *wedges, tlabel, dial, dtitle, pointer, tarc)], run_time=0.7)
        self.remove(wedges)
        ax = PLOT.axes(xtick=(3, 6, 12, 24, 48, 96), ytick=(0.8, 0.9, 1.0), size=30, ylabels=["0.8", "0.9", "1"])
        one = hp.dashed([PLOT(2.6, 1.0)[:2], PLOT(120, 1.0)[:2]], color=hp.C["aux"], width=2, closed=False)
        xl = hp.text("n", size=30).move_to(PLOT(120, 0.80) + np.array([0.0, -0.75, 0]))
        leg1 = hp.math(r"$L_n\,/\,2\pi r$", size=34).move_to([2.6, 2.95, 0])
        leg2 = hp.math(r"$\Sigma\,\theta_i\,/\,2\pi$", size=34, color=hp.C["accent"]).move_to([5.45, 2.95, 0])
        ld1 = hp.dot(leg1.get_left() + np.array([-0.3, 0, 0]), r=0.08)
        ld2 = Square(side_length=0.17, fill_color=hp.C["accent"], fill_opacity=1, stroke_width=0) \
            .move_to(leg2.get_left() + np.array([-0.3, 0, 0]))
        self.play(FadeIn(ax), FadeIn(one), FadeIn(xl), FadeIn(leg1), FadeIn(ld1), FadeIn(leg2), FadeIn(ld2),
                  run_time=0.8)
        pts = VGroup()
        self.add(pts)
        lhs = hp.math(r"$L_n\,/\,2\pi r\ =$", size=32).move_to([3.55, -1.55, 0])
        live = always_redraw(lambda: hp.text(f"{DATA[int(round(ntrack.get_value()))]['ratio']:.5f}", size=30)
                             .next_to(lhs, np.array([1.0, 0, 0]), buff=0.2))
        self.add(lhs, live)
        for n in NS:
            ntrack.set_value(n)
            d = DATA[n]
            pts.add(hp.dot(PLOT(n, d["ratio"]), r=0.08))
            pts.add(Square(side_length=0.17, fill_color=hp.C["accent"], fill_opacity=1, stroke_width=0)
                    .move_to(PLOT(n, d["turn"])))
            for m in (pg, title, live):
                m.update()                           # a static wait() does not run updaters
            self.wait(HOLD.get(n, 0.85))
        self.wait(0.6)
        exact = hp.text("exact for every n", size=30, color=hp.C["accent"]).move_to([4.45, 1.97, 0])
        bound = hp.math(r"$0 < 2\pi r - L_n \leq \pi^3 r\,/\,3n^2$", size=30).move_to([4.4, -0.75, 0])
        self.play(FadeIn(exact), run_time=0.8)
        self.play(FadeIn(bound), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=30.0)

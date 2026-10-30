"""Animation 35.4.1: d alpha is circulation per area -- shrinking a square around p
(Section 35.4, Proposition 35.4.14, (35.4.5), Corollary 35.4.11, Figure 35.4.2).

alpha = r d theta = (-y dx + x dy)/r on R^2 minus 0 (d alpha = dx ^ dy / r, Example 35.2.7(c)), p = (1, 0.5).
Q_eps = p + [0, eps]^2 with its boundary counterclockwise. Part 1 (eps = 0.8): a point runs once around the boundary
and the line integral of alpha accumulates to 0.3885. Part 2: eps shrinks from 0.8 to 0.025 (geometrically); the
right plot follows eps^{-2} times the circulation (blue curve; black dots at eps = 0.8, 0.4, ..., 0.025 as in
Figure 35.4.2(b)), which climbs to the dashed line (d alpha)_p(e1, e2) = 1/|p| = 0.894 with an error proportional to
eps. Blue arrows: alpha^sharp = (-y, x)/r (unit length, drawn at 0.17 x) on a grid.
Self-check: the circulation computed side by side (Gauss-Legendre) equals the integral of 1/r over Q_eps (Stokes,
Corollary 35.4.11) to 1e-12; the ratios at the six eps are 0.607, 0.722, 0.799, 0.844, 0.868, 0.881 (Figure 35.4.2);
they increase toward 0.894, and halving eps divides the error by 1.67, 1.81, ..., 1.97 (-> 2: error O(eps)).

Render: ``bash tools/render_anims.sh chapters/35-exterior-calculus/anim/anim-35-4-1-circulation-density.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, Polygon, ValueTracker, VGroup, always_redraw, linear, smooth)  # noqa: E402

PT = np.array([1.0, 0.5])
LIM = 1 / np.linalg.norm(PT)
GL, GW = np.polynomial.legendre.leggauss(20)


def alpha(p):
    """components of alpha = r d theta at p (a covector = the vector alpha^sharp in Euclidean R^2)."""
    r = np.linalg.norm(p)
    return np.array([-p[1], p[0]]) / r


def side(a, b):
    t = (GL + 1) / 2
    pts = a[None] + t[:, None] * (b - a)[None]
    return float(np.sum(GW / 2 * np.array([alpha(q) @ (b - a) for q in pts])))


def corners(eps):
    return [PT, PT + [eps, 0], PT + [eps, eps], PT + [0, eps]]


def circulation(eps):
    c = corners(eps)
    return sum(side(c[k], c[(k + 1) % 4]) for k in range(4))


def area_integral(eps):
    t = PT[0] + eps * (GL + 1) / 2
    s = PT[1] + eps * (GL + 1) / 2
    X, Y = np.meshgrid(t, s, indexing="ij")
    W = np.outer(GW, GW) * (eps / 2) ** 2
    return float(np.sum(W / np.sqrt(X ** 2 + Y ** 2)))


EPS_FIG = 0.8 / 2.0 ** np.arange(6)
RATIO = np.array([circulation(e) / e ** 2 for e in EPS_FIG])
for e in EPS_FIG:
    assert abs(circulation(e) - area_integral(e)) < 1e-12                       # Corollary 35.4.11
assert [round(x, 3) for x in RATIO] == [0.607, 0.722, 0.799, 0.844, 0.868, 0.881]   # Figure 35.4.2
err = LIM - RATIO
assert np.all(err > 0) and np.all(np.diff(RATIO) > 0)
assert np.all(np.diff(err[:-1] / err[1:]) > 0) and 1.9 < err[-2] / err[-1] < 2.0       # error ratio -> 2: O(eps)
C08 = circulation(0.8)
assert abs(C08 - 0.3885) < 1e-4                                   # 0.38849, shown as 0.3885

O = np.array([-6.35, -2.15, 0.0])            # scene position of the math origin
SC = 2.45


def P(x):
    return O + SC * hp.to3(np.asarray(x, float))


PLOT = hp.Plot((1.0, 0.02), (0.55, 0.95), ((1.2, -2.0), (6.6, 1.45)), logx=True)


def boundary_point(eps, s):
    """point at parameter s in [0, 4) along the counterclockwise boundary."""
    c = corners(eps)
    k = int(np.floor(s)) % 4
    f = s - np.floor(s)
    return c[k] + f * (c[(k + 1) % 4] - c[k])


def partial_circulation(eps, s):
    c = corners(eps)
    tot = 0.0
    k = 0
    while k + 1 <= s:
        tot += side(c[k], c[(k + 1) % 4])
        k += 1
    if s > k:
        tot += side(c[k], boundary_point(eps, s))
    return tot


class Anim(hp.Scene2D):
    def construct(self):
        eps = ValueTracker(0.8)
        run = ValueTracker(0.0)
        field = VGroup()
        for x in np.arange(0.0, 2.21, 0.3):
            for y in np.arange(0.0, 1.81, 0.3):
                p = np.array([x, y])
                if np.linalg.norm(p) < 0.2:
                    continue
                a = alpha(p) * 0.17
                field.add(hp.arrow(P(p - a / 2), P(p + a / 2), color=hp.C["region"], width=3, tip=0.12))
        self.add(field, hp.dot(P([0, 0]), r=0.06), hp.text("0", size=30).move_to(P([0, 0]) + np.array([-0.25, -0.25, 0])))
        self.add(hp.math(r"$\alpha = r\,d\theta$", size=34).move_to([-3.6, 3.45, 0]))

        def square():
            e = eps.get_value()
            c = [P(q) for q in corners(e)]
            g = VGroup(Polygon(*c, fill_color=hp.C["region"], fill_opacity=0.35, stroke_color=hp.BLACK, stroke_width=3))
            if e > 0.15:
                for k in range(4):
                    a, b = c[k], c[(k + 1) % 4]
                    m = (a + b) / 2
                    d = (b - a) / np.linalg.norm(b - a)
                    g.add(hp.arrow(m - 0.12 * d, m + 0.12 * d, color=hp.C["tangent"], width=5, tip=0.2))
            g.add(hp.dot(P(PT), r=0.06))
            return g
        self.add(always_redraw(square))
        self.add(hp.text("p", size=30).move_to(P(PT) + np.array([-0.25, -0.22, 0])))

        # ---- right: the ratio against eps (log axis, eps decreasing to the right)
        ax = PLOT.axes(xtick=(0.8, 0.4, 0.2, 0.1, 0.05, 0.025), ytick=(0.6, 0.7, 0.8, 0.9), size=30,
                       xlabels=["0.8", "0.4", "0.2", "0.1", ".05", ""])
        lim = hp.dashed([PLOT(1.0, LIM)[:2], PLOT(0.02, LIM)[:2]], color=hp.C["aux"], width=2.5, closed=False)
        limlab = hp.math(r"$1/|p| = 0.894$", size=30).move_to(PLOT(0.36, LIM) + np.array([0, 0.38, 0]))
        xl = hp.math(r"$\varepsilon \to 0$", size=30).move_to(PLOT(0.025, 0.55) + np.array([0.05, -0.75, 0]))
        self.add(ax, lim, limlab, xl)
        dots = VGroup()
        self.add(dots)

        def trace():
            e = eps.get_value()
            if e > 0.799:
                return VGroup()
            es = np.geomspace(0.8, e, 40)
            ys = np.array([circulation(x) / x ** 2 for x in es])
            return VGroup(PLOT.curve(es, ys, color=hp.C["tangent"], width=4),
                          hp.dot(PLOT(e, ys[-1]), color=hp.C["tangent"], r=0.08))
        self.add(always_redraw(trace))

        def readout():
            e = eps.get_value()
            s = run.get_value()
            circ = partial_circulation(e, s) if s < 4 else circulation(e)
            g = VGroup(hp.text(f"ε = {e:.3f}", size=30),
                       hp.text(f"circulation = {circ:.4f}", size=30))
            if s >= 4:
                g.add(hp.text(f"circulation / ε² = {circ / e ** 2:.3f}", size=30, color=hp.C["tangent"]))
            for m, y in zip(g, (3.35, 2.8, 2.25)):
                m.move_to([1.2, y, 0], aligned_edge=np.array([-1, 0, 0]))
            return g
        ro = always_redraw(readout)
        self.add(ro)

        def runner():
            s = run.get_value()
            if s <= 0 or s >= 4:
                return VGroup()
            return hp.dot(P(boundary_point(eps.get_value(), s)), color=hp.C["accent"], r=0.1)
        rn = always_redraw(runner)
        self.add(rn)
        self.wait(1.5)
        self.play(run.animate.set_value(3.999), run_time=4.0, rate_func=linear)
        run.set_value(4.0)
        ro.update()                                  # a static wait() does not run updaters
        rn.update()
        self.add(hp.dot(PLOT(0.8, RATIO[0]), r=0.075))
        self.wait(1.5)
        # shrink: eps = 0.8 * 2^(-5 u), u: 0 -> 1, dropping a black dot at each halving
        u = ValueTracker(0.0)
        eps.add_updater(lambda m: m.set_value(0.8 * 2.0 ** (-5 * u.get_value())))
        self.add(eps)
        for k in range(1, 6):
            self.play(u.animate.set_value(k / 5), run_time=2.0, rate_func=linear)
            dots.add(hp.dot(PLOT(EPS_FIG[k], RATIO[k]), r=0.075))
        eps.clear_updaters()
        fin = hp.math(r"$= (d\alpha)_p(e_1, e_2)$", size=30, color=hp.C["tangent"]).next_to(limlab, np.array([1.0, 0, 0]),
                                                                                         buff=0.2)
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.0)

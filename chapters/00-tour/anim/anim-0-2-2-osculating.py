"""Animation 0.2.2: osculating circle이 ellipse를 따라 미끄러진다 (Section 0.2, Figure 0.2.2, Example 0.2.5).

Figure 0.2.2와 같은 ellipse γ(t) = (2 cos t, sin t) (반시계 방향), t: 0 → π.
점 γ(t)에서의 osculating circle(파랑, 반지름 R = 1/κ_s), center of curvature c(t) = γ + n_s/κ_s (주황 점),
γ에서 c까지의 반지름(주황 점선), 지나온 center들의 자취(주황 실선, ellipse의 evolute).
κ_s(t) = 2/(4 sin² t + cos² t)^{3/2}: t = 0에서 2 (R = 1/2), t = π/4에서 R ≈ 1.98, t = π/2에서 1/4 (R = 4, 중심 (0, −3)).
멈추는 곳: t = 0, π/4, π/2, π (Figure 0.2.2의 세 원과 반대쪽 끝).
자기검사: dgsym.signed_curvature의 식이 위와 같다. center가 evolute (1.5 cos³ t, −3 sin³ t)와 같다.
osculating circle은 γ와 2차까지 접한다(γ(t₀ ± h)에서 원까지의 거리가 O(h³)).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, Circle, DashedLine, Dot, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
DOWN_ = np.array([0.0, -1.0, 0.0])

ts = sp.symbols("t", real=True)
ks_sym = dgsym.signed_curvature(sp.Matrix([2 * sp.cos(ts), sp.sin(ts), 0]), ts)
assert sp.simplify(ks_sym - 2 / (4 * sp.sin(ts) ** 2 + sp.cos(ts) ** 2) ** sp.Rational(3, 2)) == 0
ks = sp.lambdify(ts, ks_sym, "numpy")


def gam(t):
    return np.array([2 * np.cos(t), np.sin(t)])


def center(t):
    v = np.array([-2 * np.sin(t), np.cos(t)])
    tv = v / np.linalg.norm(v)
    return gam(t) + np.array([-tv[1], tv[0]]) / ks(t)


assert np.isclose(ks(0.0), 2.0) and np.isclose(ks(np.pi / 2), 0.25)
assert np.isclose(1 / ks(np.pi / 4), 1.976, atol=1e-3)
for t0 in np.linspace(0, np.pi, 13):
    assert np.allclose(center(t0), [1.5 * np.cos(t0) ** 3, -3 * np.sin(t0) ** 3], atol=1e-12)   # evolute
    c0, R0 = center(t0), 1 / ks(t0)
    d = [abs(np.linalg.norm(gam(t0 + h) - c0) - R0) for h in (1e-2, 5e-3)]
    assert d[1] < d[0] / 6, d                                   # 2차 접촉: h를 반으로 → 거리는 1/8 정도

SC = 1.45
O = np.array([-1.2, 2.1, 0.0])           # t = π/2의 center (0, −3)이 poster의 아래 20% 위에 오도록


def scr(p):
    return O + SC * np.array([p[0], p[1], 0.0])


def frame(t):
    g = VGroup()
    R = 1 / float(ks(t))
    c = center(t)
    tt = np.linspace(0, t, max(2, int(200 * t / np.pi) + 2))
    if t > 1e-3:
        g.add(ha.polyline(np.array([scr(center(x)) for x in tt]), C["accent"], 4))
    g.add(Circle(radius=SC * R, color=C["tangent"], stroke_width=5).move_to(scr(c)))
    g.add(DashedLine(scr(gam(t)), scr(c), color=C["accent"], stroke_width=3, dash_length=0.12))
    g.add(Dot(scr(c), radius=0.08, color=C["accent"]))
    g.add(Dot(scr(gam(t)), radius=0.08, color=C["main"]))
    g.add(ha.row(r"$\kappa_s =$", f"{float(ks(t)):.2f}", size=32).move_to([3.45, 3.35, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$R = 1/\kappa_s =$", f"{R:.2f}", size=32, color=C["tangent"])
          .move_to([3.45, 2.65, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$t =$", TLAB.get(round(t, 4), f"{t:.2f}"), size=32, color=C["aux"])
          .move_to([3.45, 1.95, 0], aligned_edge=LEFT))
    return g


TLAB = {0.0: "0", round(np.pi / 4, 4): r"$\pi/4$", round(np.pi / 2, 4): r"$\pi/2$", round(np.pi, 4): r"$\pi$"}


class Anim(da.Scene2D):
    def construct(self):
        t = ValueTracker(0.0)
        tt = np.linspace(0, 2 * np.pi, 500)
        ell = ha.polyline(np.array([scr(gam(x)) for x in tt]), C["main"], 6)
        lab_g = da.mtex(r"$\gamma$", 36).move_to(scr(gam(2.2)) + [-0.35, 0.3, 0])
        lab_e = da.label("evolute", 30, C["accent"]).move_to(scr([1.5 * np.cos(0.95) ** 3, -3 * np.sin(0.95) ** 3])
                                                              + [1.15, 0.0, 0])
        dyn = always_redraw(lambda: frame(t.get_value()))
        self.add(ell, lab_g, dyn)
        self.wait(2.0)
        for t1, run, hold in ((np.pi / 4, 2.5, 1.5), (np.pi / 2, 2.5, 2.0), (np.pi, 4.0, 0.0)):
            self.play(t.animate.set_value(t1), run_time=run, rate_func=smooth)
            if t1 == np.pi / 2:
                self.add(lab_e)
            if hold > 0:
                self.wait(hold)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.5)

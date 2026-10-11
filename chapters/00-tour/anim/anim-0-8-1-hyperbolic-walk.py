"""Animation 0.8.1: hyperbolic plane 𝕌²에서 metric g = (dx² + dy²)/y²로 재기 (0.8절, Example 0.8.2(c), (0.8.2), Figure 0.8.1,
Exercise 0.8.1).

회색 원은 Figure 0.8.1처럼 각 점에서 |v|_g = 1인 tangent vector의 끝점(실제 크기의 0.15배)이다.
  1부: 점(파랑)이 (−1.6, 2.5)에서 boundary y = 0을 향해 g-속력 1로 내려간다. y(t) = 2.5 e^{−t}이므로 시간 t는 지금까지 걸은
       g-length다. g-length 1마다 남긴 눈금(검은 점)의 간격이 e^{−1}배씩 줄어, boundary에는 끝내 닿지 않는다.
       점의 tangent vector 원(파랑, 0.15배)도 함께 줄어든다.
  2부: A = (−1, √3)에서 B = (1, √3)까지 두 점이 같은 g-속력으로 달린다. geodesic(주황, 원점 중심 반지름 2인 반원의 호,
       길이 ln 3 ≈ 1.099)을 따라간 점이 수평 선분(점선, 길이 2/√3 ≈ 1.155)을 따라간 점보다 먼저 닿는다. 화면의 1초가
       g-length 0.25다. geodesic 쪽이 도착한 순간 1초 멈춘다.
자기검사: y(t) = 2.5e^{−t}의 g-속력 1, 눈금 간격의 비 e^{−1}, 두 길이의 수치 적분(ln 3, 2/√3), 호의 매개화가 g-속력 1.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-1-hyperbolic-walk.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import polyline  # noqa: E402

C = da.C
Y0, X_F = 2.5, -1.6
A, B = np.array([-1.0, np.sqrt(3)]), np.array([1.0, np.sqrt(3)])
L_GEO, L_HOR = np.log(3), 2 / np.sqrt(3)
PHI_A = 2 * np.pi / 3


def fall(t):
    return np.array([X_F, Y0 * np.exp(-t)])


def geo(s):
    """원점 중심 반지름 2인 반원 위를 A에서 g-속력 1로: s(φ) = ln tan(φ_A/2) − ln tan(φ/2)."""
    phi = 2 * np.arctan(np.tan(PHI_A / 2) * np.exp(-s))
    return np.array([2 * np.cos(phi), 2 * np.sin(phi)])


def hor(s):
    return np.array([A[0] + np.sqrt(3) * s, np.sqrt(3)])


def g_length(path, s0, s1, n=20001):
    s = np.linspace(s0, s1, n)
    P = np.array([path(x) for x in s])
    mid = (P[1:] + P[:-1]) / 2
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1) / mid[:, 1]))


# ---------------------------------------------------------------- 자기검사
assert np.isclose(g_length(fall, 0, 3), 3, rtol=1e-6)
assert np.isclose(fall(1)[1] / fall(0)[1], np.exp(-1))
assert np.allclose(geo(0), A) and np.allclose(geo(L_GEO), B) and np.allclose(hor(L_HOR), B)
assert np.isclose(g_length(geo, 0, L_GEO), np.log(3), rtol=1e-6) and np.isclose(g_length(hor, 0, L_HOR), L_HOR, rtol=1e-6)
assert np.isclose(L_GEO, 1.0986, atol=1e-4) and np.isclose(L_HOR, 1.1547, atol=1e-4) and L_GEO < L_HOR

S = 1.45
O = np.array([-3.1, -2.75, 0])            # 아래 90px(control bar)를 비운다(§12.8)
SPEED = 4.0     # 화면 초 / g-length


def P(x):
    x = np.atleast_2d(x)
    return O + S * np.column_stack([x[:, 0], x[:, 1], np.zeros(len(x))])


class Anim(da.Scene2D):
    def construct(self):
        bnd = Line(P([-2.5, 0])[0], P([2.5, 0])[0], stroke_color=C["main"], stroke_width=5)
        self.add(bnd, da.mtex(r"$y = 0$", 32).next_to(P([2.5, 0])[0], RIGHT, buff=0.15))
        circ = VGroup()
        for y0 in (0.3, 0.6, 1.0, 1.6, 2.4):
            for x0 in np.arange(-2.25, 2.26, 0.75):
                circ.add(Circle(radius=0.15 * y0 * S, stroke_color="#A8A8A8", stroke_width=2).move_to(P([x0, y0])[0]))
        self.add(circ)
        title = da.mtex(r"$\mathbb{U}^2:\ \ g = (dx^2 + dy^2)/y^2$", 38).to_edge(UP, buff=0.25)
        self.add(title)
        XT = 1.0
        t = ValueTracker(0.0)

        def faller():
            tt = t.get_value()
            p = fall(tt)
            g = VGroup()
            for k in range(1, int(np.floor(tt)) + 1):
                g.add(Dot(P(fall(k))[0], radius=0.06, color=C["main"]))
            g.add(polyline(P(np.array([fall(0), p])), C["tangent"], 3, 0.6))
            g.add(Circle(radius=0.15 * p[1] * S, stroke_color=C["tangent"], stroke_width=3).move_to(P(p)[0]))
            g.add(Dot(P(p)[0], radius=0.08, color=C["tangent"]))
            return g

        F = always_redraw(faller)
        self.add(F, Dot(P(fall(0))[0], radius=0.06))
        r1 = always_redraw(lambda: VGroup(
            da.label(f"g-length walked: {t.get_value():.2f}", 32, color=C["tangent"]),
            da.label(f"height y = {fall(t.get_value())[1]:.3f}", 32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .move_to([XT, 2.75, 0], aligned_edge=UL))
        self.add(r1)
        self.wait(1.5)
        self.play(t.animate.set_value(4.0), run_time=8.0, rate_func=linear)
        n1 = VGroup(da.label("one dot per unit of length:", 30),
                    da.label("y = 0 is infinitely far", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        n1.move_to([XT, 1.6, 0], aligned_edge=UL)
        self.play(FadeIn(n1), run_time=0.6)
        self.wait(1.5)

        # 2부: 경주
        arcpts = np.array([geo(s) for s in np.linspace(0, L_GEO, 120)])
        g_arc = polyline(P(arcpts), C["accent"], 5)
        g_hor = DashedLine(P(A)[0], P(B)[0], stroke_color=C["main"], stroke_width=3, dash_length=0.1)
        labs = VGroup(Dot(P(A)[0], radius=0.07), Dot(P(B)[0], radius=0.07),
                      da.mtex(r"$A$", 34).next_to(P(A)[0], LEFT, buff=0.1),
                      da.mtex(r"$B$", 34).next_to(P(B)[0], RIGHT, buff=0.1))
        self.play(FadeIn(g_arc), Create(g_hor), FadeIn(labs), run_time=0.8)
        s = ValueTracker(0.0)

        def runners():
            ss = s.get_value()
            return VGroup(Dot(P(geo(min(ss, L_GEO)))[0], radius=0.1, color=C["accent"]),
                          Dot(P(hor(min(ss, L_HOR)))[0], radius=0.1, color=C["main"]))

        R = always_redraw(runners)
        r2 = always_redraw(lambda: VGroup(
            da.label(f"geodesic: {min(s.get_value(), L_GEO):.3f}" + ("  arrived" if s.get_value() >= L_GEO - 1e-9 else ""),
                     32, color=C["accent"]),
            da.label(f"horizontal: {min(s.get_value(), L_HOR):.3f}" + ("  arrived" if s.get_value() >= L_HOR - 1e-9 else ""),
                     32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([XT, -0.1, 0], aligned_edge=UL))
        h2 = da.label("same g-speed, A to B:", 32).move_to([XT, 0.5, 0], aligned_edge=UL)
        self.add(R)
        self.play(FadeIn(r2), FadeIn(h2), run_time=0.5)
        # geodesic 쪽이 도착한 순간 1초 멈춘다: 수평 선분의 점은 아직 B에 닿지 않았다(차이 0.056)
        self.play(s.animate.set_value(L_GEO), run_time=SPEED * L_GEO, rate_func=linear)
        self.wait(1.0)
        self.play(s.animate.set_value(L_HOR), run_time=SPEED * (L_HOR - L_GEO), rate_func=linear)
        n2 = VGroup(da.label("the geodesic bends up,", 30),
                    da.label("where lengths are shorter", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        n2.move_to([XT, -1.4, 0], aligned_edge=UL)
        self.play(FadeIn(n2), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=11.0)

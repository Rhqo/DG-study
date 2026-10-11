"""Animation 0.7.2: dx ∧ dy(v, w)는 v, w가 만드는 평행사변형의 signed area다 (0.7절, Figure 0.7.2).

v = (2, 1/2)(파랑, 첫 번째 인자)는 고정하고, w(초록, 두 번째 인자)를 처음 값 (1/2, 3/2)(Figure 0.7.2)에서 길이를
유지한 채 원점 둘레로 한 바퀴(반시계 방향) 돌린다. 평행사변형은 값이 양수이면 하늘색, 음수이면 주황으로 칠한다.
검은 호 화살표는 v에서 w로 가는 짧은 쪽 회전이다(반시계 = +, 시계 = −).
오른쪽 그래프: 가로축은 w의 각 φ_w, 세로축은 dx ∧ dy(v, w) = v¹w² − v²w¹ = |v||w| sin(φ_w − φ_v).
값은 처음 11/4 = 2.75이고, w가 v와 평행할 때(φ_w = φ_v, φ_v + π) 0, 수직일 때 ±|v||w| = ±3.26이다.
자기검사: det 공식 = |v||w| sin(φ_w − φ_v), 처음 값 11/4, 순서를 바꾸면 부호 반대, w ∥ v에서 0.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-7-2-wedge-sign.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import arrow, polyline  # noqa: E402

C = da.C
V = np.array([2.0, 0.5])
W0 = np.array([0.5, 1.5])
PV = np.arctan2(V[1], V[0])
PW0 = np.arctan2(W0[1], W0[0])
RW = np.linalg.norm(W0)


def wedge(v, w):
    return v[0] * w[1] - v[1] * w[0]


def w_of(phi):
    return RW * np.array([np.cos(phi), np.sin(phi)])


# ---------------------------------------------------------------- 자기검사
assert np.isclose(wedge(V, W0), 11 / 4) and np.isclose(wedge(W0, V), -11 / 4)
for phi in np.linspace(0, 2 * np.pi, 37):
    assert np.isclose(wedge(V, w_of(phi)), np.linalg.norm(V) * RW * np.sin(phi - PV))
assert np.isclose(wedge(V, w_of(PV)), 0) and np.isclose(wedge(V, w_of(PV + np.pi)), 0)
AMAX = np.linalg.norm(V) * RW
assert np.isclose(AMAX, 3.26, atol=5e-3)

SC = 1.1
O = np.array([-3.85, -0.25, 0])            # 아래 90px(control bar)를 비운다(§12.8)
NEG = ManimColor("#F3C77A")      # accent(주황)을 옅게


def P(x):
    return O + SC * np.array([x[0], x[1], 0])


# 그래프
GX0, GX1, GY = 0.55, 5.3, 0.1
GH = 0.68


def G(phi, val):
    return np.array([GX0 + (GX1 - GX0) * (phi - PW0) / (2 * np.pi), GY + GH * val, 0])


class Anim(da.Scene2D):
    def construct(self):
        ph = ValueTracker(PW0)
        axes = VGroup(Arrow(P([-2.2, 0]), P([2.9, 0]), buff=0, stroke_width=2, color=C["aux"], tip_length=0.15),
                      Arrow(P([0, -1.95]), P([0, 2.6]), buff=0, stroke_width=2, color=C["aux"], tip_length=0.15))
        self.add(axes)

        def pic():
            phi = ph.get_value()
            w = w_of(phi)
            val = wedge(V, w)
            g = VGroup(Polygon(P([0, 0]), P(V), P(V + w), P(w), stroke_width=0,
                               fill_color=C["region"] if val >= 0 else NEG, fill_opacity=0.55))
            g.add(arrow(P([0, 0]), P(V), C["tangent"], 7, 0.24), arrow(P([0, 0]), P(w), C["third"], 7, 0.24))
            d = np.mod(phi - PV, 2 * np.pi)
            span = d if d <= np.pi else d - 2 * np.pi
            if abs(span) > 0.05:
                s = PV + np.linspace(0, span, 40)
                r = 0.7
                pts = [P(r * np.array([np.cos(a), np.sin(a)])) for a in s]
                g.add(polyline(pts[:-3], C["main"], 3.5))
                g.add(arrow(pts[-8], pts[-1], C["main"], 3.5, 0.2))
            g.add(da.mtex(r"$v$", 38, color=C["tangent"]).next_to(P(V), RIGHT, buff=0.1))
            g.add(da.mtex(r"$w$", 38, color=C["third"]).next_to(P(w), UP if w[1] >= 0 else DOWN, buff=0.1))
            return g

        self.add(always_redraw(pic))
        title = da.mtex(r"$dx \wedge dy\,(v, w) = v^1 w^2 - v^2 w^1$", 38).to_edge(UP, buff=0.3)
        self.add(title)
        # 그래프 축
        gax = VGroup(Line(G(PW0, 0), G(PW0 + 2 * np.pi, 0) + RIGHT * 0.2, stroke_color=C["aux"], stroke_width=2),
                     Line(G(PW0, -AMAX - 0.4), G(PW0, AMAX + 0.4), stroke_color=C["aux"], stroke_width=2))
        curve = polyline([G(p, wedge(V, w_of(p))) for p in np.linspace(PW0, PW0 + 2 * np.pi, 200)], C["main"], 3)
        zeros = VGroup(*[DashedLine(G(p, -AMAX - 0.2), G(p, AMAX + 0.2), stroke_color=C["aux"], stroke_width=1.5,
                                    dash_length=0.08) for p in (PV + np.pi, PV + 2 * np.pi)])
        glab = VGroup(da.mtex(r"$\varphi_w$", 34).next_to(G(PW0 + 2 * np.pi, 0) + RIGHT * 0.2, RIGHT, buff=0.12),
                      da.label("w ∥ v", 30).next_to(G(PV + np.pi, AMAX + 0.2), UP, buff=0.1),
                      da.label("w ∥ v", 30).next_to(G(PV + 2 * np.pi, AMAX + 0.2), UP, buff=0.1))
        self.add(gax, zeros, curve, glab)

        def gdot():
            p = ph.get_value()
            return Dot(G(p, wedge(V, w_of(p))), radius=0.1, color=C["third"])

        self.add(always_redraw(gdot))

        def sign_text():
            val = wedge(V, w_of(ph.get_value()))
            num = f"{val:+.2f}".replace("-", "−")
            if val > 5e-3:
                return f"counterclockwise: {num}"
            return f"clockwise: {num}" if val < -5e-3 else "parallel: 0.00"

        # 지금 값은 그래프 옆이 아니라 여기에 둔다(그래프 곡선과 겹치지 않도록)
        sign = always_redraw(lambda: da.label(sign_text(), 32).move_to(O + np.array([-2.75, -2.5, 0]),
                                                                       aligned_edge=LEFT))
        self.add(sign)
        self.wait(2.0)
        self.play(ph.animate.set_value(PV + np.pi), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        self.play(ph.animate.set_value(PV + 1.5 * np.pi), run_time=2.0, rate_func=smooth)
        self.wait(1.5)
        self.play(ph.animate.set_value(PV + 2 * np.pi), run_time=2.0, rate_func=smooth)
        self.wait(1.5)
        self.play(ph.animate.set_value(PW0 + 2 * np.pi), run_time=1.5, rate_func=smooth)
        note = da.mtex(r"$dx\wedge dy\,(w, v) = -\,dx\wedge dy\,(v, w)$", 34).move_to([3.45, -2.75, 0])
        self.play(FadeIn(note), run_time=0.6)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.0)

"""Animation 0.6.1: 두 stereographic chart와 transition map u ↦ u/|u|² (0.6절, Example 0.6.2, (0.6.2), Figure 0.6.1, 0.6.2).

unit sphere S². σ(x, y, z) = (x, y)/(1 − z) (북극 N에서), σ̃(x, y, z) = (x, y)/(1 + z) (남극 S에서). dgsym.stereographic(2).
왼쪽: N, S, p를 지나는 단면(p의 meridian 평면). 가로선은 chart 평면 z = 0과의 교선이다.
      N에서 p를 지나는 직선(파랑)이 σ(p)에, S에서 p를 지나는 직선(주황)이 σ̃(p)에 닿는다.
오른쪽: chart 평면을 위에서 본 것. u = σ(p)(파랑)와 v = σ̃(p)(주황)는 같은 반직선 위에 있고 |u||v| = 1, 곧 v = u/|u|².
      회색 반직선은 왼쪽 단면의 가로선이다. 점선 원은 적도의 상(|u| = 1).
p의 움직임: (colatitude 60°, longitude 30°)(Figure 0.6.2의 점 x) → colatitude 35°(N 쪽) → 145°(S 쪽) → 60° →
longitude를 한 바퀴. colatitude θ에서 |u| = cot(θ/2), |v| = tan(θ/2).
자기검사: dgsym의 σ, σ̃, transition을 lambdify해 여러 점에서 σ̃ = transition(σ), 반지름 cot(θ/2), tan(θ/2),
Figure 0.6.2의 점 x에서 σ(x) = (3/2, √3/2), σ̃(x) = (1/2, √3/6).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-1-stereographic-charts.py
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgsym  # noqa: E402
from _helpers_b import arrow, polyline, sph  # noqa: E402

C = da.C
st = dgsym.stereographic(2)
X, U = st["coords"], st["chart_coords"]
sigma = sp.lambdify(X, list(st["sigma"]), "numpy")
sigma_s = sp.lambdify(X, list(st["sigma_south"]), "numpy")
trans = sp.lambdify(U, list(st["transition"]), "numpy")

# ---------------------------------------------------------------- 자기검사
for th in np.deg2rad([35, 60, 90, 120, 145]):
    for lon in np.deg2rad([0, 30, 200]):
        x = sph(th, lon)
        u, v = np.array(sigma(*x)), np.array(sigma_s(*x))
        assert np.allclose(np.array(trans(*u)), v)
        assert np.isclose(np.linalg.norm(u), 1 / np.tan(th / 2)) and np.isclose(np.linalg.norm(v), np.tan(th / 2))
x0 = sph(np.deg2rad(60), np.deg2rad(30))
assert np.allclose(sigma(*x0), [1.5, np.sqrt(3) / 2]) and np.allclose(sigma_s(*x0), [0.5, np.sqrt(3) / 6])

S = 0.9
LC = np.array([-3.55, -0.2, 0])     # 왼쪽 단면의 원점
RC = np.array([3.55, -0.2, 0])      # 오른쪽 chart 평면의 원점
HALF = 3.0


class Anim(da.Scene2D):
    def construct(self):
        th = ValueTracker(np.deg2rad(60))
        lon = ValueTracker(np.deg2rad(30))

        def L(r, z):
            return LC + S * np.array([r, z, 0])

        def R(u):
            return RC + S * np.array([u[0], u[1], 0])

        # 왼쪽: 단면(고정)
        circ = Circle(radius=S, stroke_color=C["main"], stroke_width=3, fill_color=C["surface"],
                      fill_opacity=0.35).move_to(LC)
        axis = Line(L(-HALF, 0), L(HALF, 0), stroke_color=C["aux"], stroke_width=2.5)
        Np, Sp = Dot(L(0, 1), radius=0.07), Dot(L(0, -1), radius=0.07)
        labN = da.mtex(r"$N$", 36).next_to(L(0, 1), UP + LEFT, buff=0.05)
        labS = da.mtex(r"$S$", 36).next_to(L(0, -1), DOWN + LEFT, buff=0.05)
        cap_l = da.label("slice through N, S, p", 30).move_to(LC + UP * 3.05)
        # 오른쪽: chart 평면(위에서)
        box = Square(side_length=2 * HALF * S, stroke_color=C["main"], stroke_width=1.5, fill_color=C["region"],
                     fill_opacity=0.12).move_to(RC)
        eq = DashedVMobject(Circle(radius=S, stroke_color=C["main"], stroke_width=2).move_to(RC), num_dashes=30)
        cap_r = da.label("chart plane z = 0, from above", 30).move_to(RC + UP * 3.05)
        o = Dot(RC, radius=0.05, color=C["main"])
        self.add(circ, axis, Np, Sp, labN, labS, cap_l, box, eq, cap_r, o)

        def left():
            t = th.get_value()
            p = np.array([np.sin(t), np.cos(t)])                  # (r, z) 단면 좌표
            su, sv = 1 / np.tan(t / 2), np.tan(t / 2)
            g = VGroup(Line(L(0, 1), L(su, 0), stroke_color=C["tangent"], stroke_width=4),
                       Line(L(0, -1), L(sv, 0), stroke_color=C["accent"], stroke_width=4),
                       Dot(L(su, 0), radius=0.08, color=C["tangent"]), Dot(L(sv, 0), radius=0.08, color=C["accent"]),
                       Dot(L(*p), radius=0.08, color=C["main"]))
            g.add(da.mtex(r"$p$", 36).next_to(L(*p), RIGHT, buff=0.12))
            # 바깥쪽 점의 라벨은 아래, 안쪽 점의 라벨은 왼쪽 위
            for val, tex, col in ((su, r"$\sigma(p)$", C["tangent"]), (sv, r"$\tilde\sigma(p)$", C["accent"])):
                m = da.mtex(tex, 34, color=col)
                if val >= max(su, sv) - 1e-9:
                    m.next_to(L(val, 0), DOWN, buff=0.15)
                else:
                    m.next_to(L(val, 0), UP + LEFT, buff=0.08)
                g.add(m)
            return g

        def right():
            t, a = th.get_value(), lon.get_value()
            x = sph(t, a)
            u, v = np.array(sigma(*x)), np.array(sigma_s(*x))
            dvec = np.array([np.cos(a), np.sin(a)])
            g = VGroup(Line(R(-HALF * dvec), R(HALF * dvec), stroke_color=C["aux"], stroke_width=2.5))
            ru, rv = np.linalg.norm(u), np.linalg.norm(v)
            for rr, col in ((ru, C["tangent"]), (rv, C["accent"])):
                if rr * S < HALF * S * 1.0:
                    g.add(Circle(radius=rr * S, stroke_color=col, stroke_width=2, stroke_opacity=0.55).move_to(RC))
            g.add(Dot(R(u), radius=0.08, color=C["tangent"]), Dot(R(v), radius=0.08, color=C["accent"]))
            g.add(da.mtex(r"$u$", 36, color=C["tangent"]).next_to(R(u), UR, buff=0.06))
            g.add(da.mtex(r"$v$", 36, color=C["accent"]).next_to(R(v), DR, buff=0.06))
            return g

        self.add(always_redraw(left), always_redraw(right))

        def nums():
            t = th.get_value()
            su, sv = 1 / np.tan(t / 2), np.tan(t / 2)
            # 왼쪽 단면 아래의 빈 곳에 둔다. 화면 맨 아래 12%는 멈춘 영상에서 control bar가 덮는다(§12.8)
            g = VGroup(da.label(f"colatitude {np.rad2deg(t):.0f}°   |u||v| = {su * sv:.2f}", 32),
                       da.label(f"|u| = {su:.2f}   |v| = {sv:.2f}", 32))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([-6.75, -1.85, 0], aligned_edge=UL)

        self.add(always_redraw(nums))
        formula = da.mtex(r"$v = \tilde\sigma\circ\sigma^{-1}(u) = u/|u|^2$", 38).to_edge(UP, buff=0.25)
        self.add(formula)
        self.wait(2.0)
        self.play(th.animate.set_value(np.deg2rad(35)), run_time=2.5, rate_func=smooth)
        self.wait(1.5)
        self.play(th.animate.set_value(np.deg2rad(145)), run_time=4.5, rate_func=smooth)
        self.wait(1.5)
        self.play(th.animate.set_value(np.deg2rad(60)), run_time=2.5, rate_func=smooth)
        self.wait(0.5)
        self.play(lon.animate.set_value(np.deg2rad(30) + 2 * np.pi), run_time=6.0, rate_func=smooth)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=1.5)

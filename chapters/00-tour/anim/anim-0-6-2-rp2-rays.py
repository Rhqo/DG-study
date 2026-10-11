"""Animation 0.6.2: ℝP²의 점 = 원점을 지나는 직선, chart φ₃와 point at infinity (0.6절, Example 0.6.4, (0.6.3)).

ℝ³의 좌표를 (x, y, w)로 쓰고 w축을 위로 그렸다. 원점 O를 지나는 직선(검정) 하나가 ℝP²의 점 [x : y : w]이다.
직선은 unit sphere와 대척점 두 개 ±d(검은 점)에서 만나므로 ℝP² = S²/±로도 볼 수 있다.
chart φ₃[x : y : w] = (x/w, y/w)는 직선이 평면 w = 1(하늘색, pinhole camera의 image plane)과 만나는 점(파랑)이다.
직선의 방향 d를 w축에서 25°만큼 기운 곳에서 출발해 70°까지 기울이면 만나는 점이 바깥으로 달아나고, 90°(w = 0)에서는
평면과 만나지 않는다: point at infinity [x : y : 0]. 그 점도 평면 x = 1(초록 테두리)의 chart φ₁[x : y : w] = (y/x, w/x)에서는
보통 점(초록)이다.
자기검사: dgsym.rpn_charts(2)의 φ₃, φ₁를 lambdify해 그린 교점과 비교, 대표를 −d로 바꿔도 같은 값(잘 정의됨),
transition φ₁∘φ₃⁻¹(u¹, u²) = (u²/u¹, 1/u¹)((0.6.3)).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-2-rp2-rays.py
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgsym  # noqa: E402
from _helpers_b import View, polyline, sphere_disk  # noqa: E402

C = da.C
rp = dgsym.rpn_charts(2)


Xs, Us = rp["coords"], rp["chart_coords"]
phi3 = sp.lambdify(Xs, list(rp["phi"](2)), "numpy")          # φ₃ (인덱스는 0부터)
phi1 = sp.lambdify(Xs, list(rp["phi"](0)), "numpy")          # φ₁
t31 = sp.lambdify(Us, list(rp["transition"](2, 0)), "numpy")   # φ₁∘φ₃⁻¹


def direction(alpha, beta):
    """w축에서 alpha만큼 기울고 방위각 beta인 단위벡터 (x, y, w)."""
    return np.array([np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)])


# ---------------------------------------------------------------- 자기검사
for al, be in ((0.4, 0.5), (1.0, 1.2), (1.25, 2.0)):
    d = direction(al, be)
    hit = d / d[2]
    assert np.allclose(phi3(*d), hit[:2]) and np.allclose(phi3(*(-d)), hit[:2]) and np.allclose(phi3(*(2.5 * d)), hit[:2])
    u = np.array(phi3(*d))
    assert np.allclose(phi1(*d), [u[1] / u[0], 1 / u[0]]) and np.allclose(t31(*u), phi1(*d))   # (0.6.3)
d_inf = direction(np.pi / 2, 0.55)
assert abs(d_inf[2]) < 1e-12 and np.allclose(phi1(*d_inf), [np.tan(0.55), 0])

VIEW = View(elev=16, azim=-62, scale=1.05, center=np.array([-4.1, -0.9, 0]))
HP = 3.0        # 평면 w = 1의 +x, +y 쪽 끝(교점이 가는 쪽)
HM = 1.8        # 평면 w = 1의 −x, −y 쪽 끝. 대칭으로 두면 왼쪽이 화면 밖으로 나간다
SQ3 = np.array([[-HM, -HM, 1], [HP, -HM, 1], [HP, HP, 1], [-HM, HP, 1]], float)
assert np.all(np.abs(VIEW.p(SQ3)[:, 0]) < 7.0) and np.all(np.abs(VIEW.p(SQ3)[:, 1]) < 3.9)       # 화면 안


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        al = ValueTracker(np.deg2rad(25))
        be = ValueTracker(0.35)
        show1 = ValueTracker(0.0)

        plane3 = Polygon(*v.p(SQ3), stroke_color=C["tangent"], stroke_width=2, fill_color=C["region"],
                         fill_opacity=0.22)
        sqx = np.array([[1, -HP, -1.3], [1, HP, -1.3], [1, HP, HP], [1, -HP, HP]], float)
        plane1 = Polygon(*v.p(sqx), stroke_color=C["third"], stroke_width=2.5, fill_color=C["region"],
                         fill_opacity=0.12)
        axes = VGroup(*[polyline(v.p(np.array([[0, 0, 0], e])), C["aux"], 2) for e in
                        (np.array([1.8, 0, 0]), np.array([0, 1.8, 0]), np.array([0, 0, 2.6]))])
        axl = VGroup(da.mtex(r"$x$", 32).move_to(v.p([2.0, 0, 0])), da.mtex(r"$y$", 32).move_to(v.p([0, 2.05, 0])),
                     da.mtex(r"$w$", 32).move_to(v.p([0, 0, 2.8])))
        lab3 = da.mtex(r"$w = 1$", 34, color=C["tangent"]).move_to(v.p([-HM, HP, 1]) + np.array([-1.0, 0.05, 0]))
        sphere = sphere_disk(v, fill_opacity=0.18, stroke_width=2)
        O = VGroup(Dot(v.p([0, 0, 0]), radius=0.07), da.mtex(r"$O$", 32).next_to(v.p([0, 0, 0]), DL, buff=0.06))
        self.add(sphere, axes, axl, plane3, lab3, O)

        def line():
            d = direction(al.get_value(), be.get_value())
            g = VGroup()
            tmax = 2.6
            if d[2] > 0.05:
                tmax = max(tmax, 1 / d[2] + 0.4)
            g.add(polyline(v.p(np.array([-2.4 * d, min(tmax, 4.5) * d])), C["main"], 4))
            g.add(Dot(v.p(d), radius=0.08), Dot(v.p(-d), radius=0.08))
            g.add(da.mtex(r"$d$", 32).next_to(v.p(d), LEFT, buff=0.1))
            g.add(da.mtex(r"$-d$", 32).next_to(v.p(-d), LEFT, buff=0.1))
            if d[2] > 1e-3:
                hit = d / d[2]
                if np.max(np.abs(hit[:2])) <= HP:
                    g.add(Dot(v.p(hit), radius=0.1, color=C["tangent"]))
            if show1.get_value() > 0.5 and abs(d[0]) > 1e-3:
                hit1 = d / d[0]
                g.add(Dot(v.p(hit1), radius=0.1, color=C["third"]))
            return g

        self.add(always_redraw(line))
        X0 = 0.05
        title = da.label("A point of RP² is a line through O", 34).to_edge(UP, buff=0.3)
        self.add(title)

        def coords():
            d = direction(al.get_value(), be.get_value())
            g = VGroup(da.label(f"[x : y : w] = [{d[0]:.2f} : {d[1]:.2f} : {d[2]:.2f}]", 30))
            if d[2] > 1e-3:
                u = phi3(*d)
                g.add(da.label(f"φ₃ = (x/w, y/w) = ({u[0]:.2f}, {u[1]:.2f})", 30, color=C["tangent"]))
            else:
                g.add(da.label("w = 0: misses the plane w = 1", 30, color=C["tangent"]))
            if show1.get_value() > 0.5:
                u1 = phi1(*d)
                g.add(da.label(f"φ₁ = (y/x, w/x) = ({u1[0]:.2f}, {u1[1]:.2f})", 30, color=C["third"]))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([X0, 1.8, 0], aligned_edge=UL)

        self.add(always_redraw(coords))
        note1 = da.label("d and −d: the same point of RP²", 30).move_to([X0, -0.6, 0], aligned_edge=LEFT)
        self.wait(1.5)
        self.play(FadeIn(note1), run_time=0.6)
        self.wait(1.5)
        self.play(al.animate.set_value(np.deg2rad(70)), be.animate.set_value(0.55), run_time=5.0, rate_func=smooth)
        self.wait(1.0)
        self.play(al.animate.set_value(np.pi / 2), run_time=2.5, rate_func=smooth)
        note2 = da.label("point at infinity [x : y : 0]", 30).move_to([X0, -1.2, 0],
                                                                                            aligned_edge=LEFT)
        self.play(FadeIn(note2), run_time=0.6)
        self.wait(1.5)
        lab1 = da.mtex(r"$x = 1$", 34, color=C["third"]).next_to(v.p([1, HP, HP]), RIGHT, buff=0.1).shift(DOWN * 0.3)
        self.play(FadeIn(plane1), FadeIn(lab1), show1.animate.set_value(1.0), run_time=1.0)
        note3 = da.label("in chart φ₁: an ordinary point", 30, color=C["third"]).move_to(
            [X0, -1.8, 0], aligned_edge=LEFT)
        self.play(FadeIn(note3), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.5)

"""Animation 0.6.5: SO(3)의 exponential map — rotation vector ω가 축 ω̂ 둘레의 각 |ω| 회전이 된다 (0.6절, (0.6.7), Figure 0.6.4).

오른쪽: Figure 0.6.4와 같은 공 {|ω| ≤ π}의 단면(가로 ω¹, 세로 ω³), 축 ω̂ = (cos 35°, 0, sin 35°).
왼쪽: 같은 시각의 회전 R = exp([ω]_×)(Rodrigues formula (0.6.7))로 돌린 물체. 카메라 모양의 frustum(검정, image
rectangle은 하늘색)과 body frame R e₁, R e₂, R e₃(주홍, 초록, 파랑). 검은 점선은 회전축 ω̂, 회색 직선은 world 좌표축, 연회색 frustum은 처음 자세(R = I₃)다.
ω = t ω̂를 t = 0 → 2(Figure 0.6.4의 ω) → π로 키운다. πω̂와 −πω̂는 같은 회전이므로 점은 공의 반대편 −πω̂로 넘어가고,
t가 2π가 되면(ω = 0) 물체는 처음 자세로 돌아온다. 곧 주황 선분은 SO(3) 안의 closed curve다.
자기검사: Rodrigues = 행렬 지수의 급수, Rᵀ R = I, det R = 1, R ω̂ = ω̂, 회전각 arccos((tr R − 1)/2) = |ω| (|ω| ≤ π),
exp(πω̂) = exp(−πω̂), exp(2πω̂) = I, 화면의 연속성(t가 π를 지날 때 R이 연속).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-5-so3-exp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, polyline  # noqa: E402

C = da.C


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w)
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def expm_series(A, terms=40):
    out, term = np.eye(3), np.eye(3)
    for k in range(1, terms):
        term = term @ A / k
        out = out + term
    return out


ANG = np.deg2rad(35)
WH = np.array([np.cos(ANG), 0.0, np.sin(ANG)])


def omega_of(t):
    """t ∈ [0, 2π] → 공 안의 대표 ω (|ω| ≤ π). t > π이면 반대편 −(2π − t) ω̂."""
    return t * WH if t <= np.pi else (t - 2 * np.pi) * WH


# ---------------------------------------------------------------- 자기검사
for t in (0.0, 0.7, 2.0, 3.0, np.pi, 4.0, 5.5, 2 * np.pi):
    R = rodrigues(t * WH)
    assert np.allclose(R, expm_series(hat(t * WH)))
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1) and np.allclose(R @ WH, WH)
    assert np.allclose(R, rodrigues(omega_of(t)))                      # 공 안의 대표와 같은 회전
    if t <= np.pi:
        assert np.isclose(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)), t)
assert np.allclose(rodrigues(np.pi * WH), rodrigues(-np.pi * WH)) and np.allclose(rodrigues(2 * np.pi * WH), np.eye(3))
assert np.allclose(rodrigues(omega_of(np.pi - 1e-6)), rodrigues(omega_of(np.pi + 1e-6)), atol=1e-5)

VIEW = View(elev=18, azim=-55, scale=1.7, center=np.array([-3.7, -0.25, 0]))
BC = np.array([3.55, -0.35, 0])
BS = 0.74
FR = np.array([[0.55, 0.38, 1.25], [-0.55, 0.38, 1.25], [-0.55, -0.38, 1.25], [0.55, -0.38, 1.25]])
INFO_Y = -2.65     # 아래 숫자 줄. poster의 control bar(아래 12%, y < −3) 위, 돌아가는 frustum보다 아래
_ys = np.array([VIEW.p(np.vstack([FR @ rodrigues(t * WH).T, 1.05 * rodrigues(t * WH).T]))[:, 1]
                for t in np.linspace(0, 2 * np.pi, 361)])
assert _ys.min() > INFO_Y + 0.3 and _ys.max() < 2.5                     # 숫자 줄, 위의 note와 겹치지 않는다


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        t = ValueTracker(0.0)
        # 왼쪽 고정: world 축과 회전축
        axes = VGroup(*[polyline(v.p(np.array([-e * 0.3, e * 1.6])), "#AAAAAA", 2) for e in np.eye(3)])
        axis_line = DashedVMobject(polyline(v.p(np.array([-1.9 * WH, 2.1 * WH])), C["main"], 2.5), num_dashes=24)
        ax_lab = da.mtex(r"$\hat\omega$", 36).next_to(v.p(2.1 * WH), UP, buff=0.08)
        ghost = VGroup(Polygon(*v.p(FR), stroke_color="#B0B0B0", stroke_width=2, fill_opacity=0),
                       *[polyline(v.p(np.array([[0, 0, 0], c])), "#B0B0B0", 2) for c in FR])
        self.add(axes, ghost, axis_line, ax_lab)

        def body():
            R = rodrigues(t.get_value() * WH)
            g = VGroup()
            C3 = FR @ R.T
            g.add(Polygon(*v.p(C3), stroke_color=C["main"], stroke_width=2.5, fill_color=C["region"], fill_opacity=0.35))
            for c in C3:
                g.add(polyline(v.p(np.array([[0, 0, 0], c])), C["main"], 2.5))
            for e, col in zip(np.eye(3), (C["normal"], C["third"], C["tangent"])):
                g.add(arrow(v.p([0, 0, 0]), v.p(1.05 * R @ e), col, 6, 0.2))
            g.add(Dot(v.p([0, 0, 0]), radius=0.06))
            return g

        self.add(always_redraw(body))
        # 오른쪽: 공
        disk = Circle(radius=np.pi * BS, stroke_color=C["main"], stroke_width=3, fill_color=C["region"],
                      fill_opacity=0.18).move_to(BC)
        half = DashedVMobject(Circle(radius=np.pi / 2 * BS, stroke_color=C["aux"], stroke_width=2).move_to(BC),
                              num_dashes=30)
        e2 = np.array([WH[0], WH[2], 0])
        seg = Line(BC - np.pi * BS * e2, BC + np.pi * BS * e2, stroke_color=C["accent"], stroke_width=4)
        bax = VGroup(Arrow(BC + LEFT * 3.0, BC + RIGHT * 3.0, buff=0, stroke_width=2, color=C["aux"], tip_length=0.15),
                     Arrow(BC + DOWN * 2.6, BC + UP * 2.6, buff=0, stroke_width=2, color=C["aux"], tip_length=0.15))
        blab = VGroup(da.mtex(r"$\omega^1$", 32).next_to(BC + RIGHT * 3.0, DOWN, buff=0.1),
                      da.mtex(r"$\omega^3$", 32).next_to(BC + UP * 2.6, RIGHT, buff=0.1),   # 위의 note와 겹치지 않게
                      da.mtex(r"$|\omega| \leq \pi$", 32).move_to(BC + np.array([-2.0, 2.15, 0])))
        ends = VGroup(Square(side_length=0.16, color=C["accent"], fill_opacity=1, stroke_color=C["main"],
                             stroke_width=1.5).move_to(BC + np.pi * BS * e2),
                      Square(side_length=0.16, color=C["accent"], fill_opacity=1, stroke_color=C["main"],
                             stroke_width=1.5).move_to(BC - np.pi * BS * e2))
        self.add(disk, half, bax, seg, ends, blab)

        def ball_pt():
            w = omega_of(t.get_value())
            q = BC + BS * np.array([w[0], w[2], 0])
            g = VGroup(arrow(BC, q, C["tangent"], 6, 0.2), Dot(q, radius=0.09, color=C["tangent"]))
            g.add(Dot(BC, radius=0.06))
            return g

        self.add(always_redraw(ball_pt))
        title = da.label("exp: rotation vector ω → rotation by |ω| about ω̂", 34).to_edge(UP, buff=0.25)
        self.add(title)

        def info():
            tt = t.get_value()
            w = omega_of(tt)
            ang = np.rad2deg(np.mod(tt, 2 * np.pi))
            g = VGroup(da.label(f"|ω| = {np.linalg.norm(w):.2f}", 32, color=C["tangent"]),
                       da.label(f"turned {ang:.0f}° about ω̂", 32))
            return g.arrange(RIGHT, buff=0.6).move_to([-6.8, INFO_Y, 0], aligned_edge=LEFT)

        self.add(always_redraw(info))
        self.wait(1.5)
        self.play(t.animate.set_value(2.0), run_time=3.5, rate_func=smooth)
        self.wait(1.5)
        self.play(t.animate.set_value(np.pi), run_time=2.0, rate_func=smooth)
        note = da.label("πω̂ and −πω̂: the same rotation", 32, color=C["accent"]).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(2.0)
        self.play(t.animate.set_value(2 * np.pi), run_time=5.0, rate_func=smooth)
        note2 = da.label("back to I₃: a closed curve in SO(3)", 32).next_to(title, DOWN, buff=0.2)
        self.play(FadeOut(note), FadeIn(note2), run_time=0.6)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=6.0)

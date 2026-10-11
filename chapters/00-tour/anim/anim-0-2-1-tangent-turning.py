"""Animation 0.2.1: curvature = unit tangent vector가 도는 빠르기 (Section 0.2, Figure 0.2.1, Definition 0.2.4).

Figure 0.2.1과 같은 곡선 γ(t) = (t, sin t), 0 ≤ t ≤ 2π. 점이 곡선을 arc length에 대해 일정한 빠르기로 지나간다.
왼쪽 위: 곡선과 움직이는 점의 unit tangent vector t (파랑, 0.7배). 지나온 곳의 9개 t(Figure 0.2.1의 점)는 흐리게 남긴다.
오른쪽 위: 같은 t를 원점에서 그린 것과 tangent angle θ (x축에서 잰 각).
아래: θ(s)의 그래프와 지금 점에서의 접선(주황). 그 기울기가 κ_s = dθ/ds (Theorem 4.4.9)이다.
κ_s = −sin t/(1 + cos² t)^{3/2}: 0 < t < π에서 음수(시계 방향으로 돈다), π < t < 2π에서 양수, t = π가 inflection point.
자기검사: dgsym.signed_curvature의 식이 위와 같고, 수치 미분한 dθ/ds가 κ_s와 같다. θ(0) = 45°, θ(π) = −45°.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, UP, Arc, Circle, DashedLine, Dot, FadeIn, Line, ValueTracker, VGroup,
                   always_redraw, linear)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
DOWN_ = np.array([0.0, -1.0, 0.0])

ts = sp.symbols("t", real=True)
ks_sym = dgsym.signed_curvature(sp.Matrix([ts, sp.sin(ts), 0]), ts)
assert sp.simplify(ks_sym + sp.sin(ts) / (1 + sp.cos(ts) ** 2) ** sp.Rational(3, 2)) == 0
ks = sp.lambdify(ts, ks_sym, "numpy")

T = np.linspace(0, 2 * np.pi, 40001)
dP = np.stack([np.ones_like(T), np.cos(T)], axis=1)
speed = np.linalg.norm(dP, axis=1)
S = np.concatenate([[0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(T))])
L = S[-1]
TH = np.arctan2(np.cos(T), np.ones_like(T))                 # tangent angle θ
assert np.allclose(np.gradient(TH, S)[50:-50], ks(T[50:-50]), atol=1e-4)    # κ_s = dθ/ds
assert np.isclose(np.degrees(TH[0]), 45) and np.isclose(np.degrees(TH[len(T) // 2]), -45, atol=1e-3)
assert abs(L - 7.6404) < 1e-3


def t_of_s(s):
    return float(np.interp(s, S, T))


IDX9 = [int(np.argmin(np.abs(S - L * k / 8))) for k in range(9)]   # Figure 0.2.1의 아홉 점

# 화면 배치
CU = np.array([-6.55, 1.8, 0.0])         # 곡선 패널: (t, sin t) → CU + SC·(t, sin t)
SC = 1.12
CC = np.array([4.7, 1.85, 0.0])            # 원 패널 중심
RC = 1.15
GX0, GX1 = -5.6, 5.9                       # 그래프의 s축 (화면 x)
GY0, GH = -1.7, 0.85                      # θ = 0의 화면 y, θ = π/4의 높이 (−45°가 poster의 아래 20% 위에 오도록)


def cu(t):
    return CU + SC * np.array([t, np.sin(t), 0.0])


def gpt(s, th):
    return np.array([GX0 + (GX1 - GX0) * s / L, GY0 + GH * th / (np.pi / 4), 0.0])


def tangent_unit(t):
    v = np.array([1.0, np.cos(t), 0.0])
    return v / np.linalg.norm(v)


def frame(s):
    t = t_of_s(s)
    th = float(np.arctan2(np.cos(t), 1.0))
    k = float(ks(t))
    g = VGroup()
    # 곡선 위의 점과 t
    p = cu(t)
    g.add(ha.arrow2(p, p + 0.7 * SC * tangent_unit(t), C["tangent"], width=7, tip=0.24))
    g.add(Dot(p, radius=0.07, color=C["main"]))
    # 원 위의 t와 θ
    tip = CC + RC * tangent_unit(t)
    g.add(ha.arrow2(CC, tip, C["tangent"], width=7, tip=0.24))
    if abs(th) > 0.02:
        g.add(Arc(radius=0.45, start_angle=0, angle=th, arc_center=CC, color=C["main"], stroke_width=3))
    g.add(ha.row(r"$\theta =$", f"{ha.fmt(np.degrees(th), 0)}°", size=30).move_to(CC + [0, -RC - 0.42, 0]))
    g.add(da.mtex(r"$\mathbf{t}$", 34, C["tangent"]).move_to(p + 0.7 * SC * tangent_unit(t) + [-0.05, 0.38, 0]))
    # 그래프 위의 점과 기울기
    q = gpt(s, th)
    dx = 0.9
    slope_scr = k * (GH / (np.pi / 4)) / ((GX1 - GX0) / L)    # 화면 단위의 기울기
    d = np.array([dx, dx * slope_scr, 0.0]) / np.sqrt(1 + slope_scr ** 2)
    lo = min(1.0, (q[0] - GX0) / d[0])                       # 양 끝에서 축 밖(45° 라벨 쪽)으로 나가지 않게 자른다
    hi = min(1.0, (GX1 + 0.45 - q[0]) / d[0])
    g.add(Line(q - lo * d, q + hi * d, color=C["accent"], stroke_width=7))
    g.add(Dot(q, radius=0.07, color=C["main"]))
    g.add(ha.row(r"$\kappa_s = d\theta/ds =$", ha.fmt(k, 2), size=32, color=C["accent"])
          .move_to([2.6, GY0 + GH + 0.62, 0]))
    # 지나온 곡선의 그래프
    m = S <= s
    if m.sum() > 2:
        idx = np.nonzero(m)[0][::40]
        g.add(ha.polyline(np.array([gpt(S[i], TH[i]) for i in idx] + [q]), C["main"], 5))
    return g


class Anim(da.Scene2D):
    def construct(self):
        s = ValueTracker(0.0)
        tt = np.linspace(0, 2 * np.pi, 400)
        curve = ha.polyline(np.array([cu(t) for t in tt]), C["main"], 5)
        faint = VGroup()
        for i in IDX9:
            p = cu(T[i])
            faint.add(ha.arrow2(p, p + 0.7 * SC * tangent_unit(T[i]), ManimColor_light(), width=4, tip=0.18))
        circle = DashedLine(CC + [-RC, 0, 0], CC + [RC, 0, 0], color=C["aux"], stroke_width=2)
        ucirc = Circle(radius=RC, color=C["aux"], stroke_width=2.5).move_to(CC)
        lab_circ = ha.row(r"$\mathbf{t}$", "at the origin", size=30, color=C["aux"]).move_to(CC + [0, RC + 0.42, 0])
        lab_t = da.mtex(r"$\mathbf{t}$", 34, C["tangent"]).move_to(cu(0.35) + [0.15, 0.85, 0])
        # 그래프 축
        ax = VGroup(Line([GX0, GY0, 0], [GX1 + 0.2, GY0, 0], color=C["main"], stroke_width=2.5),
                    Line([GX0, GY0 - GH - 0.15, 0], [GX0, GY0 + GH + 0.25, 0], color=C["main"], stroke_width=2.5))
        for th in (np.pi / 4, -np.pi / 4):
            ax.add(DashedLine(gpt(0, th), gpt(L, th), color=C["aux"], stroke_width=1.5))
        ax.add(da.mtex(r"$s$", 32).move_to([GX1 + 0.45, GY0, 0]))
        ax.add(da.mtex(r"$\theta$", 32).move_to([GX0 + 0.3, GY0 + GH + 0.4, 0]))
        ax.add(da.label("45°", 30, C["aux"]).next_to(gpt(0, np.pi / 4), LEFT, buff=0.15))
        ax.add(da.label("−45°", 30, C["aux"]).next_to(gpt(0, -np.pi / 4), LEFT, buff=0.15))
        neg = da.mtex(r"$\kappa_s < 0$", 30).move_to(gpt(L / 9, 0) + [0, -0.38, 0])
        pos = da.mtex(r"$\kappa_s > 0$", 30).move_to(gpt(8 * L / 9, 0) + [0, -0.38, 0])
        infl = VGroup(Circle(radius=0.12, color=C["accent"], stroke_width=4).move_to(cu(np.pi)),
                      da.label("inflection", 30, C["accent"]).move_to(cu(np.pi) + [-1.2, -0.45, 0]))   # 오른쪽 아래로 가는 t와 겹치지 않게

        dyn = always_redraw(lambda: frame(s.get_value()))
        self.add(curve, ucirc, circle, lab_circ, ax, dyn)
        self.wait(1.5)
        self.play(s.animate.set_value(L / 2), run_time=6.0, rate_func=linear)
        self.play(FadeIn(infl), FadeIn(neg), run_time=0.5)
        self.wait(1.5)
        self.play(s.animate.set_value(L), run_time=6.0, rate_func=linear)
        self.play(FadeIn(pos), FadeIn(faint), run_time=0.5)
        self.wait(2.0)


def ManimColor_light():
    from manim import ManimColor
    return ManimColor("#9CC3DE")


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=5.0)

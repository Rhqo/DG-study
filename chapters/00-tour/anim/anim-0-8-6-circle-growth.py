"""Animation 0.8.6: 반지름 ρ인 geodesic circle의 둘레가 세 model space에서 어떻게 자라는가 (0.8절, (0.8.8), Figure 0.8.4).

위: 세 패널에 중심에서 거리 ρ인 점들(geodesic circle, 주황)을 그린다. ρ를 0 → π/2 → π로 키운다.
  (a) unit sphere(K = 1): 중심은 화면 정면을 향하는 점. 둘레 2π sin ρ. ρ = π/2에서 great circle(윤곽), 그 뒤로는 뒤쪽(옅은 선)에서
      다시 줄어 ρ = π에서 대척점 한 점이 된다.
  (b) Euclidean plane(K = 0): 둘레 2πρ.
  (c) hyperbolic plane 𝕌²(K = −1): 중심 (0, 1). Euclidean으로는 중심 (0, cosh ρ), 반지름 sinh ρ인 원이고(Figure 0.8.4),
      둘레는 2π sinh ρ. 패널 밖으로 나간 부분은 그리지 않았다.
아래: Figure 0.8.4와 같은 그래프(가로 ρ, 세로 L(ρ), 세로축은 45에서 자름)와 현재 ρ의 세 점.
자기검사: (c)의 원 위의 점이 (0, 1)에서 hyperbolic distance ρ이고 g-length(수치) = 2π sinh ρ, (a)의 원 위의 점이 중심에서
거리 ρ이고 길이(수치) = 2π sin ρ.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-6-circle-growth.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, graticule, polyline, sphere_curve, sphere_disk  # noqa: E402

C = da.C
# 아래 90px(멈춤 상태의 control bar)를 비우려고 그래프를 올리고 위 패널을 조금 줄였다(§12.8)
VIEW = View(elev=25, azim=-50, scale=1.3, center=np.array([-4.7, 1.35, 0]))
CEN, U1, U2 = VIEW.d, VIEW.right, VIEW.up
T = np.linspace(0, 2 * np.pi, 361)


def sph_circle(rho, T=T):
    return np.cos(rho) * CEN + np.sin(rho) * (np.cos(T)[:, None] * U1 + np.sin(T)[:, None] * U2)


def hyp_circle(rho, T=T):
    return np.column_stack([np.sinh(rho) * np.cos(T), np.cosh(rho) + np.sinh(rho) * np.sin(T)])


def d_hyp(p, q):
    return np.arccosh(1 + ((p[..., 0] - q[0]) ** 2 + (p[..., 1] - q[1]) ** 2) / (2 * p[..., 1] * q[1]))


# ---------------------------------------------------------------- 자기검사
TF = np.linspace(0, 2 * np.pi, 40001)
for rho in (0.5, 1.0, 2.0, 2.8):
    H = hyp_circle(rho, TF)
    assert np.allclose(d_hyp(H, np.array([0.0, 1.0])), rho)
    mid = (H[1:] + H[:-1]) / 2
    assert np.isclose(np.sum(np.linalg.norm(np.diff(H, axis=0), axis=1) / mid[:, 1]), 2 * np.pi * np.sinh(rho), rtol=1e-4)
    Sc = sph_circle(min(rho, 3.0), TF)
    assert np.allclose(np.arccos(np.clip(Sc @ CEN, -1, 1)), min(rho, 3.0))
    assert np.isclose(np.sum(np.linalg.norm(np.diff(Sc, axis=0), axis=1)), 2 * np.pi * np.sin(min(rho, 3.0)), rtol=1e-4)

PL_C, PL_S = np.array([0.0, 1.35, 0]), 0.62
HY_O, HY_S, HY_W, HY_H = np.array([4.7, 0.0, 0]), 0.62, 2.2, 4.1
GX0, GX1, GY0, GY1 = -6.3, 6.3, -2.5, -0.82
GXMAX, GYMAX = 3.3, 45.0


def G(r, L):
    return np.array([GX0 + (GX1 - GX0) * r / GXMAX, GY0 + (GY1 - GY0) * L / GYMAX, 0])


def clip_poly(P, inside, color, width):
    g = VGroup()
    idx = np.where(inside)[0]
    if len(idx) < 2:
        return g
    for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(run) >= 2:
            g.add(polyline(P[run], color, width))
    return g


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        rho = ValueTracker(0.0)
        self.add(sphere_disk(v, fill_opacity=0.25), graticule(v, n_lat=5, n_lon=12, color="#C4C4C4", width=1.0))
        self.add(Dot(v.p(CEN), radius=0.05))
        self.add(Dot(PL_C, radius=0.05))
        self.add(Line(HY_O + LEFT * HY_W * HY_S, HY_O + RIGHT * HY_W * HY_S, stroke_color=C["main"], stroke_width=3.5),
                 Dot(HY_O + HY_S * UP, radius=0.05))
        heads = VGroup(da.label("sphere, K = 1", 30).move_to([-4.7, 2.95, 0]),
                       da.label("plane, K = 0", 30).move_to([0.0, 2.95, 0]),
                       da.label("hyperbolic, K = −1", 30).move_to([4.7, 2.95, 0]))
        self.add(heads)
        # 그래프
        gax = VGroup(Arrow(G(0, 0), G(GXMAX, 0) + RIGHT * 0.2, buff=0, stroke_width=2, color=C["aux"], tip_length=0.15),
                     Arrow(G(0, 0), G(0, GYMAX) + UP * 0.15, buff=0, stroke_width=2, color=C["aux"], tip_length=0.15))
        rr = np.linspace(0, np.pi, 300)
        curves = VGroup(
            clip_poly(np.array([G(r, 2 * np.pi * np.sinh(r)) for r in rr]), 2 * np.pi * np.sinh(rr) <= GYMAX,
                      C["third"], 4),
            polyline([G(r, 2 * np.pi * r) for r in rr], C["main"], 4),
            polyline([G(r, 2 * np.pi * np.sin(r)) for r in rr], C["accent"], 4))
        ticks = VGroup(da.mtex(r"$\rho$", 32).next_to(G(GXMAX, 0), RIGHT, buff=0.25),
                       da.mtex(r"$L(\rho)$", 30).next_to(G(0, GYMAX), RIGHT, buff=0.12),
                       da.mtex(r"$\pi/2$", 30).next_to(G(np.pi / 2, 0), DOWN, buff=0.08),
                       da.mtex(r"$\pi$", 30).next_to(G(np.pi, 0), DOWN, buff=0.08),
                       da.mtex(r"$2\pi\sinh\rho$", 30, color=C["third"]).next_to(G(2.55, 40), LEFT, buff=0.15)
                       .shift(UP * 0.2),
                       da.mtex(r"$2\pi\rho$", 30).next_to(G(3.05, 19.2), UP, buff=0.1),
                       # 주황 곡선과 검은 직선 사이(ρ ≈ 2.7–3.1)에 둔다
                       da.mtex(r"$2\pi\sin\rho$", 30, color=C["accent"]).move_to(
                           [G(2.9, 0)[0], (G(2.71, 2 * np.pi * np.sin(2.71))[1] + G(2.71, 2 * np.pi * 2.71)[1]) / 2, 0]))
        self.add(gax, curves, ticks)

        def circles():
            r = rho.get_value()
            g = VGroup()
            if r < 1e-3:
                return g
            if r < np.pi - 1e-3:
                g.add(sphere_curve(v, sph_circle(r), C["accent"], 5, hidden="faint", hidden_opacity=0.35))
            P = PL_C + PL_S * r * np.column_stack([np.cos(T), np.sin(T), 0 * T])
            # 패널 밖(위의 제목, 아래의 L 숫자)은 그리지 않는다
            g.add(clip_poly(P, (np.abs(P[:, 0] - PL_C[0]) < 2.1) & (np.abs(P[:, 1] - PL_C[1]) < 1.3), C["accent"], 5))
            H = hyp_circle(r)
            Hs = HY_O + HY_S * np.column_stack([H[:, 0], H[:, 1], 0 * T])
            g.add(clip_poly(Hs, (np.abs(H[:, 0]) <= HY_W) & (H[:, 1] <= HY_H), C["accent"], 5))
            for L, col in ((2 * np.pi * np.sin(r), C["accent"]), (2 * np.pi * r, C["main"]),
                           (2 * np.pi * np.sinh(r), C["third"])):
                if L <= GYMAX:
                    g.add(Dot(G(r, L), radius=0.08, color=col))
            g.add(DashedLine(G(r, 0), G(r, GYMAX), stroke_color=C["aux"], stroke_width=1.5, dash_length=0.08))
            return g

        self.add(always_redraw(circles))

        def nums():
            r = rho.get_value()
            vals = (2 * np.pi * np.sin(r), 2 * np.pi * r, 2 * np.pi * np.sinh(r))
            return VGroup(*[da.label(f"L = {L:.2f}", 30).move_to([x, -0.3, 0]) for L, x in zip(vals, (-4.7, 0, 4.7))])

        self.add(always_redraw(nums))
        rl = always_redraw(lambda: da.label(f"ρ = {rho.get_value():.2f}", 32).to_edge(UP, buff=0.2))
        self.add(rl)
        self.wait(1.5)
        self.play(rho.animate.set_value(np.pi / 2), run_time=5.0, rate_func=smooth)
        self.wait(1.5)
        self.play(rho.animate.set_value(np.pi), run_time=5.0, rate_func=smooth)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=8.0)

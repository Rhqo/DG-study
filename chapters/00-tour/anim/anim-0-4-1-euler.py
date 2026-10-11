"""Animation 0.4.1: normal section을 점 둘레로 돌리면 normal curvature가 Euler's formula를 따라 변한다
(Section 0.4, Figure 0.4.1, Definition 0.4.3, (0.4.1), Example 0.4.4).

Figure 0.4.1과 같은 saddle surface z = x² − y² (x² + y² ≤ 1)의 원점, N = (0, 0, 1) (위쪽),
principal direction e₁ = (1, 0, 0) (κ₁ = 2), e₂ = (0, 1, 0) (κ₂ = −2).
왼쪽: 단위 tangent vector w = cos θ e₁ + sin θ e₂ (파랑)와 N(주홍)이 펼치는 평면(하늘색)을 θ: 0 → π로 돌린다.
  평면과 surface의 교선(normal section, 주황)은 z = cos 2θ · s² (s는 w 방향의 거리)이다.
오른쪽: κ_n(θ) = κ₁ cos²θ + κ₂ sin²θ = 2 cos 2θ ((0.4.1))의 그래프와 지금의 값(점).
  θ = 0: κ_n = 2 (위로, N 쪽으로 휜다), θ = π/4: 0 (교선이 직선 x = y, surface 위에 놓인다), θ = π/2: −2.
자기검사: dgsym.shape_operator가 원점에서 diag(2, −2)이고 wᵀW w = 2 cos 2θ이다. 교선 z = cos 2θ s²의
원점에서의 signed curvature(N 쪽이 양수)가 2 cos 2θ와 같다(수치 미분).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, DashedLine, Dot, Line, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C

ex = dgsym.EXAMPLES["saddle"]
us, vs = ex["coords"]
W0 = dgsym.shape_operator(ex["expr"], us, vs).subs(ex["point"])
assert W0 == sp.diag(2, -2)
ths = sp.symbols("theta", real=True)
wv = sp.Matrix([sp.cos(ths), sp.sin(ths)])
assert sp.simplify((wv.T * W0 * wv)[0] - 2 * sp.cos(2 * ths)) == 0


def kn(t):
    return 2 * np.cos(2 * t)


for t0 in np.linspace(0, np.pi, 9):                     # 교선의 curvature를 직접 잰다
    h = 1e-4
    z = lambda s: np.cos(2 * t0) * s ** 2            # noqa: E731
    k_num = (z(h) - 2 * z(0) + z(-h)) / h ** 2       # 원점에서 z' = 0이므로 signed curvature = z''
    assert abs(k_num - kn(t0)) < 1e-6
    w = np.array([np.cos(t0), np.sin(t0)])
    s_ = np.linspace(-0.7, 0.7, 5)
    x, y = s_ * w[0], s_ * w[1]
    assert np.allclose(x ** 2 - y ** 2, z(s_))       # 교선은 surface 위에 있다
assert abs(kn(np.pi / 4)) < 1e-12

# 축척과 위치: θ = π/2(poster)에서 교선의 아래 끝 (0, ∓1, −1)이 화면 아래쪽 15%(정지 시 control bar와 그 음영) 위에 오게 한다
CAM = ha.Ortho(24, -58, scale=2.1, center=np.array([-3.65, -0.1, 0.0]))
for t0 in np.linspace(0, np.pi, 13):
    _s = np.linspace(-1, 1, 41)
    _sec = np.stack([_s * np.cos(t0), _s * np.sin(t0), np.cos(2 * t0) * _s ** 2], axis=1)
    assert CAM(_sec)[:, 1].min() > -4.0 + 0.15 * 8.0                # 아래쪽 15% 위
GX0, GX1, GY0, GH = 0.95, 6.15, -0.35, 0.95      # 그래프: θ ∈ [0, π] → [GX0, GX1], κ_n = 2 → GY0 + 2 GH
Rr, Pp = np.meshgrid(np.linspace(0, 1, 13), np.linspace(0, 2 * np.pi, 49), indexing="ij")
SX, SY = Rr * np.cos(Pp), Rr * np.sin(Pp)
SURF = np.stack([SX, SY, SX ** 2 - SY ** 2], axis=-1)


def gp(t, k):
    return np.array([GX0 + (GX1 - GX0) * t / np.pi, GY0 + GH * k, 0.0])


def left(t):
    pt = ha.Painter(CAM, ambient=0.6)
    pt.grid(SURF, color=C["surface"], opacity=0.92)
    w = np.array([np.cos(t), np.sin(t), 0.0])
    ss = np.linspace(-1, 1, 11)
    zz = np.linspace(-1.15, 1.15, 7)
    Spl = np.array([[s * w + np.array([0, 0, z]) for z in zz] for s in ss])
    for i in range(len(ss) - 1):
        for j in range(len(zz) - 1):
            Q = np.array([Spl[i, j], Spl[i + 1, j], Spl[i + 1, j + 1], Spl[i, j + 1]])
            pt.face(Q, C["region"], opacity=0.2, shade=False, stroke=C["region"], stroke_width=0, bias=0.02)
    border = np.array([Spl[0, 0], Spl[-1, 0], Spl[-1, -1], Spl[0, -1], Spl[0, 0]])
    for a, b in zip(border[:-1], border[1:]):
        pt.polyline(np.linspace(a, b, 8), C["tangent"], 2, bias=0.05)
    s = np.linspace(-1, 1, 61)
    sec = np.stack([s * w[0], s * w[1], np.cos(2 * t) * s ** 2], axis=1)
    pt.polyline(sec, C["accent"], 7, bias=0.15)
    pt.arrow(np.zeros(3), 0.55 * np.array([1.0, 0, 0]), C["tangent"], width=3, tip=0.16, bias=0.2)
    pt.arrow(np.zeros(3), 0.55 * np.array([0, 1.0, 0]), C["tangent"], width=3, tip=0.16, bias=0.2)
    pt.arrow(np.zeros(3), 0.75 * w, C["tangent"], width=7, tip=0.24, bias=0.3)
    pt.arrow(np.zeros(3), np.array([0, 0, 0.75]), C["normal"], width=7, tip=0.24, bias=0.3)
    pt.dot(np.zeros(3), radius=0.06, bias=0.4)
    g = pt.build()
    g.add(da.mtex(r"$\mathbf{w}$", 32, C["tangent"]).move_to(CAM(0.95 * w) + [0.0, 0.18, 0]))
    g.add(da.mtex(r"$\mathbf{N}$", 32, C["normal"]).move_to(CAM(np.array([0, 0, 0.92])) + [0.25, 0, 0]))
    g.add(da.mtex(r"$\mathbf{e}_1$", 30, C["tangent"]).move_to(CAM(np.array([0.72, 0, 0])) + [0.1, -0.25, 0]))
    g.add(da.mtex(r"$\mathbf{e}_2$", 30, C["tangent"]).move_to(CAM(np.array([0, 0.7, 0])) + [0.25, -0.12, 0]))
    return g


def right_static():
    g = VGroup(Line(gp(0, 0), gp(np.pi, 0) + [0.25, 0, 0], color=C["main"], stroke_width=2.5),
               Line(gp(0, -2.25), gp(0, 2.4), color=C["main"], stroke_width=2.5))
    for k in (2, -2):
        g.add(DashedLine(gp(0, k), gp(np.pi, k), color=C["aux"], stroke_width=1.6))
    tt = np.linspace(0, np.pi, 200)
    g.add(ha.polyline(np.array([gp(t, kn(t)) for t in tt]), "#C8C8C8", 4))
    for t, lab in ((np.pi / 4, r"$\pi/4$"), (np.pi / 2, r"$\pi/2$"), (3 * np.pi / 4, r"$3\pi/4$"), (np.pi, r"$\pi$")):
        g.add(Line(gp(t, 0) + [0, -0.08, 0], gp(t, 0) + [0, 0.08, 0], color=C["main"], stroke_width=2))
        g.add(da.mtex(lab, 30).move_to(gp(t, 0) + [0, -0.38, 0]))
    g.add(da.mtex(r"$\theta$", 32).move_to(gp(np.pi, 0) + [0.5, 0, 0]))
    g.add(da.mtex(r"$\kappa_n$", 32).move_to(gp(0, 2.4) + [0.0, 0.32, 0]))
    g.add(da.mtex(r"$\kappa_1 = 2$", 30).next_to(gp(np.pi, 2), RIGHT, buff=0.1).shift([-1.35, 0.3, 0]))
    g.add(da.mtex(r"$\kappa_2 = -2$", 30).next_to(gp(np.pi / 2, -2), np.array([0, -1.0, 0]), buff=0.12))
    g.add(da.mtex(r"$\kappa_n = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta$", 30).move_to([3.75, 3.45, 0]))
    return g


def right(t):
    g = VGroup()
    tt = np.linspace(0, t, max(2, int(200 * t / np.pi) + 2))
    if t > 1e-3:
        g.add(ha.polyline(np.array([gp(x, kn(x)) for x in tt]), C["accent"], 6))
    g.add(Dot(gp(t, kn(t)), radius=0.08, color=C["accent"]))
    g.add(ha.row(r"$\theta =$", f"{np.degrees(t):.0f}°", size=32).move_to([1.3, 2.75, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$\kappa_n =$", ha.fmt(kn(t), 2), size=32, color=C["accent"])
          .move_to([4.1, 2.75, 0], aligned_edge=LEFT))
    return g


class Anim(da.Scene2D):
    def construct(self):
        t = ValueTracker(0.0)
        lft = always_redraw(lambda: left(t.get_value()))
        rgt = always_redraw(lambda: right(t.get_value()))
        title = ha.row("saddle", r"$z = x^2 - y^2$", size=32).move_to([-3.65, 3.45, 0])
        self.add(title, right_static(), lft, rgt)
        self.wait(2.0)
        for t1, run, hold in ((np.pi / 4, 3.0, 1.8), (np.pi / 2, 3.0, 1.8), (np.pi, 5.0, 2.0)):
            self.play(t.animate.set_value(t1), run_time=run, rate_func=smooth)
            self.wait(hold)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.5)

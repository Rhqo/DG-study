"""Animation 0.1.1: 확대하면 평평하다 — 곡선과 surface를 한 점에서 확대한다 (Section 0.1, Figure 0.1.1, (0.1.1)).

왼쪽: ellipse γ(t) = (2 cos t, sin t), p = γ(π/3) (Figure 0.1.1과 같은 곡선과 점). 접선이 가로가 되도록
      p의 (t, n) 좌표로 돌려 놓고, 반너비 h인 창(가로 2h, 세로 0.6h)을 화면의 같은 크기로 보여 준다.
      창의 오른쪽 끝(접선 방향으로 h만큼 간 곳)에서 곡선이 접선에서 떨어진 거리 d를 주황 선분과 수치로 보인다.
      (0.1.1)에 의해 d ≈ κh²/2, κ ≈ 0.34.
오른쪽: unit sphere의 한 점 p 위에서 tangent plane의 원판(반지름 h)과 그 아래의 sphere 조각
      (그래프 z = √(1 − x² − y²) − 1). 원판 가장자리에서 d = 1 − √(1 − h²) ≈ h²/2 (κ = 1).
h: 1 → 0.1 → 0.01 (확대 ×1, ×10, ×100). 화면에 보이는 틈은 d/h ≈ κh/2에 비례하므로 0으로 간다.
자기검사: h ≤ 0.1에서 d와 κh²/2의 차이가 κh³ 이하. Figure 0.1.1의 수치 κ ≈ 0.34, h = 0.1에서 d ≈ 0.0017.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from manim import (LEFT, RIGHT, Dot, FadeIn, Line, Rectangle, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da

C = da.C
DOWN_ = np.array([0.0, -1.0, 0.0])
A_, B_, T0 = 2.0, 1.0, np.pi / 3


def gam(t):
    return np.stack([A_ * np.cos(t), B_ * np.sin(t)], axis=-1)


P0 = gam(T0)
V0 = np.array([-A_ * np.sin(T0), B_ * np.cos(T0)])
TV = V0 / np.linalg.norm(V0)
NV = np.array([-TV[1], TV[0]])                       # n_s = J t: 반시계 방향이므로 곡선이 휘어 들어가는 쪽
KAPPA = A_ * B_ / np.linalg.norm(V0) ** 3            # signed curvature det(γ', γ'')/|γ'|³
TT = np.linspace(T0 - 1.1, T0 + 1.1, 400001)
LOC = np.stack([(gam(TT) - P0) @ TV, (gam(TT) - P0) @ NV], axis=1)   # (t, n) 좌표
assert np.all(np.diff(LOC[:, 0]) > 0)                                # 이 범위에서 접선 방향 좌표는 단조
assert LOC[0, 0] < -1.15 and LOC[-1, 0] > 1.15


def d_curve(h):
    """접선 방향으로 h만큼 간 곳에서 곡선과 접선 사이의 거리."""
    return float(np.interp(h, LOC[:, 0], LOC[:, 1]))


def d_sphere(h):
    return 1.0 - np.sqrt(1.0 - h * h)


assert abs(KAPPA - 0.34) < 0.005                      # Figure 0.1.1: κ ≈ 0.34
assert np.all(LOC[:, 1] >= -1e-15)                    # 곡선은 접선의 한쪽(n 쪽)에 있다
for h in (0.1, 0.01):
    assert abs(d_curve(h) - KAPPA * h * h / 2) < KAPPA * h ** 3, (h, d_curve(h))
    assert abs(d_sphere(h) - h * h / 2) < h ** 3
assert abs(d_curve(0.1) - 0.0017) < 0.00005           # Figure 0.1.1의 캡션: h = 0.1에서 약 0.0017
assert d_curve(1.0) / 1.0 > 20 * d_curve(0.01) / 0.01  # 화면의 틈 d/h는 h에 비례해 줄어든다

# 화면 배치 ------------------------------------------------------------------------------------
WX = 2.6                                   # 왼쪽 창의 반너비(화면 단위). 창의 높이는 0.6 WX
CL = np.array([-3.95, 1.0, 0.0])           # 왼쪽 창에서 p의 위치
WR = 2.3                                   # 오른쪽 원판의 반지름(화면 단위)
CR = np.array([3.95, 1.9, 0.0])
CAM = ha.Ortho(elev=20, azim=-58, scale=1.0, center=CR)
STOPS = [1.0, 0.1, 0.01]


def num(x):
    nd = max(3, 1 - int(np.floor(np.log10(x))))
    return f"{x:.{nd}f}"


def curve_panel(h):
    s = WX / h
    keep = np.abs(LOC[:, 0]) <= h
    pts = LOC[keep]
    pts = pts[:: max(1, len(pts) // 700)]
    scr = np.stack([pts[:, 0] * s, pts[:, 1] * s, 0 * pts[:, 0]], axis=1) + CL
    g = VGroup()
    g.add(Rectangle(width=2 * WX, height=0.6 * WX, color=C["aux"], stroke_width=2).move_to(CL + [0, 0.15 * WX, 0]))
    g.add(Line(CL + [-1.08 * WX, 0, 0], CL + [1.08 * WX, 0, 0], color=C["tangent"], stroke_width=5))
    g.add(ha.polyline(scr, C["main"], 6))
    d = d_curve(h)
    g.add(Line(CL + [WX, 0, 0], CL + [WX, d * s, 0], color=C["accent"], stroke_width=8))
    g.add(Dot(CL, radius=0.07, color=C["main"]))
    return g


def sphere_panel(h):
    s = WR / h
    rho = np.linspace(0, h, 9)
    phi = np.linspace(0, 2 * np.pi, 49)
    Rr, Ph = np.meshgrid(rho, phi, indexing="ij")
    X, Y = Rr * np.cos(Ph), Rr * np.sin(Ph)
    Z = np.sqrt(np.clip(1 - X ** 2 - Y ** 2, 0, None)) - 1
    P = np.stack([X, Y, Z], axis=-1) * s
    g = VGroup()
    g.add(ha.surface_group(P, CAM, color=C["surface"], opacity=0.92, ambient=0.6))
    for i in range(1, len(rho)):
        g.add(ha.polyline(CAM(P[i]), C["main"], 1.5))
    for j in range(0, len(phi) - 1, 4):
        g.add(ha.polyline(CAM(P[:, j]), C["main"], 1.5))
    disk = np.stack([np.cos(phi), np.sin(phi), 0 * phi], axis=1) * WR
    g.add(ha.poly(CAM(disk), C["region"], C["tangent"], 3, opacity=0.35))
    d = d_sphere(h)
    edge = np.array([np.cos(-0.35), np.sin(-0.35), 0.0])
    g.add(Line(CAM(edge * WR), CAM(edge * WR + np.array([0, 0, -d * s])), color=C["accent"], stroke_width=8))
    g.add(ha.arrow3(CAM, [0, 0, 0], [0, 0, 1.1], C["normal"]))
    g.add(Dot(CAM(np.zeros(3)), radius=0.07, color=C["main"]))
    return g


class Anim(da.Scene2D):
    def construct(self):
        lh = ValueTracker(np.log(STOPS[0]))
        H = lambda: float(np.exp(lh.get_value()))  # noqa: E731

        left = always_redraw(lambda: curve_panel(H()))
        right = always_redraw(lambda: sphere_panel(H()))
        tl = da.mtex(r"$\gamma(t) = (2\cos t,\ \sin t),\ \ \kappa \approx 0.34$", 32).move_to([CL[0] + 0.3, 3.45, 0])
        tr = ha.row("unit sphere,", r"$\kappa = 1$", size=32).move_to([CR[0], 3.45, 0])
        lab_t = da.label("tangent line", 30, C["tangent"])
        lab_t.next_to(CL + [-WX, -0.18 * WX, 0], DOWN_, buff=0.12, aligned_edge=LEFT)
        lab_P = da.mtex(r"$T_pS$", 32, C["tangent"]).move_to(CAM(np.array([-0.95, 0.7, 0]) * WR) + [-0.55, 0.05, 0])
        lab_N = da.mtex(r"$\mathbf{N}$", 32, C["normal"]).move_to(CAM(np.array([0, 0, 1.1])) + [0.32, -0.05, 0])

        sym_h = da.mtex(r"$h =$", 34).move_to([-0.45, 2.35, 0])
        val_h = always_redraw(lambda: da.label(num(H()) if H() < 0.999 else "1", 34)
                              .next_to(sym_h, RIGHT, buff=0.18))
        zoom = always_redraw(lambda: da.label(f"zoom ×{STOPS[0] / H():.0f}", 30, C["aux"])
                             .move_to([0.1, 1.7, 0]))

        y0 = -1.5                      # 아래 두 줄이 poster의 아래 20% 위에 오도록
        sym_dl = da.mtex(r"$d =$", 32, C["accent"]).move_to([CL[0] - 0.6, y0, 0])
        val_dl = always_redraw(lambda: da.label(num(d_curve(H())), 32, C["accent"]).next_to(sym_dl, RIGHT, buff=0.15))
        sym_kl = da.mtex(r"$\kappa h^2/2 =$", 32).next_to(sym_dl, DOWN_, buff=0.3, aligned_edge=RIGHT)
        val_kl = always_redraw(lambda: da.label(num(KAPPA * H() ** 2 / 2), 32).next_to(sym_kl, RIGHT, buff=0.15))
        sym_dr = da.mtex(r"$d =$", 32, C["accent"]).move_to([CR[0] - 0.6, y0, 0])
        val_dr = always_redraw(lambda: da.label(num(d_sphere(H())), 32, C["accent"]).next_to(sym_dr, RIGHT, buff=0.15))
        sym_kr = da.mtex(r"$\kappa h^2/2 =$", 32).next_to(sym_dr, DOWN_, buff=0.3, aligned_edge=RIGHT)
        val_kr = always_redraw(lambda: da.label(num(H() ** 2 / 2), 32).next_to(sym_kr, RIGHT, buff=0.15))
        nums = VGroup(sym_dl, val_dl, sym_kl, val_kl, sym_dr, val_dr, sym_kr, val_kr)

        self.add(left, right, tl, tr, lab_t, lab_P, lab_N, sym_h, val_h, zoom)
        self.wait(1.0)
        self.play(FadeIn(nums), run_time=1.0)
        self.wait(2.0)
        for h in STOPS[1:]:
            self.play(lh.animate.set_value(np.log(h)), run_time=4.0, rate_func=smooth)
            self.wait(3.0 if h != STOPS[-1] else 2.0)      # 마지막 상태는 2초 (§12.8)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.5)

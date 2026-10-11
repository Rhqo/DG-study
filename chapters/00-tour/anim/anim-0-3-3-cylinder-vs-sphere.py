"""Animation 0.3.3: 평면 조각은 cylinder에 isometric하게 감기지만 sphere에 붙이려면 늘이거나 줄여야 한다
(Section 0.3, Example 0.3.7, Exercise 0.3.1, Theorema Egregium 예고).

같은 평면 원판(반지름 1, polar 좌표 (ρ, φ), 고리 10개와 바퀴살 24개)을 두 곳에 붙인다. curvature c를 0에서 1로 올린다.
왼쪽: cylinder(반지름 1/c)에 감기. isometric bending (x, y) ↦ (sin(cx)/c, y, −(1 − cos cx)/c), (x, y) = (ρ cos φ, ρ sin φ).
  polar 좌표의 first fundamental form은 E = 1, F = 0, G = ρ²로 평면과 같다. 가장자리 원의 길이 2π는 그대로다.
오른쪽: sphere(반지름 1/c)에 붙이기. 중심에서 잰 거리 ρ를 지키는 map
  (ρ, φ) ↦ (sin(cρ)/c · cos φ, sin(cρ)/c · sin φ, −(1 − cos cρ)/c)  (geodesic polar coordinates).
  E = 1, F = 0, G = sin²(cρ)/c². 반지름 ρ인 원의 길이가 2πρ에서 2π sin(cρ)/c로 줄어든다(c = 1, ρ = 1에서 2π sin 1 ≈ 5.29).
  면의 색: 원 방향으로 줄어든 비율 sin(cρ)/(cρ) (흰색 1 → 주황 0.84).
자기검사: dgsym.first_ff로 두 map의 E, F, G를 구한다. 매 c에서 바퀴살의 길이는 1, 가장자리 길이는 위의 값이다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, FadeIn, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
DOWN_ = np.array([0.0, -1.0, 0.0])

rho, phi = sp.symbols("rho phi", positive=True)
cs = sp.Rational(1)
cyl = sp.Matrix([sp.sin(rho * sp.cos(phi)), rho * sp.sin(phi), -(1 - sp.cos(rho * sp.cos(phi)))])
sph = sp.Matrix([sp.sin(rho) * sp.cos(phi), sp.sin(rho) * sp.sin(phi), -(1 - sp.cos(rho))])
Ec, Fc, Gc = dgsym.first_ff(cyl, rho, phi)
Es, Fs, Gs = dgsym.first_ff(sph, rho, phi)
for a, b in ((Ec, 1), (Fc, 0), (Gc, rho ** 2), (Es, 1), (Fs, 0), (Gs, sp.sin(rho) ** 2)):
    assert sp.simplify(a - b) == 0, (a, b)


def bend(c, x):
    """sin(cx)/c, (1 − cos cx)/c (c → 0에서도 매끄럽다)."""
    return x * np.sinc(c * x / np.pi), 0.5 * c * x ** 2 * np.sinc(c * x / (2 * np.pi)) ** 2


def P_cyl(Rh, Ph, c):
    x, y = Rh * np.cos(Ph), Rh * np.sin(Ph)
    s1, s2 = bend(c, x)
    return np.stack([s1, y, -s2], axis=-1)


def P_sph(Rh, Ph, c):
    s1, s2 = bend(c, Rh)
    return np.stack([s1 * np.cos(Ph), s1 * np.sin(Ph), -s2], axis=-1)


def plen(X):
    return float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=-1)))


NR, NP = 10, 48
Rg, Pg = np.meshgrid(np.linspace(0, 1, NR + 1), np.linspace(0, 2 * np.pi, NP + 1), indexing="ij")
rim_phi = np.linspace(0, 2 * np.pi, 4001)
spoke = np.linspace(0, 1, 2001)
for c in (0.0, 0.5, 1.0):
    assert abs(plen(P_cyl(1 + 0 * rim_phi, rim_phi, c)) - 2 * np.pi) < 1e-5
    rim = 2 * np.pi * (np.sin(c) / c if c > 0 else 1.0)
    assert abs(plen(P_sph(1 + 0 * rim_phi, rim_phi, c)) - rim) < 1e-5
    for p0 in (0.0, 0.7, 2.0):
        assert abs(plen(P_cyl(spoke, p0 + 0 * spoke, c)) - 1) < 1e-5
        assert abs(plen(P_sph(spoke, p0 + 0 * spoke, c)) - 1) < 1e-5
assert abs(2 * np.pi * np.sin(1) - 5.287) < 1e-3

# 아래의 E, F, G 줄이 poster 아래쪽 15%(정지 시 control bar와 그 음영) 위에 오도록 그림과 글자를 조금 올렸다
CAM_L = ha.Ortho(32, -60, scale=1.75, center=np.array([-3.55, 1.1, 0.0]), origin=(0, 0, -0.3))
CAM_R = ha.Ortho(32, -60, scale=1.75, center=np.array([3.55, 1.1, 0.0]), origin=(0, 0, -0.3))
WHITE_ = np.array([0.93, 0.93, 0.93])
ORANGE_ = np.array(da.ManimColor(C["accent"]).to_rgb())


def shrink_color(f):
    """원 방향 비율 f = sin(cρ)/(cρ): 1이면 흰색, 0.84(= sin 1)이면 주황."""
    t = np.clip((1 - f) / (1 - np.sin(1.0)), 0, 1)
    return da.ManimColor(tuple((1 - t) * WHITE_ + t * ORANGE_))


def ghost(cam, c, kind):
    """반지름 1/c인 cylinder 또는 sphere의 윤곽(연한 회색). 너무 크면 그리지 않는다."""
    g = VGroup()
    if c < 0.35:
        return g
    R0 = 1 / c
    col, w = "#C4C4C4", 2.0
    t = np.linspace(0, 2 * np.pi, 160)
    if kind == "sphere":
        from manim import Circle
        g.add(Circle(radius=R0 * cam.scale, color=col, stroke_width=w).move_to(cam(np.array([0, 0, -R0]))))
        eq = np.stack([R0 * np.cos(t), R0 * np.sin(t), -R0 + 0 * t], axis=1)
        g.add(ha.curve_group(eq, cam, col, w, (eq - [0, 0, -R0]) @ cam.view > 0, hidden="dashed"))
    else:                                            # cylinder의 단면 원(y = 0): 원판의 가운데 줄이 이 원을 따라 감긴다
        ring = np.stack([R0 * np.sin(t), 0 * t, -R0 + R0 * np.cos(t)], axis=1)
        g.add(ha.curve_group(ring, cam, col, w, None, hidden=None))
    return g


def panel(cam, Pf, c, colorize):
    S = Pf(Rg, Pg, c)
    cols = None
    if colorize:
        rc = 0.5 * (Rg[:-1, :-1] + Rg[1:, 1:])
        f = np.where(rc > 0, np.sinc(c * rc / np.pi), 1.0)
        cols = [[shrink_color(f[i, j]) for j in range(NP)] for i in range(NR)]
    pt = ha.Painter(cam, ambient=0.62)
    pt.grid(S, color=da.ManimColor(tuple(WHITE_)), colors=cols, opacity=1.0)
    for i in range(1, NR + 1):
        pt.polyline(S[i], "#7A7A7A" if i < NR else C["tangent"], 1.4 if i < NR else 5, bias=0.12)
    for j in range(0, NP, 2):
        pt.polyline(S[:, j], "#7A7A7A", 1.4, bias=0.12)
    return pt.build()


class Anim(da.Scene2D):
    def construct(self):
        c = ValueTracker(0.0)
        left = always_redraw(lambda: panel(CAM_L, P_cyl, c.get_value(), False))
        right = always_redraw(lambda: panel(CAM_R, P_sph, c.get_value(), True))
        tl = da.label("onto a cylinder", 32).move_to([-4.75, 3.4, 0])
        tr = da.label("onto a sphere", 32).move_to([4.75, 3.4, 0])
        rad = always_redraw(lambda: ha.row("radius", r"$1/c =$", "∞" if c.get_value() < 1e-3 else
                                           f"{1 / c.get_value():.2f}", size=30, color=C["aux"]).move_to([0, 3.4, 0]))
        rl = always_redraw(lambda: ha.row("rim length =", f"{plen(P_cyl(1 + 0 * rim_phi[::20], rim_phi[::20], c.get_value())):.2f}",
                                          size=32, color=C["tangent"]).move_to([-3.55, -1.95, 0]))
        rr = always_redraw(lambda: ha.row("rim length =", f"{plen(P_sph(1 + 0 * rim_phi[::20], rim_phi[::20], c.get_value())):.2f}",
                                          size=32, color=C["tangent"]).move_to([3.55, -1.95, 0]))
        gl = ha.row(r"$E = 1,\ F = 0,\ G = \rho^2$", size=30).move_to([-3.55, -2.52, 0])
        gr = ha.row(r"$E = 1,\ F = 0,\ G = \sin^2(c\rho)/c^2$", size=30).move_to([3.55, -2.52, 0])
        gh_l = always_redraw(lambda: ghost(CAM_L, c.get_value(), "cylinder"))
        gh_r = always_redraw(lambda: ghost(CAM_R, c.get_value(), "sphere"))
        leg = da.label("orange: circles shortened", 30, C["accent"]).move_to([3.55, 2.8, 0])
        self.add(gh_l, gh_r, left, right, tl, tr, rad, rl, rr, leg)
        self.wait(1.5)
        self.play(FadeIn(gl), FadeIn(gr), run_time=0.8)
        self.play(c.animate.set_value(1.0), run_time=5.0, rate_func=smooth)
        self.wait(3.0)
        self.play(c.animate.set_value(0.0), run_time=3.5, rate_func=smooth)
        self.wait(1.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.5)

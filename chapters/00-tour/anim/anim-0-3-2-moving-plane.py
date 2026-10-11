"""Animation 0.3.2: sphere 위를 움직이는 tangent plane과, UV 평면에서 본 first fundamental form의 ellipse
(Section 0.3, Figure 0.3.2, Figure 0.3.3, Definition 0.3.3, Definition 0.3.5, Example 0.3.6, Exercise 0.3.4).

unit sphere의 standard parametrization x(θ, φ) (dgsym.EXAMPLES["sphere"], r = 1). E = 1, F = 0, G = sin²θ.
왼쪽: 점 p = x(θ, φ)가 적도에서 북극 쪽으로(θ: π/2 → 0.3), 위도원을 따라 동쪽으로(φ: −1.2 → −0.2), 다시 적도로 움직인다.
  tangent plane T_pS(하늘색), x_θ와 x_φ(파랑, 실제 길이의 0.55배), N = x(주홍, 0.6배, 바깥쪽),
  T_pS 안에서 길이가 ε = 0.3인 tangent vector들의 끝이 이루는 원(주황).
오른쪽: UV 평면(가로 φ, 세로 θ, 위가 북극: Figure 0.3.3의 왼쪽과 같은 배치)의 점 q = (θ, φ)와
  같은 tangent vector들의 UV 성분 (a, b)가 이루는 ellipse a² + sin²θ b² = ε² (주황).
  반축: θ 방향 a_θ = ε, φ 방향 a_φ = ε/sin θ. 극에 가까울수록 φ 방향으로 길어진다(Exercise 0.3.4: θ = 30°에서 비 2).
자기검사: dgsym.first_ff와 unit_normal이 §7과 같다. |x_φ| = sin θ, ⟨N, x_θ⟩ = ⟨N, x_φ⟩ = 0.
그린 ellipse 위의 점은 I_q(w) = ε²를 만족하고, 그것을 a x_θ + b x_φ로 옮긴 3D 벡터의 길이가 ε이다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, Circle, DashedLine, Dot, Line, Rectangle, ValueTracker, VGroup, always_redraw,
                   smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
EPS = 0.3

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Es = dgsym.first_ff(ex["expr"], th, ph, ex["positive"])
assert all(sp.simplify(a - ex["expected"][k]) == 0 for a, k in zip(Es, "EFG"))
Ns = dgsym.unit_normal(ex["expr"], th, ph, ex["positive"])
assert sp.simplify(Ns - ex["expected"]["normal"]) == sp.zeros(3, 1)
Xe = ex["expr"].subs(rs, 1)
fX = sp.lambdify((th, ph), list(Xe), "numpy")
fXt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
fXp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")
fEFG = sp.lambdify(th, [e.subs(rs, 1) for e in Es], "numpy")


def vec(f, t, p):
    return np.array(f(t, p), float).ravel()


def ellipse_uv(t, n=72):
    """UV 성분 (a, b) = (θ 성분, φ 성분)의 ellipse a² E + b² G = ε²."""
    E, F, G = (float(x) for x in fEFG(t))
    s = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([EPS / np.sqrt(E) * np.cos(s), EPS / np.sqrt(G) * np.sin(s)], axis=1)


for t0, p0 in ((np.pi / 2, -0.6), (0.3, -0.6), (0.3, 0.4), (np.pi / 6, 0.0)):
    xt, xp, x0 = vec(fXt, t0, p0), vec(fXp, t0, p0), vec(fX, t0, p0)
    assert np.isclose(np.linalg.norm(xp), np.sin(t0)) and np.isclose(np.linalg.norm(xt), 1)
    assert abs(x0 @ xt) < 1e-12 and abs(x0 @ xp) < 1e-12                 # N = x ⟂ T_pS
    E, F, G = (float(x) for x in fEFG(t0))
    for a, b in ellipse_uv(t0, 12):
        assert np.isclose(E * a * a + 2 * F * a * b + G * b * b, EPS ** 2)
        assert np.isclose(np.linalg.norm(a * xt + b * xp), EPS)
assert np.isclose(1 / np.sin(np.pi / 6), 2.0)                            # Exercise 0.3.4

CAM = ha.Ortho(24, -40, scale=2.15, center=np.array([-3.75, -0.35, 0.0]))
UV0 = np.array([3.55, 1.45, 0.0])        # UV 패널에서 (θ, φ) = (π/2, 0)의 화면 위치
SU = 0.9                                 # 화면 단위 / rad
# 카메라 정면의 경도는 φ = −40°(−0.70)다. 그 양쪽 0.5 rad 안에서 움직여 tangent plane이 비스듬히 누워 보이지 않게 한다
TH_STOPS = [(np.pi / 2, -1.2), (0.3, -1.2), (0.3, -0.2), (np.pi / 2, -0.2)]


def uv(t, p):
    return UV0 + SU * np.array([p, np.pi / 2 - t, 0.0])


def sphere_static():
    pt = ha.Painter(CAM, ambient=0.6)
    T, Pp = np.meshgrid(np.linspace(0, np.pi, 25), np.linspace(-np.pi, np.pi, 49), indexing="ij")
    S = np.stack([np.broadcast_to(c, T.shape) for c in fX(T, Pp)], axis=-1)
    pt.grid(S, color=C["surface"], cull=True)
    g = pt.build()
    view = CAM.view
    tt = np.linspace(-np.pi, np.pi, 200)
    for k in range(1, 12):
        L = np.stack([np.broadcast_to(c, tt.shape) for c in fX(k * np.pi / 12 + 0 * tt, tt)], axis=-1)
        g.add(ha.curve_group(L, CAM, "#9A9A9A", 1.2, L @ view > 0, hidden=None))
    tt = np.linspace(0, np.pi, 120)
    for k in range(12):
        L = np.stack([np.broadcast_to(c, tt.shape) for c in fX(tt, k * np.pi / 6 + 0 * tt)], axis=-1)
        g.add(ha.curve_group(L, CAM, "#9A9A9A", 1.2, L @ view > 0, hidden=None))
    return g


def uv_static():
    g = VGroup()
    lo, hi = uv(np.pi, -np.pi), uv(0, np.pi)
    g.add(Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1], color=C["main"], stroke_width=2)
          .move_to(0.5 * (lo + hi)))
    for k in range(1, 6):
        g.add(DashedLine(uv(k * np.pi / 6, -np.pi), uv(k * np.pi / 6, np.pi), color="#BBBBBB", stroke_width=1.5))
    for k in range(-5, 6):
        g.add(DashedLine(uv(0, k * np.pi / 6), uv(np.pi, k * np.pi / 6), color="#BBBBBB", stroke_width=1.5))
    g.add(da.mtex(r"$\varphi$", 32).move_to(uv(np.pi, np.pi) + [0.3, 0.0, 0]))
    g.add(da.mtex(r"$\theta\downarrow$", 32).move_to(uv(np.pi, -np.pi) + [-0.42, 0.05, 0]))   # 아래로 커진다
    g.add(da.label("north pole", 30, C["aux"]).next_to(uv(0, 0), np.array([0, 1.0, 0]), buff=0.12))
    g.add(da.label("equator", 30, C["aux"]).next_to(uv(np.pi / 2, -np.pi), LEFT, buff=0.12))
    return g


def moving(t, p):
    g = VGroup()
    x0, xt, xp = vec(fX, t, p), vec(fXt, t, p), vec(fXp, t, p)
    et, ep = xt / np.linalg.norm(xt), xp / np.linalg.norm(xp)
    h = 0.55
    quad = np.array([x0 + h * (a * et + b * ep) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    g.add(ha.poly(CAM(quad), C["region"], C["tangent"], 2.5, opacity=0.4))
    E, F, G = (float(x) for x in fEFG(t))
    ab = ellipse_uv(t)
    circ = np.array([x0 + a * xt + b * xp for a, b in ab])
    g.add(ha.polyline(CAM(circ), C["accent"], 6))
    g.add(ha.arrow3(CAM, x0, 0.55 * xt, C["tangent"], width=6, tip=0.2))
    if np.linalg.norm(xp) > 0.05:
        g.add(ha.arrow3(CAM, x0, 0.55 * xp, C["tangent"], width=6, tip=0.2))
    g.add(ha.arrow3(CAM, x0, 0.6 * x0, C["normal"], width=6, tip=0.2))
    g.add(Dot(CAM(x0), radius=0.06, color=C["main"]))
    g.add(da.mtex(r"$\mathbf{x}_\theta$", 30, C["tangent"]).move_to(CAM(x0 + 0.8 * xt) + [0.1, -0.05, 0]))
    g.add(da.mtex(r"$\mathbf{x}_\varphi$", 30, C["tangent"]).move_to(CAM(x0 + 0.55 * xp) + [0.32, 0.2, 0]))
    g.add(da.mtex(r"$\mathbf{N}$", 30, C["normal"]).move_to(CAM(x0 + 0.8 * x0)))
    # UV 쪽
    q = uv(t, p)
    pts = np.array([q + SU * np.array([b, -a, 0.0]) for a, b in ab])     # 가로 = φ 성분, 세로 = −θ 성분
    g.add(ha.poly(pts, C["accent"], C["accent"], 5, opacity=0.25))
    g.add(Dot(q, radius=0.06, color=C["main"]))
    g.add(da.mtex(r"$q$", 30).move_to(q + [0.22, 0.2, 0]))
    # 수치
    st = np.sin(t)
    g.add(ha.row(r"$\theta =$", f"{np.degrees(t):.0f}°", size=30).move_to([0.75, -1.1, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$|\mathbf{x}_\varphi| = \sin\theta =$", f"{st:.2f}", size=30, color=C["tangent"])
          .move_to([0.75, -1.8, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$a_\varphi / a_\theta = 1/\sin\theta =$", f"{1 / st:.2f}", size=30,
                 color=C["accent"]).move_to([0.75, -2.5, 0], aligned_edge=LEFT))
    return g


class Anim(da.Scene2D):
    def construct(self):
        tv = ValueTracker(TH_STOPS[0][0])
        pv = ValueTracker(TH_STOPS[0][1])
        sph = sphere_static()
        uvp = uv_static()
        dyn = always_redraw(lambda: moving(tv.get_value(), pv.get_value()))
        fx = da.mtex(r"$\varepsilon = 0.3$", 30, C["aux"]).move_to([3.6, -1.1, 0], aligned_edge=LEFT)
        self.add(sph, uvp, dyn, fx)
        self.wait(2.0)
        self.play(tv.animate.set_value(TH_STOPS[1][0]), run_time=5.0, rate_func=smooth)
        self.wait(2.0)
        self.play(pv.animate.set_value(TH_STOPS[2][1]), run_time=3.0, rate_func=smooth)
        self.wait(1.5)
        self.play(tv.animate.set_value(TH_STOPS[3][0]), run_time=4.0, rate_func=smooth)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=8.0)

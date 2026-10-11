"""Animation 0.4.2: torus of revolution의 K 부호 — meridian을 따라 바깥 적도에서 안쪽 적도로 가며 두 principal curvature를 본다
(Section 0.4, Figure 0.4.2, Example 0.4.6, Exercise 0.4.3).

Figure 0.4.2와 같은 torus of revolution (R = 2, r = 0.8), dgsym.EXAMPLES["torus"]의 x(u, v)와
N = x_u × x_v/|x_u × x_v| (안쪽, 관의 중심원을 향한다). 시점은 Figure 0.4.2와 같은 elev 30°, azim −60°.
색: K = cos u/(r(R + r cos u)) (RdBu_r, 0이 흰색. 빨강 K > 0, 파랑 K < 0).
점 p가 화면 오른쪽 가장자리의 meridian(검정, v = azim + 90°의 단면 원. 그 평면은 화면에서 elev만큼 기운다) 위를 u: 0 → π → 0으로 움직인다
(바깥 적도 → 맨 위 → 안쪽 적도 → 되돌아옴. K(−u) = K(u)이므로 아래 절반은 같다). 주홍: N.
  meridian 방향의 principal curvature κ₁ = 1/r = 1.25 (언제나 N 쪽으로 휜다),
  parallel(청록, p를 지나는 수평 원) 방향의 principal curvature κ₂ = cos u/(R + r cos u).
오른쪽 그래프: κ₁(검은 점선), κ₂(청록), K = κ₁κ₂(검정 굵은 선). 배경: K > 0 빨강, K < 0 파랑.
  u = 0 (바깥 적도): κ₂ = 1/(R + r) > 0, K ≈ 0.45, elliptic. u = π/2 (맨 위): κ₂ = 0, parabolic.
  u = π (안쪽 적도): κ₂ = −1/(R − r) < 0, K ≈ −1.04, hyperbolic.
자기검사: dgsym.shape_operator의 eigenvalue가 1/r과 cos u/(R + r cos u)이고, dgsym.K_H의 K가 그 곱이다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, DashedLine, Dot, Line, Rectangle, ValueTracker, VGroup, always_redraw, smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
R, r = 2.0, 0.8
ex = dgsym.EXAMPLES["torus"]
us, vs = ex["coords"]
Rs, rs = ex["params"]
vals = {Rs: R, rs: r}
Ksym, Hsym = dgsym.K_H(ex["expr"], us, vs, ex["positive"])
assert sp.simplify(Ksym - ex["expected"]["K"]) == 0
Wsym = dgsym.shape_operator(ex["expr"], us, vs, ex["positive"]).subs(vals)
fW = sp.lambdify((us, vs), Wsym, "numpy")
Kf = sp.lambdify(us, Ksym.subs(vals), "numpy")


def k1(u):
    return 1 / r + 0 * u


def k2(u):
    return np.cos(u) / (R + r * np.cos(u))


for u0 in np.linspace(0, 2 * np.pi, 13):
    ev = np.sort(np.linalg.eigvals(np.array(fW(u0, 0.3), float)).real)[::-1]
    assert np.allclose(ev, sorted([k1(u0), k2(u0)], reverse=True))
    assert np.isclose(Kf(u0), k1(u0) * k2(u0))
assert np.isclose(Kf(0.0), 1 / (r * (R + r))) and np.isclose(Kf(np.pi), -1 / (r * (R - r)))
assert abs(Kf(0.0) - 0.446) < 1e-3 and abs(Kf(np.pi) + 1.042) < 1e-3

ELEV, AZ = 30, -60
a_ = np.radians(AZ)
CAM = ha.Ortho(ELEV, AZ, scale=1.05, center=np.array([-3.4, -0.1, 0.0]))
V0 = a_ + np.pi / 2                        # 오른쪽 단면(화면 오른쪽 방향의 meridian)
KMIN, KMAX = float(Kf(np.pi)), float(Kf(0.0))


def kcolor(k):
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps["RdBu_r"]
    x = 0.5 + 0.5 * (k / KMAX if k > 0 else -k / KMIN)       # TwoSlopeNorm(KMIN, 0, KMAX)
    return da.ManimColor(tuple(cmap(float(np.clip(x, 0, 1)))[:3]))


def X(u, v):
    return np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], axis=-1)


# 격자의 v 방향을 거꾸로 둔다. Painter.grid(cull=True)는 (P[i+1,j]−P[i,j]) × (P[i,j+1]−P[i,j])가 카메라 쪽인 면만
# 그리는데, 이 방향은 x_u × x_v, 곧 안쪽이다. 그대로 두면 앞면이 빠지고 뒷면(torus의 안쪽 벽)이 그려진다.
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(V0 + 2 * np.pi, V0, 73), indexing="ij")
SURF = X(Ug, Vg)
Uc = 0.5 * (Ug[:-1, :-1] + Ug[1:, 1:])
Vc = 0.5 * (Vg[:-1, :-1] + Vg[1:, 1:])
OUT = np.stack([np.cos(Uc) * np.cos(Vc), np.cos(Uc) * np.sin(Vc), np.sin(Uc)], axis=-1)   # 바깥쪽 normal
_fn = np.cross(SURF[1:, 1:] - SURF[:-1, :-1], SURF[:-1, 1:] - SURF[1:, :-1])                 # Painter.grid의 면 방향
assert np.all(np.einsum("ijk,ijk->ij", _fn, OUT) > 0)                                       # 바깥쪽이다
FACE_COL = [[kcolor(Kf(Uc[i, j])) for j in range(Uc.shape[1])] for i in range(Uc.shape[0])]


def scene(u):
    pt = ha.Painter(CAM, ambient=0.82)
    pt.grid(SURF, colors=FACE_COL, opacity=1.0, cull=True)
    uu = np.linspace(-0.05, np.pi + 0.05, 61)
    pt.polyline(X(uu, V0 + 0 * uu), C["main"], 5, bias=0.3)
    vv = np.linspace(V0, V0 + 2 * np.pi, 121)
    pt.polyline(X(u + 0 * vv, vv), C["third"], 6, bias=0.1)
    p = X(np.array(u), np.array(V0))
    n_in = -np.array([np.cos(u) * np.cos(V0), np.cos(u) * np.sin(V0), np.sin(u)])
    g = pt.build()
    # 안쪽 N은 관 속을 향하고 p는 화면 가장자리(윤곽)에 있어 surface에 가려진다. 깊이 정렬 없이 맨 위에 그린다.
    g.add(ha.arrow3(CAM, p, 0.65 * n_in, C["normal"], width=7, tip=0.24))
    g.add(Dot(CAM(p), radius=0.08, color=C["main"]))
    return g


GX0, GX1 = 1.15, 6.05
GY0, GS = -0.55, 1.45                       # 값 0의 화면 y, 값 1의 높이


def gp(u, y):
    return np.array([GX0 + (GX1 - GX0) * u / np.pi, GY0 + GS * y, 0.0])


def graph_static():
    g = VGroup()
    for a, b, col in ((0, np.pi / 2, "#F6D5D0"), (np.pi / 2, np.pi, "#D3E3F1")):
        lo, hi = gp(a, -1.15), gp(b, 1.4)
        g.add(Rectangle(width=hi[0] - lo[0], height=hi[1] - lo[1], stroke_width=0, fill_color=col, fill_opacity=1)
              .move_to(0.5 * (lo + hi)))
    g.add(Line(gp(0, 0), gp(np.pi, 0) + [0.2, 0, 0], color=C["main"], stroke_width=2.5))
    g.add(Line(gp(0, -1.15), gp(0, 1.45), color=C["main"], stroke_width=2.5))
    uu = np.linspace(0, np.pi, 300)
    g.add(DashedLine(gp(0, 1 / r), gp(np.pi, 1 / r), color=C["main"], stroke_width=3, dash_length=0.12))
    g.add(ha.polyline(np.array([gp(x, k2(x)) for x in uu]), C["third"], 4))
    g.add(ha.polyline(np.array([gp(x, Kf(x)) for x in uu]), C["main"], 6))
    for x, lab in ((np.pi / 4, r"$\pi/4$"), (np.pi / 2, r"$\pi/2$"), (3 * np.pi / 4, r"$3\pi/4$"), (np.pi, r"$\pi$")):
        g.add(da.mtex(lab, 30).move_to(gp(x, -1.15) + [0, -0.3, 0]))
    g.add(da.mtex(r"$u$", 32).move_to(gp(np.pi, 0) + [0.45, 0, 0]))
    g.add(da.mtex(r"$\kappa_1 = 1/r$", 30).move_to(gp(np.pi / 2, 1 / r) + [0, 0.32, 0]))
    g.add(da.mtex(r"$\kappa_2$", 32, C["third"]).move_to(gp(0.25, k2(0.25)) + [0.1, -0.35, 0]))
    g.add(da.mtex(r"$K$", 32).move_to(gp(1.0, Kf(1.0)) + [0.05, 0.35, 0]))
    return g


def kind(u):
    k = float(Kf(u))
    if abs(k) < 0.02:
        return "parabolic", C["main"]
    return ("elliptic", da.ManimColor("#B2182B")) if k > 0 else ("hyperbolic", da.ManimColor("#2166AC"))


def readouts(u):
    g = VGroup()
    x0 = 1.2
    g.add(ha.row(r"$u =$", f"{np.degrees(u):.0f}°", size=30).move_to([x0, 3.4, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$\kappa_1 =$", f"{k1(u):.2f}", size=30).move_to([x0, 2.75, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$\kappa_2 =$", ha.fmt(k2(u), 2), size=30, color=C["third"]).move_to([x0 + 2.35, 2.75, 0],
                                                                                         aligned_edge=LEFT))
    g.add(ha.row(r"$K =$", ha.fmt(Kf(u), 2), size=30).move_to([x0 + 2.35, 3.4, 0], aligned_edge=LEFT))
    name, col = kind(u)
    g.add(da.label(name, 32, col).move_to([-3.4, 3.4, 0]))
    g.add(Line(gp(u, -1.15), gp(u, 1.4), color=C["aux"], stroke_width=3))
    g.add(Dot(gp(u, Kf(u)), radius=0.07, color=C["main"]))
    g.add(Dot(gp(u, k2(u)), radius=0.06, color=C["third"]))
    return g


class Anim(da.Scene2D):
    def construct(self):
        u = ValueTracker(0.0)
        pic = always_redraw(lambda: scene(u.get_value()))
        ro = always_redraw(lambda: readouts(u.get_value()))
        legend = VGroup(da.label("meridian", 30, C["main"]).move_to([-5.6, 2.75, 0]),
                        da.label("parallel", 30, C["third"]).move_to([-3.4, 2.75, 0]),
                        ha.row(r"$\mathbf{N}$", "(inward)", size=30, color=C["normal"]).move_to([-1.2, 2.75, 0]))
        self.add(graph_static(), pic, ro, legend)
        self.wait(2.0)
        for u1, run, hold in ((np.pi / 2, 3.5, 1.6), (np.pi, 3.5, 2.2), (0.0, 4.5, 1.5)):
            self.play(u.animate.set_value(u1), run_time=run, rate_func=smooth)
            self.wait(hold)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=11.5)

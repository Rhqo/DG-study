"""Animation 0.4.3: Gauss map — 점이 surface 위의 고리를 돌 때 N의 끝이 unit sphere 위에 그리는 고리
(Section 0.4, Definition 0.4.1, Intuition 상자 "K는 Gauss map이 넓이를 바꾸는 비율", Example 0.4.6).

세 surface에서 같은 일을 동시에 한다. 위: surface와 고리(주황), 고리 위를 도는 점과 그 unit normal vector N(주홍).
아래: 같은 N을 원점에서 그린 것(unit sphere 안, 같은 시점). N의 끝이 그리는 고리가 Gauss map의 image N(고리)이다.
  (a) sphere S²(2), 바깥쪽 N = x/2. 고리: p₀(colatitude 60°)를 중심으로 surface 위의 거리 0.6인 원.
      image: 각반지름 0.3인 원. 넓이의 비 = 2π(1 − cos 0.3)/(8π(1 − cos 0.3)) = 1/4 = K. 같은 방향으로 돈다.
  (b) cylinder (반지름 1), 바깥쪽 N = (cos u, sin u, 0). 고리: (u, v) 평면에서 반지름 0.6인 원(E = G = 1인 좌표).
      image: 적도 위의 호를 오가는 곡선 — 넓이 0 = K.
  (c) saddle surface z = x² − y², 위쪽 N = (−2x, 2y, 1)/√(1 + 4x² + 4y²). 고리: 원점 둘레의 x² + y² = 0.2².
      image는 반대 방향으로 돈다(부호 있는 넓이가 음수). 넓이의 비는 고리를 줄이면 |K(0)| = 4로 간다(ρ = 0.2에서 약 3.4).
위 줄은 surface마다 축척이 다르다(아래 줄의 0.65배, 1.05배, 2.47배). 그래서 마지막에 아래 칸에 위의 고리를 아래 칸과 같은
축척으로 옮겨 회색 점선으로 겹친다(고리를 평행이동해 무게중심을 image의 무게중심에 맞춘 것). 넓이는 주황 image와 이 점선을 비교한다.
자기검사: unit sphere 위의 넓이는 중심 방향과 고리의 점들로 만든 spherical triangle의 solid angle을 더해 구한다.
(a)의 비가 1/4, (b)가 0, (c)가 음수이고 ρ → 0에서 −4에 가까워진다. dgsym.K_H의 K가 1/4, 0, −4다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, Circle, DashedVMobject, Dot, FadeIn, ValueTracker, VGroup, always_redraw, linear)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
AZ = -50.0
RS, RHO_S = 2.0, 0.6            # sphere 반지름, 고리 반지름(surface 위의 거리)
RC, RHO_C = 1.0, 0.6            # cylinder
RHO_H = 0.2                     # saddle
TH0 = np.radians(60.0)
PH0 = np.radians(AZ)

# dgsym으로 K 확인 ----------------------------------------------------------------------------------
_sph = dgsym.EXAMPLES["sphere"]
_K = dgsym.K_H(_sph["expr"], *_sph["coords"], _sph["positive"])[0]
assert sp.simplify(_K.subs(_sph["params"][0], 2) - sp.Rational(1, 4)) == 0
_cyl = dgsym.EXAMPLES["cylinder"]
assert sp.simplify(dgsym.K_H(_cyl["expr"], *_cyl["coords"], _cyl["positive"])[0]) == 0
_sad = dgsym.EXAMPLES["saddle"]
assert sp.simplify(dgsym.K_H(_sad["expr"], *_sad["coords"])[0].subs(_sad["point"]) + 4) == 0


def rot_to(n):
    """z축을 n으로 보내는 회전 행렬."""
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    s, c = np.linalg.norm(v), z @ n
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


N0_S = np.array([np.sin(TH0) * np.cos(PH0), np.sin(TH0) * np.sin(PH0), np.cos(TH0)])
ROT_S = rot_to(N0_S)
e1_S = ROT_S @ np.array([1.0, 0, 0])
e2_S = ROT_S @ np.array([0, 1.0, 0])


def loop_sphere(t):
    a = RHO_S / RS                                   # 각반지름
    d = np.cos(a) * N0_S[None] + np.sin(a) * (np.cos(t)[:, None] * e1_S + np.sin(t)[:, None] * e2_S)
    return RS * d, d                                 # 점, N


def loop_cyl(t):
    u = PH0 + RHO_C * np.cos(t) / RC
    v = RHO_C * np.sin(t)
    P = np.stack([RC * np.cos(u), RC * np.sin(u), v], axis=1)
    return P, np.stack([np.cos(u), np.sin(u), 0 * u], axis=1)


def saddle_N(x, y):
    n = np.stack([-2 * x, 2 * y, np.ones_like(x)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def loop_saddle(t, rho=RHO_H):
    x, y = rho * np.cos(t), rho * np.sin(t)
    return np.stack([x, y, x ** 2 - y ** 2], axis=1), saddle_N(x, y)


def sph_area(center, pts):
    """unit sphere 위 닫힌 고리(pts)가 둘러싼 부호 있는 넓이 (center에서 본 방향이 반시계면 양수)."""
    a = center / np.linalg.norm(center)
    total = 0.0
    for b, c in zip(pts, np.roll(pts, -1, axis=0)):
        num = a @ np.cross(b, c)
        den = 1 + a @ b + b @ c + c @ a
        total += 2 * np.arctan2(num, den)
    return total


TT = np.linspace(0, 2 * np.pi, 721)[:-1]
_, nS = loop_sphere(TT)
ratio_S = sph_area(N0_S, nS) / (2 * np.pi * RS ** 2 * (1 - np.cos(RHO_S / RS)))
_, nC = loop_cyl(TT)
ratio_C = sph_area(np.array([np.cos(PH0), np.sin(PH0), 0]), nC) / (np.pi * RHO_C ** 2)


def saddle_ratio(rho):
    _, nH = loop_saddle(TT, rho)
    area_surf = np.pi / 6 * ((1 + 4 * rho ** 2) ** 1.5 - 1)
    return sph_area(np.array([0, 0, 1.0]), nH) / area_surf


ratio_H = saddle_ratio(RHO_H)
assert abs(ratio_S - 0.25) < 1e-4 and abs(ratio_C) < 1e-6
assert -4 < ratio_H < -3 and abs(saddle_ratio(0.02) + 4) < 0.02            # ρ → 0에서 −4

COLS = [  # (title, camera elev, x 중심)
    ("S", 20, -4.6), ("C", 16, 0.0), ("H", 48, 4.6)]
TOP_Y, BOT_Y = 1.9, -0.85      # 결과 글자가 poster 아래쪽 15%(정지 시 control bar와 그 음영) 위에 오도록
SCALE_TOP = {"S": 0.62, "C": 1.0, "H": 2.35}
R_UNIT = 0.95


def cam_top(key, elev, x):
    org = {"S": (0, 0, 0), "C": (0, 0, 0), "H": (0, 0, 0.05)}[key]
    return ha.Ortho(elev, AZ, scale=SCALE_TOP[key], center=np.array([x, TOP_Y, 0]), origin=org)


def cam_bot(elev, x):
    return ha.Ortho(elev, AZ, scale=R_UNIT, center=np.array([x, BOT_Y, 0]))


def surface_static(key, cam):
    pt = ha.Painter(cam, ambient=0.6)
    if key == "S":
        T, P = np.meshgrid(np.linspace(0, np.pi, 19), np.linspace(-np.pi, np.pi, 37), indexing="ij")
        S = RS * np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
        pt.grid(S, color=C["surface"], cull=True)
    elif key == "C":
        U, V = np.meshgrid(np.linspace(-np.pi, np.pi, 37), np.linspace(-1.05, 1.05, 7), indexing="ij")
        S = np.stack([RC * np.cos(U), RC * np.sin(U), V], axis=-1)
        pt.grid(S, color=C["surface"], cull=True)
    else:
        X, Y = np.meshgrid(np.linspace(-0.5, 0.5, 15), np.linspace(-0.5, 0.5, 15), indexing="ij")
        S = np.stack([X, Y, X ** 2 - Y ** 2], axis=-1)
        pt.grid(S, color=C["surface"], cull=False)
    return pt.build()


def unit_static(cam):
    g = VGroup(Circle(radius=R_UNIT, color=C["aux"], stroke_width=2).move_to(cam(np.zeros(3))))
    t = np.linspace(0, 2 * np.pi, 120)
    eq = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1)
    g.add(ha.curve_group(eq, cam, "#AAAAAA", 1.6, eq @ cam.view > 0, hidden="dashed"))
    mer = np.stack([np.cos(t) * np.cos(PH0), np.cos(t) * np.sin(PH0), np.sin(t)], axis=1)
    g.add(ha.curve_group(mer, cam, "#AAAAAA", 1.6, mer @ cam.view > 0, hidden=None))
    return g


LOOPS = {"S": loop_sphere, "C": loop_cyl, "H": loop_saddle}
NSCALE = {"S": 0.9, "C": 0.75, "H": 0.32}


def column(key, elev, x, t, filled):
    ct, cb = cam_top(key, elev, x), cam_bot(elev, x)
    g = VGroup()
    P, Nn = LOOPS[key](TT)
    g.add(ha.polyline(ct(np.vstack([P, P[:1]])), C["accent"], 5))
    m = TT <= t
    if m.sum() >= 2:
        g.add(ha.polyline(cb(np.vstack([Nn[m]])), C["accent"], 6))
    if filled:
        g.add(ha.poly(ct(P), C["accent"], C["accent"], 0, opacity=0.3))
        g.add(ha.poly(cb(Nn), C["accent"], C["accent"], 0, opacity=0.3))
        g.add(same_scale_loop(key, cb))
    p1, n1 = LOOPS[key](np.array([t]))
    p1, n1 = p1[0], n1[0]
    g.add(ha.arrow3(ct, p1, NSCALE[key] * n1, C["normal"], width=6, tip=0.2))
    g.add(Dot(ct(p1), radius=0.06, color=C["main"]))
    g.add(ha.arrow3(cb, np.zeros(3), n1, C["normal"], width=6, tip=0.2))
    g.add(Dot(cb(n1), radius=0.06, color=C["normal"]))
    return g


def same_scale_loop(key, cam):
    """위의 고리를 평행이동해 무게중심을 image의 무게중심에 둔 것(회색 점선). 아래 칸과 같은 축척이다."""
    P, Nn = LOOPS[key](TT)
    G = P - P.mean(axis=0) + Nn.mean(axis=0)
    S = cam(np.vstack([G, G[:1]]))
    L = float(np.sum(np.linalg.norm(np.diff(S, axis=0), axis=1)))
    return DashedVMobject(ha.polyline(S, C["aux"], 3.5), num_dashes=max(10, int(L / 0.13)), dashed_ratio=0.55)


def _shoelace(S):
    return 0.5 * abs(float(S[:, 0] @ np.roll(S[:, 1], -1) - S[:, 1] @ np.roll(S[:, 0], -1)))


# 자기검사: sphere에서는 점선 고리와 image가 같은 평면(N₀에 수직)에 있으므로, 화면에서 본 넓이의 비도 정확히 1/4이다.
# saddle에서는 image가 화면에서도 점선보다 3배 넘게 넓다(같은 축척으로 비교할 수 있다).
_cS, _cH = cam_bot(COLS[0][1], 0.0), cam_bot(COLS[2][1], 0.0)
_PS, _NS = loop_sphere(TT)
assert abs(_shoelace(_cS(_NS)) / _shoelace(_cS(_PS - _PS.mean(axis=0) + _NS.mean(axis=0))) - 0.25) < 1e-3
_PH, _NH = loop_saddle(TT)
assert _shoelace(_cH(_NH)) / _shoelace(_cH(_PH - _PH.mean(axis=0) + _NH.mean(axis=0))) > 3.0


def orient_marks(key, elev, x):
    """고리와 image 위에 진행 방향 화살표 두 개씩."""
    ct, cb = cam_top(key, elev, x), cam_bot(elev, x)
    g = VGroup()
    for t0 in (0.5, 0.5 + np.pi):
        for cam, idx in ((ct, 0), (cb, 1)):
            if key == "C" and idx == 1:
                continue
            a = LOOPS[key](np.array([t0]))[idx][0]
            b = LOOPS[key](np.array([t0 + 0.5]))[idx][0]
            g.add(ha.arrow2(cam(a), cam(b), C["accent"], width=7, tip=0.3))
    return g


class Anim(da.Scene2D):
    def construct(self):
        t = ValueTracker(0.0)
        statics = VGroup()
        for key, elev, x in COLS:
            statics.add(surface_static(key, cam_top(key, elev, x)), unit_static(cam_bot(elev, x)))
        titles = VGroup(ha.row("sphere,", r"$r = 2$", size=30).move_to([-4.6, 3.5, 0]),
                        ha.row("cylinder,", r"$r = 1$", size=30).move_to([0.0, 3.5, 0]),
                        ha.row("saddle", r"$z = x^2 - y^2$", size=30).move_to([4.6, 3.5, 0]))
        mid = ha.row(r"$\mathbf{N}$", "moved to the unit sphere", size=30, color=C["aux"]).move_to([-3.3, 0.38, 0])
        mid2 = da.label("dashed: loop at the same scale", 30, C["aux"]).move_to([3.3, 0.38, 0])
        filled = ValueTracker(0.0)
        dyn = always_redraw(lambda: VGroup(*[column(k, e, x, t.get_value(), filled.get_value() > 0.5)
                                             for k, e, x in COLS]))
        res = VGroup(
            ha.row("area ratio", r"$= 1/4$", size=30).move_to([-4.6, -2.05, 0]),
            ha.row("area ratio", r"$= 0$", size=30).move_to([0.0, -2.05, 0]),
            ha.row("area ratio", r"$\approx {-3.4}$", size=30).move_to([4.6, -2.05, 0]),
            ha.row(r"$K = 1/4$", "(same way)", size=30, color=C["tangent"]).move_to([-4.6, -2.53, 0]),
            ha.row(r"$K = 0$", "(no area)", size=30, color=C["tangent"]).move_to([0.0, -2.53, 0]),
            ha.row(r"$K(0) = -4$", "(reversed)", size=30, color=C["tangent"]).move_to([4.6, -2.53, 0]))
        marks = VGroup(*[orient_marks(k, e, x) for k, e, x in COLS])
        self.add(statics, titles, mid, dyn)
        self.wait(1.5)
        self.play(t.animate.set_value(TT[-1]), run_time=8.0, rate_func=linear)
        filled.set_value(1.0)
        self.play(FadeIn(marks), FadeIn(res), FadeIn(mid2), run_time=1.0)
        self.wait(3.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.0)

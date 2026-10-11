"""Animation 0.2.3: helix 위를 움직이는 Frenet frame — torsion은 binormal이 도는 빠르기 (Section 0.2, Figure 0.2.3, (0.2.5)).

helix γ(u) = (a cos u, a sin u, b u) (dgsym.EXAMPLES["helix"]), a = 1. 화면과 캡션에서는 본문처럼 매개변수를 t로 쓴다. 점이 u를 따라 움직이고,
b(한 바퀴에 2πb만큼 올라간다)는 처음 0(반지름 1인 원, 평면곡선)에서 Figure 0.2.3의 값 b = 0.3으로 커진다.
왼쪽: helix와 움직이는 Frenet frame t(파랑), n(주홍), b(청록), 길이 0.7배.
오른쪽: 같은 세 벡터를 원점에서 그린 것(unit sphere 안). 끝점은 위도원 위를 돈다:
  t의 끝은 높이 b/c, n의 끝은 적도, b의 끝은 높이 a/c (c = √(a² + b²)).
  arc length에 대한 빠르기는 |t'| = κ = a/c², |b'| = τ = b/c² ((0.2.5)의 t' = κn, b' = −τn).
b = 0이면 b(binormal)는 위쪽에 고정되고 τ = 0이다. b = 0.3이면 κ ≈ 0.917, τ ≈ 0.275 (Figure 0.2.3).
자기검사: b = 0.3에서 dgsym.frenet_frame, curvature_torsion이 아래의 닫힌 식과 같다. frame이 orthonormal이고
b' = −τ n (arc length 기준, 수치 미분)이다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import sympy as sp
from manim import (LEFT, RIGHT, Circle, Dot, FadeIn, ValueTracker, VGroup, always_redraw, linear, smooth)

import _helpers_a as ha
import dganim as da
import dgsym

C = da.C
A = 1.0
B1 = 0.3

ex = dgsym.EXAMPLES["helix"]
(us,) = ex["coords"]
asym, bsym = ex["params"]
vals = {asym: 1, bsym: sp.Rational(3, 10)}
Gs = ex["expr"].subs(vals)
Ts, Ns, Bs = dgsym.frenet_frame(Gs, us)
kap, tau = dgsym.curvature_torsion(Gs, us)
assert sp.simplify(kap - ex["expected"]["kappa"].subs(vals)) == 0
assert sp.simplify(tau - ex["expected"]["tau"].subs(vals)) == 0
fT, fN, fB = (sp.lambdify(us, list(M), "numpy") for M in (Ts, Ns, Bs))


def frame(u, b):
    c = np.hypot(A, b)
    t = np.array([-A * np.sin(u), A * np.cos(u), b]) / c
    n = np.array([-np.cos(u), -np.sin(u), 0.0])
    bb = np.array([b * np.sin(u), -b * np.cos(u), A]) / c
    return t, n, bb


def kt(b):
    c2 = A * A + b * b
    return A / c2, b / c2


for u0 in np.linspace(0, 4 * np.pi, 9):
    t, n, bb = frame(u0, B1)
    assert np.allclose(t, np.array(fT(u0), float).ravel()) and np.allclose(n, np.array(fN(u0), float).ravel())
    assert np.allclose(bb, np.array(fB(u0), float).ravel())
    M = np.stack([t, n, bb])
    assert np.allclose(M @ M.T, np.eye(3)) and np.allclose(np.cross(t, n), bb)
    h = 1e-6
    db = (frame(u0 + h, B1)[2] - frame(u0 - h, B1)[2]) / (2 * h) / np.hypot(A, B1)   # d/ds = (1/c) d/du
    assert np.allclose(db, -kt(B1)[1] * n, atol=1e-8)                                  # b' = −τ n
assert abs(kt(B1)[0] - 0.917) < 1e-3 and abs(kt(B1)[1] - 0.275) < 1e-3                # Figure 0.2.3
assert kt(0.0) == (1.0, 0.0)

CAM_L = ha.Ortho(16, -58, scale=1.3, center=np.array([-3.6, -0.05, 0.0]))   # 아래 고리가 control bar 위에 오도록
CAM_R = ha.Ortho(16, -58, scale=1.55, center=np.array([3.75, 0.55, 0.0]))
U_END = 4 * np.pi


def helix_pts(b, uu):
    return np.stack([A * np.cos(uu), A * np.sin(uu), b * (uu - 2 * np.pi)], axis=1)


def left(u, b):
    g = VGroup()
    uu = np.linspace(0, U_END, 500)
    P = helix_pts(b, uu)
    radial = np.stack([np.cos(uu), np.sin(uu), 0 * uu], axis=1)
    g.add(ha.curve_group(P, CAM_L, C["main"], 4.5, radial @ CAM_L.view > 0, hidden="dashed", dash=0.07))
    for z0 in (-2 * np.pi * b, 2 * np.pi * b):
        cc = np.linspace(0, 2 * np.pi, 120)
        ring = np.stack([np.cos(cc), np.sin(cc), z0 + 0 * cc], axis=1)
        g.add(ha.curve_group(ring, CAM_L, C["aux"], 1.8, None, hidden=None))
    p = helix_pts(b, np.array([u]))[0]
    t, n, bb = frame(u, b)
    for v, col, lab in ((n, C["normal"], r"$\mathbf{n}$"), (t, C["tangent"], r"$\mathbf{t}$"),
                        (bb, C["third"], r"$\mathbf{b}$")):
        g.add(ha.arrow3(CAM_L, p, 0.7 * v, col, width=7, tip=0.24))
        g.add(da.mtex(lab, 32, col).move_to(CAM_L(p + 0.95 * v)))
    g.add(Dot(CAM_L(p), radius=0.07, color=C["main"]))
    return g


def right(u, b):
    g = VGroup()
    c = np.hypot(A, b)
    O = np.zeros(3)
    cc = np.linspace(0, 2 * np.pi, 160)
    g.add(Circle(radius=CAM_R.scale, color=C["aux"], stroke_width=2).move_to(CAM_R(O)))
    for z0, r0, col in ((0.0, 1.0, C["normal"]), (b / c, A / c, C["tangent"]), (A / c, b / c, C["third"])):
        if r0 < 1e-3:
            continue
        ring = np.stack([r0 * np.cos(cc), r0 * np.sin(cc), z0 + 0 * cc], axis=1)
        front = ring @ CAM_R.view > z0 * CAM_R.view[2] - 1e-9
        g.add(ha.curve_group(ring, CAM_R, col, 2.2, front, hidden="dashed", dash=0.06))
    t, n, bb = frame(u, b)
    for v, col in ((n, C["normal"]), (t, C["tangent"]), (bb, C["third"])):
        g.add(ha.arrow3(CAM_R, O, v, col, width=7, tip=0.24))
        g.add(Dot(CAM_R(v), radius=0.06, color=col))
    g.add(Dot(CAM_R(O), radius=0.05, color=C["main"]))
    k, tt = kt(b)
    g.add(ha.row(r"$b =$", f"{b:.2f}", size=32).move_to([4.6, 3.4, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$|\mathbf{t}'| = \kappa =$", f"{k:.3f}", size=32, color=C["tangent"])
          .move_to([1.35, -1.55, 0], aligned_edge=LEFT))
    g.add(ha.row(r"$|\mathbf{b}'| = \tau =$", f"{tt:.3f}", size=32, color=C["third"])
          .move_to([1.35, -2.2, 0], aligned_edge=LEFT))
    return g


class Anim(da.Scene2D):
    def construct(self):
        u = ValueTracker(0.0)
        b = ValueTracker(0.0)
        lft = always_redraw(lambda: left(u.get_value(), b.get_value()))
        rgt = always_redraw(lambda: right(u.get_value(), b.get_value()))
        # 화면의 매개변수 이름은 본문·Figure 0.2.3과 같이 t로 쓴다(스크립트 안에서는 굵은 t와 구별하려고 u)
        # 왼쪽 위는 t = 4π에서 b의 라벨이 올라가는 자리라 비워 둔다
        title = ha.row(r"$\gamma(t) = (\cos t,\ \sin t,\ b\,t)$", size=32).move_to([-0.3, 3.4, 0], aligned_edge=LEFT)
        self.add(title, lft, rgt)
        self.wait(1.0)
        self.play(u.animate.set_value(np.pi), run_time=3.5, rate_func=linear)
        self.wait(1.5)
        self.play(u.animate.set_value(2 * np.pi), b.animate.set_value(B1), run_time=4.0, rate_func=linear)
        self.play(u.animate.set_value(U_END), run_time=7.0, rate_func=linear)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=15.5)

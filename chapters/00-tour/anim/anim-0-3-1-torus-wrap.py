"""Animation 0.3.1: UV 격자가 torus of revolution T_{R,r}에 감긴다 (Section 0.3, Figure 0.3.1, Example 0.3.6).

Figure 0.3.1과 같은 R = 2, r = 0.8, 같은 checkerboard(u 방향 8칸, v 방향 16칸)와 parametrization
x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u) (dgsym.EXAMPLES["torus"]).
U에서 surface로 가는 과정을 세 단계의 변형으로 보인다.
  A. U = (0, 2π)²를 u 방향으로 r배, v 방향으로 R배 늘인 띠(2πR × 2πr)로 만든다.
  B. 띠를 u 방향으로 말아 반지름 r인 tube로 만든다(isometric bending, 칸의 모양이 그대로).
  C. tube를 v 방향으로 굽혀 중심원의 반지름이 R인 torus로 닫는다. 중심원에서 바깥쪽으로 r cos u만큼 떨어진 곳의
     길이는 (1 + k r cos u)배가 된다(k: 중심선의 curvature, 0 → 1/R). 끝에서 |x_v| = R + r cos u = √G.
주황 테두리: 같은 크기의 두 칸 — 바깥쪽 칸(u ∈ [0, π/4])과 안쪽 칸(u ∈ [3π/4, π]).
오른쪽 수치: 두 칸의 가운데에서 |x_u| = √E와 |x_v| = √G. 끝에서 √E = r = 0.8, √G = 2.74 (바깥), 1.26 (안쪽).
자기검사: 마지막 모양이 dgsym의 x(u, v)와 같고 implicit equation (√(x²+y²) − R)² + z² = r²를 만족한다.
dgsym.first_ff의 E, F, G가 (r², 0, (R + r cos u)²)이다. 단계 B는 모든 점에서 |∂P/∂u| = r을 지킨다.
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
R, r = 2.0, 0.8

ex = dgsym.EXAMPLES["torus"]
us, vs = ex["coords"]
Rs, rs = ex["params"]
vals = {Rs: R, rs: r}
Xf = sp.lambdify((us, vs), list(ex["expr"].subs(vals)), "numpy")
E_, F_, G_ = dgsym.first_ff(ex["expr"], us, vs, ex["positive"])
assert sp.simplify(E_ - rs ** 2) == 0 and F_ == 0 and sp.simplify(G_ - (Rs + rs * sp.cos(us)) ** 2) == 0


def bend(k, s):
    """(sin ks)/k, (1 − cos ks)/k, k → 0에서도 매끄럽다."""
    return s * np.sinc(k * s / np.pi), 0.5 * k * s ** 2 * np.sinc(k * s / (2 * np.pi)) ** 2


def P(U, V, a, m, k):
    """a: 단계 A (0 → 1), m: 말기 curvature (0 → 1/r), k: 굽히기 curvature (0 → 1/R)."""
    au, av = 1 + a * (r - 1), 1 + a * (R - 1)
    w = au * (U - np.pi / 2)
    s = av * (V - np.pi)
    sk, ck = bend(k, s)                        # sin(ks)/k, (1 − cos ks)/k
    sm, cm = bend(m, w)
    cks, sks = np.cos(k * s), np.sin(k * s)
    ek = np.stack([-cks, -sks, 0 * s], axis=-1)
    cen = np.stack([-R + ck, -sk, 0 * s], axis=-1)
    ez = np.array([0.0, 0.0, 1.0])
    return cen + (r - cm)[..., None] * ez - sm[..., None] * ek


NU, NV, SU, SV = 8, 16, 3, 3
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, NU * SU + 1), np.linspace(0, 2 * np.pi, NV * SV + 1), indexing="ij")
fin = P(Ug, Vg, 1, 1 / r, 1 / R)
ref = np.stack([np.broadcast_to(c, Ug.shape) for c in Xf(Ug, Vg)], axis=-1)
assert np.allclose(fin, ref, atol=1e-12)
assert np.allclose((np.hypot(fin[..., 0], fin[..., 1]) - R) ** 2 + fin[..., 2] ** 2, r ** 2)
for m_ in (0.3 / r, 0.7 / r, 1 / r):                                      # 단계 B: |∂P/∂u| = r
    d = (P(Ug + 1e-6, Vg, 1, m_, 0) - P(Ug, Vg, 1, m_, 0)) / 1e-6
    assert np.allclose(np.linalg.norm(d, axis=-1), r, atol=1e-5)
U_OUT, U_IN = np.pi / 8, 7 * np.pi / 8                                     # 두 주황 칸의 가운데
assert abs((R + r * np.cos(U_OUT)) - 2.739) < 1e-3 and abs((R + r * np.cos(U_IN)) - 1.261) < 1e-3

Uc = 0.5 * (Ug[:-1, :-1] + Ug[1:, 1:])
Vc = 0.5 * (Vg[:-1, :-1] + Vg[1:, 1:])
checker = ((np.floor(Uc / (2 * np.pi / NU)) + np.floor(Vc / (2 * np.pi / NV))) % 2).astype(int)
reg = np.array(da.ManimColor(C["region"]).to_rgb())
COL_A = da.ManimColor(tuple(0.55 * reg + 0.45))
COL_B = da.ManimColor("#FFFFFF")
COLORS = [[COL_A if checker[i, j] else COL_B for j in range(checker.shape[1])] for i in range(checker.shape[0])]
CELLS = [((0.0, np.pi / 4), (13 * np.pi / 8, 14 * np.pi / 8)),      # 바깥쪽 칸, 보는 쪽(v ≈ −62°)
         ((3 * np.pi / 4, np.pi), (5 * np.pi / 8, 6 * np.pi / 8))]   # 안쪽 칸, 구멍 건너편
TARGET = np.array([-1.6, -0.25, 0.0])


def camera(Pall):
    c0 = ha.Ortho(38, -62, scale=1.0)
    S = c0(Pall.reshape(-1, 3))
    lo, hi = S.min(axis=0), S.max(axis=0)
    s = min(1.25, 8.6 / (hi[0] - lo[0]), 6.2 / (hi[1] - lo[1]))
    return ha.Ortho(38, -62, scale=s, center=TARGET - s * 0.5 * (lo + hi))


def stage_label(a, m, k):
    if k > 1e-3:
        return ("torus of revolution", r"$T_{R,r}$") if k > 1 / R - 1e-3 else ("bending in", r"$v$")
    if m > 1e-3:
        return ("tube of radius", r"$r$") if m > 1 / r - 1e-3 else ("rolling in", r"$u$")
    if a < 1e-3:
        return (r"$U = (0, 2\pi)^2$",)
    return ("strip", r"$2\pi R \times 2\pi r$") if a > 1 - 1e-3 else ("stretching",)


def widths(a, k, u0):
    au, av = 1 + a * (r - 1), 1 + a * (R - 1)
    # 굽힌 뒤 길이 배율 1 + k·(중심선에서 e_k 쪽으로의 거리). 거리 = r cos u (말린 뒤), 말기 전에는 단계별로 계산
    return au, av * (1 + k * r * np.cos(u0))


def scene(a, m, k):
    S3 = P(Ug, Vg, a, m, k)
    cam = camera(S3)
    pt = ha.Painter(cam, ambient=0.55)
    pt.grid(S3, colors=COLORS, opacity=1.0)
    for (u0, u1), (v0, v1) in CELLS:
        uu = np.linspace(u0, u1, 8)
        vv = np.linspace(v0, v1, 8)
        edges = [(uu, v0 + 0 * uu), (u1 + 0 * vv, vv), (uu[::-1], v1 + 0 * uu), (u0 + 0 * vv, vv[::-1])]
        for eu, ev in edges:
            pt.polyline(P(eu, ev, a, m, k), C["accent"], 6, bias=0.25)
    return pt.build()


class Anim(da.Scene2D):
    def construct(self):
        a = ValueTracker(0.0)
        m = ValueTracker(0.0)
        k = ValueTracker(0.0)
        pic = always_redraw(lambda: scene(a.get_value(), m.get_value(), k.get_value()))
        lab = always_redraw(lambda: ha.row(*stage_label(a.get_value(), m.get_value(), k.get_value()), size=32)
                            .move_to([-1.6, 3.4, 0]))
        x0 = 3.05
        hE = ha.row(r"$\sqrt{E} = |\mathbf{x}_u| =$", size=30).move_to([x0, 2.2, 0], aligned_edge=LEFT)
        vE = always_redraw(lambda: da.label(f"{widths(a.get_value(), k.get_value(), U_OUT)[0]:.2f}", 30)
                           .next_to(hE, RIGHT, buff=0.25))
        hG = ha.row(r"$\sqrt{G} = |\mathbf{x}_v|$", size=30).move_to([x0, 1.15, 0], aligned_edge=LEFT)
        oG = ha.row("outer cell:", size=30, color=C["accent"]).move_to([x0 + 0.35, 0.45, 0], aligned_edge=LEFT)
        iG = ha.row("inner cell:", size=30, color=C["accent"]).move_to([x0 + 0.35, -0.2, 0], aligned_edge=LEFT)
        voG = always_redraw(lambda: da.label(f"{widths(a.get_value(), k.get_value(), U_OUT)[1]:.2f}", 30,
                                             C["accent"]).next_to(oG, RIGHT, buff=0.25))
        viG = always_redraw(lambda: da.label(f"{widths(a.get_value(), k.get_value(), U_IN)[1]:.2f}", 30,
                                             C["accent"]).next_to(iG, RIGHT, buff=0.25))
        fG = da.mtex(r"$\sqrt{G} = R + r\cos u$", 30, C["aux"]).move_to([x0, -1.2, 0], aligned_edge=LEFT)
        fP = da.mtex(r"$R = 2,\ r = 0.8$", 30, C["aux"]).move_to([x0, -1.9, 0], aligned_edge=LEFT)
        self.add(pic, lab, hE, vE, hG, oG, iG, voG, viG)
        self.wait(2.0)
        self.play(a.animate.set_value(1.0), run_time=3.0, rate_func=smooth)
        self.wait(1.5)
        self.play(m.animate.set_value(1 / r), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        self.play(k.animate.set_value(1 / R), run_time=5.0, rate_func=smooth)
        self.play(FadeIn(fG), FadeIn(fP), run_time=0.8)
        self.wait(2.2)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.5)

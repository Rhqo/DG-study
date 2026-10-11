"""Animation 0.5.5: 닫힌 mesh를 변형해도 angle defect의 합은 2πχ로 그대로다 (0.5절, Example 0.5.9, (0.5.3), (0.5.4)).

(1) icosphere(정이십면체를 3번 나눈 triangle mesh, vertex 642개, face 1280개)를 반지름 함수 R(u)·u로 변형한다.
    처음 R = 1(sphere) → z축 방향으로 늘여 가운데가 잘록한 땅콩 모양(허리는 δ_i/A_i ≈ −1) → 혹 세 개와 오목한 곳 하나를 더한 모양.
(2) torus of revolution(Figure 0.4.2와 같은 R = 2, r = 0.8, 격자 18 × 36의 triangle mesh)의 tube 굵기와 높이를 바꾼다.
색: vertex i의 discrete Gaussian curvature δ_i / A_i(δ_i = 2π − Σ_j θ_j, A_i = 이웃 face 넓이 합의 1/3)를 face마다
평균한 값. 발산형 컬러맵 RdBu_r(빨강 K > 0, 파랑 K < 0, 흰색 0. 양 끝은 sphere ±2.5, torus ±1.5)에 Lambert 음영을 더했다.
매 frame마다 Σ_i δ_i를 다시 계산해 보여 준다: sphere 쪽은 언제나 4π ≈ 12.566(χ = 2), torus 쪽은 0(χ = 0).
자기검사: 여러 변형 단계에서 Σ δ_i = 4π, 0 (1e-9), n_0 − n_1 + n_2 = 2, 0, 처음 sphere에서 δ_i/A_i ≈ 1, 변형한 sphere에
δ_i < 0인 vertex가 생김, 변형 중 R > 0.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-5-5-defect-invariance.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View  # noqa: E402

C = da.C


# ---------------------------------------------------------------- mesh
def icosphere(n_sub=3):
    t = (1 + 5 ** 0.5) / 2
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    for _ in range(n_sub):
        cache, F2 = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m / np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    V, F = np.array(V), np.array(F)
    cen = V[F].mean(1)
    flip = np.einsum("ij,ij->i", np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), cen) < 0
    F[flip] = F[flip][:, ::-1]
    return V, F


def torus_mesh(nu=18, nv=36):
    F = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            F += [(a, d, c), (a, c, b)]
    u, v = np.meshgrid(2 * np.pi * np.arange(nu) / nu, 2 * np.pi * np.arange(nv) / nv, indexing="ij")
    return u.ravel(), v.ravel(), np.array(F)


def n_edges(F):
    return len({(min(a, b), max(a, b)) for f in F for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))})


def defects(V, F):
    """vertex마다 angle defect δ_i와 barycentric 넓이 A_i."""
    ang = np.zeros(len(V))
    area = np.zeros(len(V))
    P = V[F]
    for k in range(3):
        a, b, c = P[:, k], P[:, (k + 1) % 3], P[:, (k + 2) % 3]
        u, w = b - a, c - a
        cs = np.einsum("ij,ij->i", u, w) / np.linalg.norm(u, axis=1) / np.linalg.norm(w, axis=1)
        np.add.at(ang, F[:, k], np.arccos(np.clip(cs, -1, 1)))
    fa = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    for k in range(3):
        np.add.at(area, F[:, k], fa / 3)
    return 2 * np.pi - ang, area


US, FS = icosphere(3)
BUMPS = [np.array(b, float) / np.linalg.norm(b) for b in ((1, -1, 0.3), (-0.8, -0.9, -0.2), (0.2, 0.6, 1.0))]
DENT = np.array([0.1, -1.0, -0.6]) / np.linalg.norm([0.1, -1.0, -0.6])


def sphere_shape(s1, s2):
    z = US[:, 2]
    R = 1 + s1 * 0.45 * (3 * z ** 2 - 1) / 2
    for b in BUMPS:
        R = R + s2 * 0.28 * np.exp(-np.sum((US - b) ** 2, 1) / 0.3)
    R = R - s2 * 0.25 * np.exp(-np.sum((US - DENT) ** 2, 1) / 0.35)
    return R[:, None] * US, R


TU, TV, FT = torus_mesh()


def torus_shape(s):
    rho = 0.8 * (1 + 0.3 * s * np.cos(3 * TV))
    h = 0.4 * s * np.sin(2 * TV)
    X = np.stack([(2 + rho * np.cos(TU)) * np.cos(TV), (2 + rho * np.cos(TU)) * np.sin(TV), rho * np.sin(TU) + h], 1)
    return X


# ---------------------------------------------------------------- 자기검사
assert len(US) - n_edges(FS) + len(FS) == 2 and len(US) == 642 and len(FS) == 1280
assert len(TU) - n_edges(FT) + len(FT) == 0
d0, a0 = defects(US, FS)
assert np.all(np.abs(d0 / a0 - 1) < 0.15)                      # 처음 sphere: δ_i/A_i ≈ K = 1
for s1, s2 in ((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1)):
    X, R = sphere_shape(s1, s2)
    assert R.min() > 0.4
    d, _ = defects(X, FS)
    assert abs(d.sum() - 4 * np.pi) < 1e-9
assert defects(sphere_shape(1, 1)[0], FS)[0].min() < 0          # 오목한 곳: δ < 0
_d, _a = defects(sphere_shape(1, 0)[0], FS)
assert -1.2 < np.mean((_d / _a)[np.abs(US[:, 2]) < 0.15]) < -0.8   # 늘이기만 해도 잘록한 허리는 δ_i/A_i ≈ −1 (캡션)
for s in (0, 0.5, 1):
    d, _ = defects(torus_shape(s), FT)
    assert abs(d.sum()) < 1e-9
assert np.allclose(torus_shape(0)[:, 2], 0.8 * np.sin(TU))       # s = 0: Figure 0.4.2의 torus of revolution

LIGHT = (-0.35, 0.5, 0.8)


def mesh_mob(view, X, F, m):
    """앞을 향한 face만 남기고 먼 것부터 그린다(painter's algorithm). 색은 face의 δ_i/A_i 평균."""
    d, A = defects(X, F)
    k = d / A
    vals = k[F].mean(1)
    Vs = np.column_stack([view.p(X)[:, :2], view.depth(X) * view.scale])
    P = Vs[F]
    nz = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])[:, 2]
    keep = np.where(nz > 0)[0]
    order = keep[np.argsort(P[keep, :, 2].mean(1))]
    cols = da.value_colors(vals[order], -m, m, "RdBu_r")
    return da.mesh_group(Vs, F[order], colors=cols, edge_color=ManimColor("#606060"), edge_width=0.35,
                         opacity=1.0, light=LIGHT), d.sum()


class Anim(da.Scene2D):
    def colorbar(self, m, top, bottom):
        g = VGroup()
        n = 40
        cols = da.value_colors(np.linspace(-m, m, n), -m, m, "RdBu_r")
        h = (top - bottom) / n
        for i, c in enumerate(cols):
            g.add(Rectangle(width=0.35, height=h, stroke_width=0, fill_color=c, fill_opacity=1)
                  .move_to([6.3, bottom + (i + 0.5) * h, 0]))
        g.add(Rectangle(width=0.35, height=top - bottom, stroke_color=C["main"], stroke_width=1.5)
              .move_to([6.3, (top + bottom) / 2, 0]))
        g.add(da.label("K > 0", 30).next_to([6.12, top, 0], LEFT, buff=0.12).shift(DOWN * 0.15))
        g.add(da.label("K < 0", 30).next_to([6.12, bottom, 0], LEFT, buff=0.12).shift(UP * 0.15))
        g.add(da.label("0", 30).next_to([6.12, (top + bottom) / 2, 0], LEFT, buff=0.12))
        return g

    def construct(self):
        view = View(elev=18, azim=-60, scale=1.75, center=np.array([-3.3, -0.3, 0]))
        az = ValueTracker(-60.0)
        s1, s2, s3 = ValueTracker(0.0), ValueTracker(0.0), ValueTracker(0.0)
        which = {"obj": "sphere"}
        total = {"v": 4 * np.pi}

        def mob():
            view.set(18 if which["obj"] == "sphere" else 32, az.get_value())
            if which["obj"] == "sphere":
                view.scale = 1.75
                X, _ = sphere_shape(s1.get_value(), s2.get_value())
                g, tot = mesh_mob(view, X, FS, 2.5)
            else:
                view.scale = 1.1
                view.center = np.array([-3.0, -0.3, 0])
                g, tot = mesh_mob(view, torus_shape(s3.get_value()), FT, 1.5)
            total["v"] = tot
            return g

        M = always_redraw(mob)
        X0 = 0.35
        title = da.label("Bend a closed mesh: the total stays", 34).to_edge(UP, buff=0.3)
        col = da.mtex(r"$\mathrm{color}:\ \delta_i/A_i\ \ (\mathrm{discrete}\ K)$", 34).move_to([X0, 2.3, 0], aligned_edge=LEFT)
        tot = always_redraw(lambda: da.mtex(r"$\sum_i \delta_i = %.3f$" % (0.0 if abs(total["v"]) < 5e-4 else total["v"]), 38).move_to(
            [X0, 1.0, 0], aligned_edge=LEFT))
        rhs = da.mtex(r"$2\pi\chi = 4\pi,\ \ \chi = 2$", 36).move_to([X0, -0.3, 0], aligned_edge=LEFT)
        cb = self.colorbar(2.5, 1.5, -2.7)
        self.add(M, title, col, tot, rhs, cb)
        self.wait(2.0)
        self.play(s1.animate.set_value(1.0), az.animate.set_value(-40), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        self.play(s2.animate.set_value(1.0), az.animate.set_value(-10), run_time=5.0, rate_func=smooth)
        self.wait(2.0)

        # torus
        self.play(FadeOut(rhs), FadeOut(cb), run_time=0.5)
        which["obj"] = "torus"
        az.set_value(-60)
        rhs2 = da.mtex(r"$2\pi\chi = 0,\ \ \chi = 0$", 36).move_to([X0, -0.3, 0], aligned_edge=LEFT)
        cb2 = self.colorbar(1.5, 1.5, -2.7)
        self.play(FadeIn(rhs2), FadeIn(cb2), run_time=0.6)
        self.wait(1.5)
        self.play(s3.animate.set_value(1.0), az.animate.set_value(-30), run_time=5.0, rate_func=smooth)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=11.0)

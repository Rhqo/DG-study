"""Animation 0.4.4: mean curvature flow가 모양을 매끄럽게 하며 줄인다 (Section 0.4, In computer vision 상자
"mean curvature flow와 mesh smoothing", (0.4.3), Exercise 0.4.4, [Desbrun99]의 implicit step을 다시 계산).

∂x/∂t = 2H N = Δ_S x ((0.4.3)). mesh: icosphere (level 3, vertex 642개, Chapter 31의 dgmesh.icosphere).
이산화: cotan Laplacian L(dgmesh.cotan_laplacian, Δ 부호)과 mixed Voronoi 넓이 M(dgmesh.mass_voronoi)으로
매 단계 (M − δt L) x_new = M x_old를 푼다(δt = 0.0015, L과 M은 매 단계 다시 만든다). t: 0 → 0.27.
왼쪽: 울퉁불퉁한 blob r = 1.2 (1 + 0.55 (x⁴ + y⁴ + z⁴ − 0.6) + 0.07 sin(5x + 1) sin 4y sin(3z + 2)).
오른쪽 위: 반지름 r₀ = 1.2인 sphere(같은 mesh, 같은 계산).
색: vertex의 mean curvature H = ⟨(M⁻¹L x)/2, N⟩ (바깥쪽 N, RdBu_r, 0이 흰색): 파랑 H < 0 (볼록), 빨강 H > 0 (오목).
오른쪽 아래: sphere의 r² (점, mesh에서 잰 값)과 공식 r(t)² = r₀² − 4t (선, Exercise 0.4.4).
영상 시간 s에 대해 t = 0.27 s²로 잡아 처음(요철이 빨리 사라지는 때)을 천천히 보여 준다.
자기검사: sphere의 r²가 공식과 0.015 안에서 맞는다. 처음 sphere의 H가 −1/r₀에 1% 안으로 맞는다.
blob의 반지름 상대 표준편차가 처음 5% 넘다가 끝에서 1.5% 아래로 준다. 넓이가 매 단계 줄어든다.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "31-meshes" / "lib"))

import numpy as np
from manim import (LEFT, RIGHT, DashedLine, Dot, Line, ValueTracker, VGroup, always_redraw, linear)

import _helpers_a as ha
import dganim as da
import dgmesh as dm

C = da.C
R0, DT, T_END = 1.2, 0.0015, 0.27
V1, F = dm.icosphere(3, 1.0)
assert len(V1) == 642 and len(F) == 1280


def blob(V):
    d = V / np.linalg.norm(V, axis=1, keepdims=True)
    x, y, z = d.T
    r = R0 * (1 + 0.55 * (x ** 4 + y ** 4 + z ** 4 - 0.6) + 0.07 * np.sin(5 * x + 1) * np.sin(4 * y) * np.sin(3 * z + 2))
    return d * r[:, None]


def flow(X0):
    X = X0.copy()
    out, Hs, areas = [X.copy()], [], []
    for k in range(int(round(T_END / DT))):
        L = dm.cotan_laplacian(X, F)
        m = dm.mass_voronoi(X, F)
        Hv = 0.5 * (L @ X) / m[:, None]
        Hs.append(np.einsum("ij,ij->i", Hv, dm.vertex_normals(X, F)))
        areas.append(dm.face_areas(X, F).sum())
        X = np.linalg.solve(np.diag(m) - DT * L, m[:, None] * X)
        out.append(X.copy())
    L = dm.cotan_laplacian(X, F)
    m = dm.mass_voronoi(X, F)
    Hs.append(np.einsum("ij,ij->i", 0.5 * (L @ X) / m[:, None], dm.vertex_normals(X, F)))
    areas.append(dm.face_areas(X, F).sum())
    return np.array(out), np.array(Hs), np.array(areas)


XS, HS, AS = flow(R0 * V1)
XB, HB, AB = flow(blob(V1))
TS = DT * np.arange(len(XS))
r2 = np.mean(np.sum(XS ** 2, axis=2), axis=1)
assert np.all(np.abs(r2 - (R0 ** 2 - 4 * TS)) < 0.015), np.max(np.abs(r2 - (R0 ** 2 - 4 * TS)))
assert abs(np.mean(HS[0]) + 1 / R0) < 0.01 / R0
rough = [np.std(np.linalg.norm(X, axis=1)) / np.mean(np.linalg.norm(X, axis=1)) for X in XB]
assert rough[0] > 0.05 and rough[-1] < 0.015
assert np.all(np.diff(AB) < 0) and np.all(np.diff(AS) < 0)

HMIN, HMAX = -3.0, 1.5


def hcolors(Hv):
    import matplotlib
    matplotlib.use("Agg")
    cmap = matplotlib.colormaps["RdBu_r"]
    hf = Hv[F].mean(axis=1)
    x = np.where(hf < 0, 0.5 - 0.5 * hf / HMIN, 0.5 + 0.5 * hf / HMAX)
    return [da.ManimColor(tuple(cmap(float(np.clip(v, 0, 1)))[:3])) for v in x]


CAM_B = ha.Ortho(22, -55, scale=1.2, center=np.array([-4.75, 0.35, 0.0]))     # 두 패널은 같은 축척
CAM_S = ha.Ortho(22, -55, scale=1.2, center=np.array([-1.15, 0.35, 0.0]))
GX0, GX1, GY0, GY1 = 1.65, 6.1, -2.25, 2.2


def gp(t, y):
    return np.array([GX0 + (GX1 - GX0) * t / 0.3, GY0 + (GY1 - GY0) * y / 1.5, 0.0])


def state(t):
    x = t / DT
    i = int(np.clip(np.floor(x), 0, len(XS) - 2))
    a = x - i
    lerp = lambda A: (1 - a) * A[i] + a * A[i + 1]  # noqa: E731
    return lerp(XB), lerp(HB), lerp(XS), lerp(HS), i


def frame(t):
    xb, hb, xs, hs, i = state(t)
    g = VGroup()
    g.add(ha.tri_group(xb, F, CAM_B, colors=hcolors(hb), cull=True, ambient=0.62))
    g.add(ha.tri_group(xs, F, CAM_S, colors=hcolors(hs), cull=True, ambient=0.62))
    n = i + 1
    pts = [gp(TS[j], r2[j]) for j in range(0, n, 6)]
    if pts:
        g.add(ha.dots_mobject(pts, size=0.09, color=C["tangent"]))
    g.add(Dot(gp(t, np.mean(np.sum(xs ** 2, axis=1))), radius=0.08, color=C["tangent"]))
    g.add(ha.row(r"$t =$", f"{t:.3f}", size=32).move_to([-2.95, 3.45, 0]))
    return g


def graph_static():
    g = VGroup(Line(gp(0, 0), gp(0.3, 0) + [0.2, 0, 0], color=C["main"], stroke_width=2.5),
               Line(gp(0, 0), gp(0, 1.5) + [0, 0.15, 0], color=C["main"], stroke_width=2.5))
    g.add(Line(gp(0, R0 ** 2), gp(0.3, R0 ** 2 - 1.2), color=C["accent"], stroke_width=4))
    g.add(da.mtex(r"$r^2 = r_0^2 - 4t$", 30, C["accent"]).move_to(gp(0.2, 1.05)))
    g.add(da.mtex(r"$t$", 32).move_to(gp(0.3, 0) + [0.45, 0, 0]))
    g.add(da.mtex(r"$r^2$", 32).move_to(gp(0, 1.5) + [0.0, 0.45, 0]))
    for tt in (0.1, 0.2, 0.3):
        g.add(da.label(f"{tt:.1f}", 30, C["aux"]).move_to(gp(tt, 0) + [0, -0.32, 0]))
    g.add(da.label("1.44", 30, C["aux"]).next_to(gp(0, R0 ** 2), LEFT, buff=0.12))
    g.add(ha.row("sphere", r"$r_0 = 1.2$", size=30).move_to([-1.15, 2.6, 0]))
    g.add(da.label("bumpy blob", 30).move_to([-4.75, 2.6, 0]))
    g.add(ha.row("blue:", r"$H < 0$", size=30, color=da.ManimColor("#2166AC")).move_to([-4.75, -2.5, 0]))
    g.add(ha.row("red:", r"$H > 0$", size=30, color=da.ManimColor("#B2182B")).move_to([-1.15, -2.5, 0]))
    g.add(ha.row("same scale in both panels", size=30, color=C["aux"]).move_to([-2.95, -1.9, 0]))
    return g


class Anim(da.Scene2D):
    def construct(self):
        s = ValueTracker(0.0)
        dyn = always_redraw(lambda: frame(T_END * s.get_value() ** 2))
        self.add(graph_static(), dyn)
        self.wait(1.5)
        self.play(s.animate.set_value(1.0), run_time=12.0, rate_func=linear)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=6.0)

"""Animation 0.5.4: geodesic triangle이 커질 때 내각의 합 − π = 넓이 (0.5절, Theorem 0.5.8(a), (0.5.2), Figure 0.5.3(a)).

unit sphere(K = 1) 위의 정삼각형 geodesic triangle. 중심 c(화면 정면을 향하는 점)에서 거리 ρ인 세 점을 vertex로 하고
세 변은 great circle의 호(주황)다. ρ를 0.12 → arccos(1/√3) ≈ 54.74° → arccos(1/3) ≈ 70.53°로 키운다.
  ρ = arccos(1/√3): 세 각이 모두 90°인 octant triangle(Figure 0.5.3(a)), 넓이 4π/8
  ρ = arccos(1/3):  세 각이 모두 120°, 정사면체의 한 면을 sphere에 투영한 삼각형, 넓이 4π/4 = π (Exercise 0.5.2)
내각은 vertex의 tangent plane에서 두 변의 방향 사이의 각으로 계산하고, 넓이는 따로 수치 적분한다.
자기검사: 여러 ρ에서 (α + β + γ) − π = 넓이(수치 적분, 그리고 solid angle 공식), 두 특별한 ρ에서 90°, 120°.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-5-4-triangle-excess.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, graticule, great_arc, polyline, sphere_curve, sphere_disk  # noqa: E402

C = da.C
FILL = ManimColor("#BCDDF1")
VIEW = View(elev=25, azim=-50, scale=3.0, center=np.array([-3.0, -0.35, 0]))
CEN, U1, U2 = VIEW.d, VIEW.right, VIEW.up
RHO_OCT, RHO_TET = np.arccos(1 / np.sqrt(3)), np.arccos(1 / 3)


def vertices(rho):
    b = np.deg2rad([90, 210, 330])
    return [np.cos(rho) * CEN + np.sin(rho) * (np.cos(t) * U1 + np.sin(t) * U2) for t in b]


def angle_at(A, B, Cc):
    tb, tc = B - (B @ A) * A, Cc - (Cc @ A) * A
    return np.arccos(np.clip(tb @ tc / np.linalg.norm(tb) / np.linalg.norm(tc), -1, 1))


def angles(rho):
    A, B, Cc = vertices(rho)
    return angle_at(A, B, Cc), angle_at(B, Cc, A), angle_at(Cc, A, B)


def area_numeric(rho, n=200):
    """X(u, v) = normalize(A + u(B − A) + v(C − A))의 넓이 요소 |X_u × X_v|를 (u, v) 삼각형 위에서 적분한다.

    (u, v) 삼각형을 n² 개의 작은 삼각형으로 나누고 각 무게중심에서 값을 잰다(넓이 공식을 쓰지 않는 독립 검산).
    """
    A, B, Cc = vertices(rho)
    h = 1.0 / n
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    up = (i + j <= n - 1)
    dn = (i + j <= n - 2)
    uv = np.vstack([np.stack([(i[up] + 1 / 3) * h, (j[up] + 1 / 3) * h], 1),
                    np.stack([(i[dn] + 2 / 3) * h, (j[dn] + 2 / 3) * h], 1)])
    Pm = A + np.outer(uv[:, 0], B - A) + np.outer(uv[:, 1], Cc - A)
    r = np.linalg.norm(Pm, axis=1)
    Xu = (B - A) / r[:, None] - Pm * ((Pm @ (B - A)) / r ** 3)[:, None]
    Xv = (Cc - A) / r[:, None] - Pm * ((Pm @ (Cc - A)) / r ** 3)[:, None]
    return np.sum(np.linalg.norm(np.cross(Xu, Xv), axis=1)) * h * h / 2


def area_solid(rho):
    A, B, Cc = vertices(rho)
    return 2 * np.arctan2(abs(A @ np.cross(B, Cc)), 1 + A @ B + B @ Cc + Cc @ A)


# ---------------------------------------------------------------- 자기검사
for rho in (0.12, 0.5, RHO_OCT, 1.0, RHO_TET):
    exc = sum(angles(rho)) - np.pi
    assert np.isclose(exc, area_solid(rho), atol=1e-10)
    assert np.isclose(exc, area_numeric(rho), rtol=1e-4), (rho, exc, area_numeric(rho))
assert np.allclose(angles(RHO_OCT), np.pi / 2) and np.isclose(area_solid(RHO_OCT), 4 * np.pi / 8)
assert np.allclose(angles(RHO_TET), 2 * np.pi / 3) and np.isclose(area_solid(RHO_TET), np.pi)
assert all(VIEW.depth(x) > 0.3 for x in vertices(RHO_TET))                 # 삼각형 전체가 앞쪽에 있다


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        rho = ValueTracker(0.12)
        self.add(sphere_disk(v, fill_opacity=0.25), graticule(v, n_lat=5, n_lon=12, color="#B8B8B8", width=1.0))

        def tri():
            A, B, Cc = vertices(rho.get_value())
            sides = [great_arc(A, B, 80), great_arc(B, Cc, 80), great_arc(Cc, A, 80)]
            bnd = np.vstack(sides)
            g = VGroup(Polygon(*v.p(bnd), stroke_width=0, fill_color=FILL, fill_opacity=1))
            for s in sides:
                g.add(polyline(v.p(s), C["accent"], 6))
            for X, Y, Z in ((A, B, Cc), (B, Cc, A), (Cc, A, B)):
                tb, tc = Y - (Y @ X) * X, Z - (Z @ X) * X
                tb, tc = tb / np.linalg.norm(tb), tc / np.linalg.norm(tc)
                ang = np.arccos(np.clip(tb @ tc, -1, 1))
                w = tc - (tc @ tb) * tb
                w /= np.linalg.norm(w)
                r = min(0.16, 0.35 * np.arccos(np.clip(X @ Y, -1, 1)))
                s = np.linspace(0, ang, 30)[:, None]
                arc = X + r * (np.cos(s) * tb + np.sin(s) * w)
                g.add(polyline(v.p(arc), C["main"], 3))
                g.add(Dot(v.p(X), radius=0.07, color=C["main"]))
            return g

        self.add(always_redraw(tri))
        X0 = 0.9
        title = da.label("Geodesic triangles on the unit sphere", 34).to_edge(UP, buff=0.3)
        eq = da.mtex(r"$\alpha + \beta + \gamma = \pi + \iint_{T}\;K\,dA$", 38).move_to([X0, 2.3, 0], aligned_edge=LEFT)
        r1 = always_redraw(lambda: da.label(f"angle sum = {np.rad2deg(sum(angles(rho.get_value()))):.1f}°", 32)
                           .move_to([X0, 1.25, 0], aligned_edge=LEFT))
        r2 = always_redraw(lambda: da.label(f"angle sum − π = {sum(angles(rho.get_value())) - np.pi:.3f}", 32)
                           .move_to([X0, 0.6, 0], aligned_edge=LEFT))
        r3 = always_redraw(lambda: da.label(f"area × K = {area_solid(rho.get_value()):.3f}", 32,
                                            color=ManimColor("#2A7AB0")).move_to([X0, -0.05, 0], aligned_edge=LEFT))
        self.add(title, eq, r1, r2, r3)
        self.wait(2.0)

        self.play(rho.animate.set_value(RHO_OCT), run_time=4.5, rate_func=smooth)
        n1 = VGroup(da.label("three right angles", 32),
                    da.mtex(r"$\mathrm{area} = 4\pi/8 = \pi/2$", 34)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        n1.move_to([X0, -1.3, 0], aligned_edge=LEFT)
        self.play(FadeIn(n1), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(n1), run_time=0.4)
        self.play(rho.animate.set_value(RHO_TET), run_time=3.5, rate_func=smooth)
        n2 = VGroup(da.label("three 120° angles", 32),
                    da.mtex(r"$\mathrm{area} = 4\pi/4 = \pi$", 34)).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        n2.move_to([X0, -1.3, 0], aligned_edge=LEFT)
        self.play(FadeIn(n2), run_time=0.6)
        self.wait(2.0)
        msg = VGroup(da.label("excess over π", 34), da.label("= area × K  (K = 1)", 34)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12).move_to([X0, -2.58, 0], aligned_edge=LEFT)
        assert msg.get_right()[0] < config.frame_width / 2 - 0.2 and msg.get_bottom()[1] > -3.0   # 가장자리 여백, control bar 위
        self.play(FadeIn(msg), run_time=0.6)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=8.0)

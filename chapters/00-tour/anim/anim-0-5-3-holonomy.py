"""Animation 0.5.3: circle of latitude를 따른 parallel transport와 holonomy = 둘러싼 넓이 (0.5절, Example 0.5.7, Figure 0.5.2(b)).

unit sphere(K = 1), colatitude θ₀인 circle of latitude를 경도가 증가하는 쪽으로 한 바퀴 돈다(Figure 0.5.2(b)와 같은 시점).
V₀ = e_θ(남쪽, 파랑)를 parallel transport한 V(φ)(주황, 0.33배)는 orthonormal frame (e_θ, e_φ)에서
V(φ) = cos(φ cos θ₀) e_θ − sin(φ cos θ₀) e_φ 이다(Christoffel symbols로 쓴 transport 방정식의 해. dgnum의 RK4로 확인).
한 바퀴 뒤 V(2π)는 바깥에서 보아 반시계 방향으로 2π(1 − cos θ₀) (mod 2π)만큼 돌아 있고, 이것은 극관(하늘색)의
넓이 × K와 같다.
  1부: θ₀ = π/3에서 천천히 한 바퀴(30°마다 흔적). 회전 π = 180°.
  2부: θ₀를 π/3 → π/12 → 0.49π → π/4로 바꾸며 흔적, V(2π), 극관, 두 수를 함께 갱신한다.
       θ₀ = π/4에서 105.4°(Exercise 0.5.3), 적도에 가까우면 360°에 가까워져 회전이 사라진다.
자기검사: dgnum.parallel_transport(dgsym의 Christoffel symbols)와 위 닫힌 식이 여러 θ₀에서 일치, 길이 보존,
회전각 = 2π(1 − cos θ₀) (mod 2π), 극관 넓이의 수치 적분 = 2π(1 − cos θ₀).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-5-3-holonomy.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgnum  # noqa: E402
import dgsym  # noqa: E402
from _helpers_b import View, arrow, polyline, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C
SC = 0.33
CAP = ManimColor("#BCDDF1")     # 하늘색(region)을 alpha 0.35로 흰 면에 칠한 색(불투명하게 그려 이음매를 없앤다)


def frame(theta, phi):
    et = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    ep = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return et, ep


def V(theta0, phi):
    """parallel transport된 V(φ) (ℝ³ 성분). V(0) = e_θ."""
    et, ep = frame(theta0, phi)
    return np.cos(phi * np.cos(theta0)) * et - np.sin(phi * np.cos(theta0)) * ep


def holonomy(theta0):
    """V(2π)가 V₀에서 바깥 N으로 보아 반시계 방향으로 돈 각, [0, 2π)."""
    v0, v1, n = V(theta0, 0.0), V(theta0, 2 * np.pi), sph(theta0, 0.0)
    return np.mod(np.arctan2(np.cross(v0, v1) @ n, v0 @ v1), 2 * np.pi)


# ---------------------------------------------------------------- 자기검사: RK4(dgnum)와 닫힌 식
ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
g = dgsym.induced_metric(ex["expr"], (th, ph), ex["positive"])
Gam = dgnum.numeric_christoffel(g, (th, ph), {rs: 1}, ex["positive"])
ts = np.linspace(0, 2 * np.pi, 1441)
for t0 in (np.pi / 12, np.pi / 4, np.pi / 3, 0.49 * np.pi):
    W = dgnum.parallel_transport(Gam, lambda t, t0=t0: np.array([t0, t]), lambda t: np.array([0.0, 1.0]),
                                 np.array([1.0, 0.0]), ts)
    for k in range(0, len(ts), 120):
        et, ep = frame(t0, ts[k])
        num = W[k, 0] * et + W[k, 1] * np.sin(t0) * ep          # ∂_θ = e_θ, ∂_φ = sin θ e_φ
        assert np.allclose(num, V(t0, ts[k]), atol=1e-7)
        assert np.isclose(np.linalg.norm(num), 1, atol=1e-8)
    want = np.mod(2 * np.pi * (1 - np.cos(t0)), 2 * np.pi)
    assert np.isclose(holonomy(t0), want, atol=1e-9)
    tt = np.linspace(0, t0, 2001)
    cap = 2 * np.pi * np.sum((np.sin(tt[1:]) + np.sin(tt[:-1])) / 2 * np.diff(tt))
    assert np.isclose(cap, 2 * np.pi * (1 - np.cos(t0)), atol=1e-6)
assert np.isclose(np.rad2deg(holonomy(np.pi / 3)), 180) and np.isclose(np.rad2deg(holonomy(np.pi / 4)), 105.44, atol=0.01)

VIEW = View(elev=58, azim=-30, scale=3.0, center=np.array([-2.9, -0.3, 0]))


def cap_mob(v, theta0, n=721):
    """극관 {θ ≤ θ₀} 가운데 보이는 부분(앞쪽 hemisphere와의 교집합)을 polygon 하나로 그린다.

    경계 = 위도원의 앞쪽 호 + 그 두 끝을 잇는 윤곽(limb)의 호 가운데 극관 안에 있는 쪽.
    """
    s = np.linspace(0, 2 * np.pi, n)[:-1]
    P = sph(theta0, s)
    dep = v.depth(P)
    if np.all(dep > 0):
        pts = v.p(P)
    else:
        k = int(np.argmax((dep > 0) & (np.roll(dep, 1) <= 0)))       # 뒤 → 앞으로 넘어가는 첫 index
        P, dep = np.roll(P, -k, axis=0), np.roll(dep, -k)
        j = int(np.argmax(dep <= 0))                                   # 앞쪽 구간 [0, j)
        a = P[0] + (P[-1] - P[0]) * dep[0] / (dep[0] - dep[-1])        # 두 교차점(선형 보간)
        b = P[j - 1] + (P[j] - P[j - 1]) * dep[j - 1] / (dep[j - 1] - dep[j])
        front = np.vstack([a, P[:j], b])
        ang = lambda X: np.arctan2(X @ v.up, X @ v.right)               # noqa: E731
        aa, ab = ang(a), ang(b)
        best = None
        for sgn in (1, -1):
            d = np.mod(sgn * (aa - ab), 2 * np.pi)
            arc = ab + sgn * np.linspace(0, d, 120)
            L = np.cos(arc)[:, None] * v.right + np.sin(arc)[:, None] * v.up
            if L[60, 2] > np.cos(theta0):
                best = L
        pts = v.p(np.vstack([front, best]))
    return Polygon(*pts, stroke_width=0, fill_color=CAP, fill_opacity=1)


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        t0 = ValueTracker(np.pi / 3)
        phi = ValueTracker(0.0)
        show_cap = ValueTracker(0.0)
        disk = sphere_disk(v, fill_opacity=0.25)
        self.add(disk)

        def scene_mob():
            a = t0.get_value()
            gg = VGroup()
            if show_cap.get_value() > 0.5:
                gg.add(cap_mob(v, a))
            s = np.linspace(0, 2 * np.pi, 241)
            gg.add(sphere_curve(v, sph(a, s), C["main"], 5, hidden="faint"))
            gg.add(Dot(v.p(np.array([0, 0, 1.0])), radius=0.06, color=C["main"]))
            # 흔적: 30°마다
            f = phi.get_value()
            for k in range(1, 12):
                pk = k * np.pi / 6
                if pk < f - 1e-6:
                    x = sph(a, pk)
                    op = 1.0 if v.depth(x) > 0 else 0.25
                    gg.add(arrow(v.p(x), v.p(x) + v.vec(SC * V(a, pk)), C["accent"], 4, 0.16, op))
            x0 = sph(a, 0.0)
            P0 = v.p(x0)
            gg.add(arrow(P0, P0 + v.vec(SC * V(a, 0.0)), C["tangent"], 6, 0.2))
            x = sph(a, f)
            if f > 1e-6:                     # φ = 0에서는 파란 V₀만 보이게 한다(주황이 덮지 않도록)
                gg.add(arrow(v.p(x), v.p(x) + v.vec(SC * V(a, f)), C["accent"], 7, 0.22))
            gg.add(Dot(v.p(x), radius=0.07, color=C["main"]))
            gg.add(Dot(P0, radius=0.07, color=C["main"]))
            return gg

        def hol_arc():
            a = t0.get_value()
            if phi.get_value() < 2 * np.pi - 1e-6:
                return VGroup()
            x0 = sph(a, 0.0)
            v0 = V(a, 0.0)
            w0 = np.cross(x0, v0)
            h = holonomy(a)
            if h < 1e-3:
                return VGroup()
            s = np.linspace(0, h, 60)[:, None]
            r = 0.55 * SC
            P = x0 + r * (np.cos(s) * v0 + np.sin(s) * w0)
            return polyline(v.p(P), C["main"], 4)

        mob = always_redraw(scene_mob)
        arcm = always_redraw(hol_arc)
        self.add(mob, arcm)
        lab0 = da.mtex(r"$V_0$", 38, color=C["tangent"])
        lab0.move_to(v.p(sph(np.pi / 3, 0.0)) + v.vec(SC * V(np.pi / 3, 0.0)) + np.array([0.35, -0.1, 0]))
        self.add(lab0)

        X0 = 0.9
        title = da.label("Parallel transport around a circle of latitude", 34).to_edge(UP, buff=0.3)
        self.add(title)
        r_t0 = always_redraw(lambda: da.label(f"θ₀ = {np.rad2deg(t0.get_value()):.1f}°", 36).move_to(
            [X0, 2.2, 0], aligned_edge=LEFT))
        r_phi = always_redraw(lambda: da.label(f"φ = {round(np.rad2deg(phi.get_value()))}°", 36).move_to(
            [X0, 1.5, 0], aligned_edge=LEFT))
        self.add(r_t0, r_phi)
        self.wait(1.5)

        # 1부: θ₀ = π/3에서 한 바퀴
        self.play(phi.animate.set_value(2 * np.pi), run_time=8.0, rate_func=linear)
        lab1 = da.mtex(r"$V(2\pi)$", 38, color=C["accent"])
        lab1.move_to(v.p(sph(np.pi / 3, 0.0)) + v.vec(SC * V(np.pi / 3, 2 * np.pi)) + np.array([-0.15, 0.32, 0]))
        rot = always_redraw(lambda: da.label(f"V(2π) turned {np.rad2deg(holonomy(t0.get_value())):.1f}°", 32).move_to([X0, 0.5, 0], aligned_edge=LEFT))
        self.play(FadeIn(lab1), FadeIn(rot), run_time=0.8)
        self.wait(2.0)

        # 극관과 넓이
        area = always_redraw(lambda: da.label(
            f"K × cap area = {np.rad2deg(2 * np.pi * (1 - np.cos(t0.get_value()))):.1f}°", 32,
            color=ManimColor("#2A7AB0")).move_to([X0, -0.15, 0], aligned_edge=LEFT))
        formula = da.mtex(r"$2\pi(1-\cos\theta_0)$", 36).move_to([X0, -0.95, 0], aligned_edge=LEFT)
        self.play(show_cap.animate.set_value(1.0), FadeIn(area), FadeIn(formula), run_time=0.8)
        self.wait(2.5)

        # 2부: θ₀를 바꾼다
        self.play(FadeOut(lab0), FadeOut(lab1), FadeOut(r_phi), run_time=0.5)
        self.play(t0.animate.set_value(np.pi / 12), run_time=3.0, rate_func=smooth)
        self.wait(1.5)
        self.play(t0.animate.set_value(0.49 * np.pi), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        self.play(t0.animate.set_value(np.pi / 4), run_time=2.5, rate_func=smooth)
        msg = VGroup(da.label("rotation angle", 34), da.label("= enclosed curvature", 34)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12).move_to([X0, -2.4, 0], aligned_edge=LEFT)
        self.play(FadeIn(msg), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.5)

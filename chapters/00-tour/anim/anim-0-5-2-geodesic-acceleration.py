"""Animation 0.5.2: circle of latitude와 great circle을 따라 달릴 때의 가속도 (0.5절, Example 0.5.5, Figure 0.5.2(a)).

unit sphere, 북위 45°(colatitude θ₀ = π/4)의 두 점 p = x(π/4, −π/3), q = x(π/4, π/3) (Figure 0.5.2(a)와 같다).
점이 단위 속력으로 (1) circle of latitude의 호(검정, 길이 sin θ₀ · 2π/3 ≈ 1.481), (2) great circle의 호(주황, 길이
arccos(1/4) ≈ 1.318)를 따라 간다. 움직이는 점에서 가속도 γ''(검정)를 sphere에 수직인 부분(주홍)과 tangent plane
안의 부분(파랑)으로 나누어 그린다(0.6배). circle of latitude의 tangent 부분은 크기 cot θ₀ = 1로 북쪽을 향하고
(옆으로 꺾인다), great circle의 tangent 부분은 0이다(Definition 0.5.4: geodesic).
자기검사: 두 호의 길이, 수치 미분한 가속도의 tangent 성분(위도원 1, great circle 0), 둘 다 normal 성분 크기 1.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-5-2-geodesic-acceleration.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, graticule, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C
T0 = np.pi / 4
p, q = sph(T0, -np.pi / 3), sph(T0, np.pi / 3)
L_LAT = np.sin(T0) * 2 * np.pi / 3
L_GC = np.arccos(p @ q)


def lat_pt(s):
    """circle of latitude, 호의 길이 s (0 ≤ s ≤ L_LAT)"""
    return sph(T0, -np.pi / 3 + s / np.sin(T0))


def gc_pt(s):
    """great circle, 호의 길이 s (0 ≤ s ≤ L_GC)"""
    w = (q - np.cos(L_GC) * p) / np.sin(L_GC)
    return np.cos(s) * p + np.sin(s) * w


def accel(f, s, h=1e-4):
    x = f(s)
    a = (f(s + h) - 2 * x + f(s - h)) / h ** 2
    an = (a @ x) * x
    return x, a, an, a - an


# ---------------------------------------------------------------- 자기검사
assert np.isclose(L_LAT, 1.481, atol=1e-3) and np.isclose(L_GC, np.arccos(0.25)) and np.isclose(L_GC, 1.318, atol=1e-3)
assert np.allclose(gc_pt(L_GC), q) and np.allclose(lat_pt(L_LAT), q)
for s in np.linspace(0.05, 1.25, 7):
    x, a, an, at = accel(lat_pt, s)
    assert np.isclose(np.linalg.norm(at), 1 / np.tan(T0), atol=1e-5)          # 옆으로 꺾임 = cot θ₀ = 1
    assert at[2] > 0                                                           # 북쪽(극 쪽)을 향한다
    assert np.isclose(np.linalg.norm(an), 1, atol=1e-5)
    x, a, an, at = accel(gc_pt, s)
    assert np.linalg.norm(at) < 1e-5 and np.isclose(np.linalg.norm(an), 1, atol=1e-5)   # geodesic

SC = 0.6
VIEW = View(elev=22, azim=0, scale=3.0, center=np.array([-2.6, -0.35, 0]))


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        base = VGroup(sphere_disk(v, fill_opacity=0.3), graticule(v, n_lat=5, n_lon=12, color="#B8B8B8", width=1.0))
        ss = np.linspace(0, 1, 200)
        lat = sphere_curve(v, np.array([lat_pt(s * L_LAT) for s in ss]), C["main"], 5)
        gc = sphere_curve(v, np.array([gc_pt(s * L_GC) for s in ss]), C["accent"], 5)
        dots = VGroup(Dot(v.p(p), radius=0.08, color=C["main"]), Dot(v.p(q), radius=0.08, color=C["main"]))
        labs = VGroup(da.mtex(r"$p$", 38).next_to(v.p(p), DOWN + LEFT, buff=0.08),
                      da.mtex(r"$q$", 38).next_to(v.p(q), DOWN + RIGHT, buff=0.08))
        title = da.label("Which path from p to q is straight?", 36).to_edge(UP, buff=0.3)
        self.add(base, lat, gc, dots, labs, title)

        X0 = 1.0
        leg = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=C["main"], stroke_width=5), da.label("acceleration", 30)).arrange(RIGHT),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=C["normal"], stroke_width=5),
                   da.label("normal part", 30)).arrange(RIGHT),
            VGroup(Line(ORIGIN, RIGHT * 0.6, color=C["tangent"], stroke_width=5),
                   da.label("sideways part", 30)).arrange(RIGHT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([X0, 2.2, 0], aligned_edge=LEFT)
        self.add(leg)

        s_t = ValueTracker(0.0)
        which = {"f": lat_pt, "L": L_LAT}

        def runner():
            f, L = which["f"], which["L"]
            s = np.clip(s_t.get_value(), 1e-3, 1 - 1e-3) * L
            x, a, an, at = accel(f, s)
            P = v.p(x)
            # 가속도(검정)는 두 성분 위에 가늘게 그린다. 위도원의 가운데에서는 세 화살표가 화면에서 한 직선에 놓이고
            # (meridian 평면이 시선 방향을 포함한다), great circle에서는 가속도 = normal 성분이라 겹친다.
            g = VGroup(arrow(P, P + v.vec(SC * an), C["normal"], 8, 0.24),
                       arrow(P, P + v.vec(SC * at), C["tangent"], 8, 0.24),
                       arrow(P, P + v.vec(SC * a), C["main"], 3.5, 0.18),
                       Dot(P, radius=0.1, color=C["main"]))
            return g

        def sideways():
            f, L = which["f"], which["L"]
            s = np.clip(s_t.get_value(), 1e-3, 1 - 1e-3) * L
            return np.linalg.norm(accel(f, s)[3])

        run = always_redraw(runner)

        # (1) circle of latitude
        h1 = da.label("circle of latitude", 34).move_to([X0, 0.45, 0], aligned_edge=LEFT)
        r1 = always_redraw(lambda: da.label(f"sideways part = {sideways():.2f}", 32, color=C["tangent"]).move_to(
            [X0, -0.15, 0], aligned_edge=LEFT))
        self.wait(1.5)
        self.play(FadeIn(h1), FadeIn(run), FadeIn(r1), lat.animate.set_stroke(width=7), run_time=0.8)
        self.play(s_t.animate.set_value(1.0), run_time=5.5, rate_func=linear)
        self.wait(1.0)
        res1 = VGroup(da.label("bends: not a geodesic", 30),
                      da.label(f"length {L_LAT:.3f}", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        res1.next_to(h1, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeOut(r1), FadeIn(res1), run_time=0.6)
        self.wait(1.5)

        # (2) great circle
        self.remove(run)
        which.update(f=gc_pt, L=L_GC)
        s_t.set_value(0.0)
        run = always_redraw(runner)
        h2 = da.label("great circle", 34, color=C["accent"]).move_to([X0, -1.75, 0], aligned_edge=LEFT)
        r2 = always_redraw(lambda: da.label(f"sideways part = {sideways():.2f}", 32, color=C["tangent"]).move_to(
            [X0, -2.35, 0], aligned_edge=LEFT))
        self.play(FadeIn(h2), FadeIn(run), FadeIn(r2), lat.animate.set_stroke(width=5),
                  gc.animate.set_stroke(width=7), run_time=0.8)
        self.play(s_t.animate.set_value(1.0), run_time=5.0, rate_func=linear)
        self.wait(1.0)
        res2 = VGroup(da.label("never bends: a geodesic", 30),
                      da.label(f"length {L_GC:.3f} (shorter)", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        res2.next_to(h2, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeOut(r2), FadeIn(res2), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=5.5)

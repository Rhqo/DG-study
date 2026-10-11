"""Animation 0.8.2: unit sphere의 exponential map exp_N이 tangent plane의 원을 circle of latitude로 보낸다 (0.8절, (0.8.5), Figure 0.8.2).

exp_N(v) = cos|v| N + sin|v| v/|v| (unit sphere, 북극 N). 왼쪽은 T_N S²(하늘색 disk |v| ≤ π), 오른쪽은 S²(Figure 0.8.2와
같은 시점 elev 18°, azim −60°). 회색 반직선 8개는 meridian(N에서 나가는 geodesic)으로 간다.
ρ를 0 → π/2 → π로 키우며 원 |v| = ρ(주황, 왼쪽)와 그 상인 geodesic circle(주황, 오른쪽, colatitude ρ인 circle of latitude)을
함께 그린다. 반직선 위의 굵은 검은 부분은 길이 ρ의 geodesic 조각이다. 파란 v는 Figure 0.8.2의 v(길이 2, 각 −30°)이고,
ρ ≥ 2가 되면 exp_N(v)(파란 점)가 나타난다.
둘레: T_N에서 2πρ, S²에서 2π sin ρ. ρ = π에서 원 전체가 남극 S 한 점으로 간다(cut point).
자기검사: exp_N(ρ(cos a, sin a))는 N에서의 거리가 ρ이고 z = cos ρ, circle of latitude의 둘레(수치) = 2π sin ρ,
|v| = π는 −N, geodesic t ↦ exp_N(tv)의 길이 = |v|.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-2-exp-sphere.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, polyline, sphere_curve, sphere_disk  # noqa: E402

C = da.C
N = np.array([0.0, 0.0, 1.0])


def exp_N(v2):
    v2 = np.atleast_2d(v2)
    r = np.linalg.norm(v2, axis=1)
    u = np.where(r[:, None] > 1e-14, v2 / np.maximum(r, 1e-14)[:, None], 0)
    return np.column_stack([np.sin(r) * u[:, 0], np.sin(r) * u[:, 1], np.cos(r)])


ANGS = np.deg2rad(np.arange(0, 360, 45))
VV = 2.0 * np.array([np.cos(np.deg2rad(-30)), np.sin(np.deg2rad(-30))])
S_ = np.linspace(0, 2 * np.pi, 241)

# ---------------------------------------------------------------- 자기검사
for rho in (0.3, np.pi / 4, np.pi / 2, 2.0, 3 * np.pi / 4):
    Q = exp_N(rho * np.column_stack([np.cos(S_), np.sin(S_)]))
    assert np.allclose(np.arccos(np.clip(Q @ N, -1, 1)), rho) and np.allclose(Q[:, 2], np.cos(rho))
    assert np.isclose(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)), 2 * np.pi * np.sin(rho), rtol=1e-4)
assert np.allclose(exp_N(np.pi * np.column_stack([np.cos(S_), np.sin(S_)])), -N)
geo = exp_N(np.outer(np.linspace(0, 1, 400), VV))
assert np.isclose(np.sum(np.linalg.norm(np.diff(geo, axis=0), axis=1)), 2.0, rtol=1e-5)

# 아래 90px(멈춤 상태의 control bar)를 비우려고 disk와 sphere를 줄이고 올렸다(§12.8)
TC, TS = np.array([-3.9, 0.7, 0]), 0.6
VIEW = View(elev=18, azim=-60, scale=1.85, center=np.array([3.55, 0.7, 0]))


def T(v2):
    v2 = np.atleast_2d(v2)
    return TC + TS * np.column_stack([v2[:, 0], v2[:, 1], np.zeros(len(v2))])


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        rho = ValueTracker(0.0)
        disk = Circle(radius=np.pi * TS, stroke_color=C["main"], stroke_width=3, fill_color=C["region"],
                      fill_opacity=0.18).move_to(TC)
        rays = VGroup(*[Line(TC, T(np.pi * np.array([np.cos(a), np.sin(a)]))[0], stroke_color=C["aux"],
                             stroke_width=2) for a in ANGS])
        sph = sphere_disk(v, fill_opacity=0.3)
        mer = VGroup(*[sphere_curve(v, exp_N(np.outer(np.linspace(0, np.pi, 100), [np.cos(a), np.sin(a)])),
                                    C["aux"], 2, hidden=None) for a in ANGS])
        labs = VGroup(da.mtex(r"$T_N S^2$", 36).move_to(TC + np.array([-1.9, 2.0, 0])),
                      da.mtex(r"$|v| = \pi$", 32).move_to(TC + np.array([1.95, 1.75, 0])),
                      da.mtex(r"$S^2$", 36).move_to(v.center + np.array([1.7, 1.75, 0])),
                      Dot(v.p(N), radius=0.07), da.mtex(r"$N$", 34).next_to(v.p(N), UP + LEFT, buff=0.05),
                      Dot(v.p(-N), radius=0.07), da.mtex(r"$S$", 34).next_to(v.p(-N), DOWN, buff=0.08))
        varrow = arrow(TC, T(VV)[0], C["tangent"], 6, 0.22)
        vlab = da.mtex(r"$v$", 36, color=C["tangent"]).next_to(T(VV)[0], DOWN, buff=0.1)
        arr = Arrow(TC + np.array([2.25, 0.9, 0]), v.center + np.array([-2.15, 0.9, 0]), buff=0, stroke_width=3,
                    color=C["main"], tip_length=0.2)
        alab = da.mtex(r"$\exp_N$", 36).next_to(arr, UP, buff=0.08)
        self.add(disk, rays, sph, mer, labs, varrow, vlab, arr, alab)

        def grow():
            r = rho.get_value()
            g = VGroup()
            if r > 1e-3:
                circ = r * np.column_stack([np.cos(S_), np.sin(S_)])
                g.add(polyline(T(circ), C["accent"], 5))
                g.add(sphere_curve(v, exp_N(circ), C["accent"], 5, hidden="faint", hidden_opacity=0.35))
                for a in ANGS:
                    d = np.array([np.cos(a), np.sin(a)])
                    g.add(Line(TC, T(r * d)[0], stroke_color=C["main"], stroke_width=4))
                    g.add(sphere_curve(v, exp_N(np.outer(np.linspace(0, r, 60), d)), C["main"], 4,
                                       hidden="faint", hidden_opacity=0.35))
            if r >= 2.0:
                q = exp_N(VV)[0]
                g.add(Dot(v.p(q), radius=0.09, color=C["tangent"]))
                g.add(Dot(T(VV)[0], radius=0.07, color=C["tangent"]))
            return g

        self.add(always_redraw(grow))

        def nums():
            r = rho.get_value()
            g = VGroup(da.label(f"ρ = |v| = {r:.2f}", 32),
                       da.label(f"circle in the tangent plane: 2πρ = {2 * np.pi * r:.2f}", 32),
                       da.label(f"its image on S²: 2π sin ρ = {2 * np.pi * np.sin(r):.2f}", 32, color=C["accent"]))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_edge(DOWN, buff=1.05).to_edge(LEFT, buff=0.35)

        self.add(always_redraw(nums))
        title = da.label("The exponential map at the north pole", 34).to_edge(UP, buff=0.25)
        self.add(title)
        self.wait(1.5)
        self.play(rho.animate.set_value(np.pi / 2), run_time=5.0, rate_func=smooth)
        self.wait(1.5)
        self.play(rho.animate.set_value(np.pi), run_time=6.0, rate_func=smooth)
        note = VGroup(da.label("|v| = π: all go to S,", 30),
                      da.label("the cut point of N", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        note.to_edge(DOWN, buff=1.05).to_edge(RIGHT, buff=0.3)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.5)

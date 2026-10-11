"""Animation 0.6.3: tangent vector = 곡선의 velocity, 그리고 chart를 바꾸면 성분이 Jacobian으로 바뀐다 (0.6절, (0.6.4), Definition 0.6.5).

가운데: unit sphere S²와 점 p = x(60°, 30°)(colatitude, longitude; Figure 0.6.2의 점 x).
p를 지나는 두 곡선을 단위 속력으로 따라간다.
  γ₁: circle of latitude(검정), γ₁(t) = x(60°, 30° + t/sin 60°)
  γ₂: p에서 γ₁에 접하는 great circle(주황), γ₂(t) = cos t · p + sin t · e_φ
둘 다 γ'(0) = e_φ(동쪽 단위벡터)다. 왼쪽은 chart σ, 오른쪽은 chart σ̃의 좌표 평면이고, 같은 축척(눈금 1)으로 그렸다.
움직이는 점의 velocity(파랑)를 세 곳에 실제 길이로 그린다. p를 지날 때 두 곡선의 velocity 성분은 각 chart에서 같다:
  σ에서 (σ∘γ)'(0) = (−1, √3),  σ̃에서 (σ̃∘γ)'(0) = (−1/3, √3/3),  그리고 D(u/|u|²)(σ(p)) · (−1, √3) = (−1/3, √3/3) ((0.6.4)).
자기검사: 수치 미분한 (σ∘γ_k)'(0), (σ̃∘γ_k)'(0)이 k = 1, 2에서 같고 위 값과 일치, transition의 Jacobian(sympy)과 (0.6.4).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-3-velocity-charts.py
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgsym  # noqa: E402
from _helpers_b import View, arrow, graticule, polyline, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C
st = dgsym.stereographic(2)
X, U = st["coords"], st["chart_coords"]
sigma = sp.lambdify(X, list(st["sigma"]), "numpy")
sigma_s = sp.lambdify(X, list(st["sigma_south"]), "numpy")
Jtr = sp.lambdify(U, st["transition"].jacobian(list(U)), "numpy")

TH0, PH0 = np.deg2rad(60), np.deg2rad(30)
P = sph(TH0, PH0)
EPH = np.array([-np.sin(PH0), np.cos(PH0), 0.0])


def g1(t):
    return sph(TH0, PH0 + t / np.sin(TH0))


def g2(t):
    return np.cos(t) * P + np.sin(t) * EPH


def vel(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2 * h)


def chart(name):
    m = sigma if name == "s" else sigma_s
    return lambda x: np.array(m(*x), float)


# ---------------------------------------------------------------- 자기검사
for name, want in (("s", np.array([-1.0, np.sqrt(3)])), ("t", np.array([-1 / 3, np.sqrt(3) / 3]))):
    ch = chart(name)
    for f in (g1, g2):
        assert np.allclose(vel(f, 0.0), EPH, atol=1e-8)
        assert np.allclose(vel(lambda t: ch(f(t)), 0.0), want, atol=1e-6)
u0 = chart("s")(P)
assert np.allclose(u0, [1.5, np.sqrt(3) / 2])
assert np.allclose(np.array(Jtr(*u0), float) @ np.array([-1.0, np.sqrt(3)]), [-1 / 3, np.sqrt(3) / 3])   # (0.6.4)

SCH = 0.7       # chart 평면의 단위 길이. 아래 숫자 줄이 poster의 control bar(아래 12%) 위에 오도록 줄였다
CL, CR = np.array([-4.75, -0.1, 0]), np.array([4.75, -0.1, 0])
HALF = 2.6
VIEW = View(elev=22, azim=np.rad2deg(PH0) - 10, scale=1.55, center=np.array([0, -0.25, 0]))
TS = np.linspace(-1.0, 1.0, 161)


class Anim(da.Scene2D):
    def chart_panel(self, c, title):
        g = VGroup(Square(side_length=2 * HALF * SCH, stroke_color=C["main"], stroke_width=1.5, fill_color=C["region"],
                          fill_opacity=0.12).move_to(c))
        g.add(Line(c + LEFT * HALF * SCH, c + RIGHT * HALF * SCH, stroke_color=C["aux"], stroke_width=1.2))
        g.add(Line(c + DOWN * HALF * SCH, c + UP * HALF * SCH, stroke_color=C["aux"], stroke_width=1.2))
        for k in (-2, -1, 1, 2):
            g.add(Line(c + np.array([k * SCH, -0.06, 0]), c + np.array([k * SCH, 0.06, 0]), stroke_color=C["aux"],
                       stroke_width=1.5))
            g.add(Line(c + np.array([-0.06, k * SCH, 0]), c + np.array([0.06, k * SCH, 0]), stroke_color=C["aux"],
                       stroke_width=1.5))
        g.add(DashedVMobject(Circle(radius=SCH, stroke_color=C["main"], stroke_width=1.5).move_to(c), num_dashes=24))
        g.add(da.mtex(title, 36).next_to(c + UP * HALF * SCH, UP, buff=0.15))
        return g

    def construct(self):
        v = VIEW
        self.add(self.chart_panel(CL, r"$\mathrm{chart}\ \sigma$"), self.chart_panel(CR, r"$\mathrm{chart}\ \tilde\sigma$"))
        self.add(sphere_disk(v, fill_opacity=0.3), graticule(v, n_lat=5, n_lon=12, color="#B8B8B8", width=1.0))
        self.add(da.mtex(r"$S^2$", 36).next_to(v.center + UP * v.scale, UP, buff=0.15))
        cs, ct = chart("s"), chart("t")

        def to_c(c, u):
            return c + SCH * np.array([u[0], u[1], 0])

        curves = VGroup()
        for f, col in ((g1, C["main"]), (g2, C["accent"])):
            pts = np.array([f(t) for t in TS])
            curves.add(sphere_curve(v, pts, col, 4))
            curves.add(polyline([to_c(CL, cs(x)) for x in pts], col, 4))
            curves.add(polyline([to_c(CR, ct(x)) for x in pts], col, 4))
        pdots = VGroup(Dot(v.p(P), radius=0.07), Dot(to_c(CL, cs(P)), radius=0.07), Dot(to_c(CR, ct(P)), radius=0.07))
        # 라벨은 곡선이 지나가지 않는 쪽(위·오른쪽 위)에 둔다. 아래쪽은 두 곡선이 지나간다
        plabs = VGroup(da.mtex(r"$p$", 34).next_to(v.p(P), UP, buff=0.14),
                       da.mtex(r"$\sigma(p)$", 32).next_to(to_c(CL, cs(P)), UP + RIGHT, buff=0.05),
                       da.mtex(r"$\tilde\sigma(p)$", 32).next_to(to_c(CR, ct(P)), UP + RIGHT, buff=0.05))
        self.add(curves, pdots, plabs)
        title = da.label("A tangent vector is a velocity", 34).to_edge(UP, buff=0.25)
        self.add(title)

        t = ValueTracker(-1.0)
        which = {"f": g1, "col": C["main"]}

        def runner():
            f = which["f"]
            tt = t.get_value()
            x, w = f(tt), vel(f, tt)
            g = VGroup()
            g.add(arrow(v.p(x), v.p(x) + v.vec(w), C["tangent"], 6, 0.2), Dot(v.p(x), radius=0.09, color=which["col"]))
            for c, ch in ((CL, cs), (CR, ct)):
                q = ch(x)
                dq = vel(lambda s: ch(f(s)), tt)
                g.add(arrow(to_c(c, q), to_c(c, q + dq), C["tangent"], 6, 0.2),
                      Dot(to_c(c, q), radius=0.09, color=which["col"]))
            return g

        run = always_redraw(runner)
        self.add(run)
        self.wait(1.0)
        self.play(t.animate.set_value(0.0), run_time=3.0, rate_func=linear)
        comp = VGroup(da.label("σ: (−1.00, 1.73)", 32, color=C["tangent"]),
                      da.label("σ̃: (−0.33, 0.58)", 32, color=C["tangent"])).arrange(RIGHT, buff=2.2)
        comp.move_to([0, -2.25, 0])
        h1 = da.label("velocity at p in each chart", 30).next_to(comp, DOWN, buff=0.12)
        self.play(FadeIn(comp), FadeIn(h1), run_time=0.6)
        self.wait(2.0)
        self.play(t.animate.set_value(1.0), run_time=2.5, rate_func=linear)
        # 두 번째 곡선
        ghost = VGroup(arrow(v.p(P), v.p(P) + v.vec(EPH), C["tangent"], 6, 0.2, 0.35),
                       arrow(to_c(CL, cs(P)), to_c(CL, cs(P) + np.array([-1, np.sqrt(3)])), C["tangent"], 6, 0.2, 0.35),
                       arrow(to_c(CR, ct(P)), to_c(CR, ct(P) + np.array([-1, np.sqrt(3)]) / 3), C["tangent"], 6, 0.2,
                             0.35))
        self.remove(run)
        which.update(f=g2, col=C["accent"])
        t.set_value(-1.0)
        run = always_redraw(runner)
        self.add(ghost, run)
        self.wait(0.6)
        self.play(t.animate.set_value(0.0), run_time=3.0, rate_func=linear)
        h2 = da.label("another curve, same velocity at p: the same tangent vector", 30).next_to(comp, DOWN, buff=0.12)
        self.play(FadeOut(h1), FadeIn(h2), run_time=0.6)
        self.wait(2.0)
        rel = da.mtex(r"$D(u/|u|^2)\,(-1.00,\ 1.73) = (-0.33,\ 0.58)$", 36).next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(rel), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=5.5)

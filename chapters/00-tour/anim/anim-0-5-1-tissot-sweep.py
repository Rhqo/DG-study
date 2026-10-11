"""Animation 0.5.1: 같은 작은 원을 위도에 따라 옮기면 Mercator와 Lambert 지도에서 어떻게 변하는가 (0.5절, Figure 0.5.1).

unit sphere의 standard parametrization x(θ, φ) (dgsym.EXAMPLES["sphere"], r = 1).
Mercator ψ_M(x(θ, φ)) = (φ, log cot(θ/2)), Lambert(Archimedes) ψ_L(x(θ, φ)) = (φ, cos θ). Figure 0.5.1과 같은 식이다.
sphere 위의 반지름 ε = 0.2인 작은 원(Figure 0.5.1의 0.15보다 크게 그렸다)을 dψ로 옮긴 Tissot indicatrix를
위도 0° → 60° → 70° → 0° → −60°로 움직인다. 반축은 orthonormal frame e_θ, e_φ에서 dψ의 singular value다.
  Mercator: a = b = 1/sin θ (원 유지, 넓이 1/sin²θ배)
  Lambert:  a = 1/sin θ (가로), b = sin θ (세로) (넓이 1배)
자기검사: sympy로 dψ를 미분해 위 반축과 conformal·넓이 보존을 확인한다. 위도 60°에서 Mercator a = b = 2, Lambert a = 2, b = 1/2.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-5-1-tissot-sweep.py
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgsym  # noqa: E402
from _helpers_b import View, graticule, polyline, sphere_curve, sphere_disk, sph  # noqa: E402

C = da.C
EPS = 0.2

# ---------------------------------------------------------------- 계산과 자기검사 (Figure 0.5.1과 같은 식)
ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
E, F, G = (e.subs(rs, 1) for e in dgsym.first_ff(ex["expr"], th, ph, ex["positive"]))
maps = {"mercator": sp.Matrix([ph, sp.log(sp.cot(th / 2))]), "lambert": sp.Matrix([ph, sp.cos(th)])}
frame_A = {}
for name, psi in maps.items():
    J = psi.jacobian([th, ph])
    A = J * sp.diag(1 / sp.sqrt(E), 1 / sp.sqrt(G))
    A = sp.simplify(A.subs({sp.sqrt(sp.sin(th) ** 2): sp.sin(th), sp.Abs(sp.sin(th)): sp.sin(th)}))
    frame_A[name] = sp.lambdify(th, A, "numpy")
    if name == "mercator":
        assert sp.simplify(A.T * A - sp.eye(2) / sp.sin(th) ** 2) == sp.zeros(2, 2)          # conformal
    else:
        assert sp.simplify(sp.Abs(A.det()) - 1) == 0                                         # 넓이 보존


def semi_axes(name, theta):
    """(가로 반축, 세로 반축) / ε. A의 열은 e_θ, e_φ의 상이고, 화면은 (가로 φ, 세로 y)이다."""
    A = np.array(frame_A[name](theta), float)
    return abs(A[0, 1]), abs(A[1, 0])


for lat in (0, 30, 60, 70):
    t0 = np.deg2rad(90 - lat)
    sv = np.linalg.svd(np.array(frame_A["mercator"](t0), float), compute_uv=False)
    assert np.allclose(sv, 1 / np.sin(t0))
    a, b = semi_axes("lambert", t0)
    assert np.isclose(a, 1 / np.sin(t0)) and np.isclose(b, np.sin(t0)) and np.isclose(a * b, 1)
assert np.allclose(semi_axes("mercator", np.deg2rad(30)), (2, 2))
assert np.allclose(semi_axes("lambert", np.deg2rad(30)), (2, 0.5))

# ---------------------------------------------------------------- 배치
W = 6.0                       # 지도의 가로 (2π rad)
S = W / (2 * np.pi)           # rad → 화면 단위
LAT_CUT = np.deg2rad(80)      # Mercator를 ±80°에서 자른다
YM = np.log(1 / np.tan((np.pi / 2 - LAT_CUT) / 2))
M0 = np.array([-3.45, 0.8, 0])           # Mercator 지도 중심. 아래 숫자 줄이 poster의 control bar(아래 12%) 위에 오도록 올렸다
L0 = np.array([3.45, 2.0, 0])            # Lambert 지도 중심
PHI0 = 7 * np.pi / 6                       # 움직이는 원의 경도. 끝에 까는 indicatrix 격자(k·π/3 + π/6)의 한 칸과 겹친다


def to_map(name, theta, phi):
    y = np.log(1 / np.tan(theta / 2)) if name == "mercator" else np.cos(theta)
    c = M0 if name == "mercator" else L0
    return c + S * np.array([phi - np.pi, y, 0.0])


assert to_map("mercator", np.deg2rad(20), PHI0)[1] + S * EPS * semi_axes("mercator", np.deg2rad(20))[1] < M0[1] + S * YM
assert any(np.isclose(PHI0, k * np.pi / 3 + np.pi / 6) for k in range(6))     # 끝 장면에서 격자의 한 칸과 정확히 겹친다


class Anim(da.Scene2D):
    def map_frame(self, name):
        c = M0 if name == "mercator" else L0
        ymax = YM if name == "mercator" else 1.0
        g = VGroup()
        g.add(Rectangle(width=W, height=2 * S * ymax, stroke_color=C["main"], stroke_width=2).move_to(c))
        for lat in (-60, -30, 0, 30, 60):
            t = np.deg2rad(90 - lat)
            g.add(Line(to_map(name, t, 0), to_map(name, t, 2 * np.pi), stroke_color="#A0A0A0", stroke_width=1.5))
        tmin = np.pi / 2 - LAT_CUT if name == "mercator" else 1e-4
        tt = np.linspace(tmin, np.pi - tmin, 60)
        for k in range(1, 12):
            g.add(polyline([to_map(name, t, k * np.pi / 6) for t in tt], "#A0A0A0", 1.5))
        return g

    def tissot(self, name, theta, phi, opacity=0.75, stroke=2.5):
        a, b = semi_axes(name, theta)
        return Ellipse(width=2 * S * EPS * a, height=2 * S * EPS * b, stroke_color=C["tangent"], stroke_width=stroke,
                       fill_color=C["region"], fill_opacity=opacity).move_to(to_map(name, theta, phi))

    def construct(self):
        lat = ValueTracker(0.0)

        def theta():
            return np.deg2rad(90 - lat.get_value())

        tM = da.label("Mercator: conformal", 34).next_to(M0 + UP * S * YM, UP, buff=0.15)
        tL = da.label("Lambert: equal-area", 34).next_to(L0 + UP * S, UP, buff=0.15)
        frames = VGroup(self.map_frame("mercator"), self.map_frame("lambert"))

        circM = always_redraw(lambda: self.tissot("mercator", theta(), PHI0))
        circL = always_redraw(lambda: self.tissot("lambert", theta(), PHI0))

        fM = da.mtex(r"$a = b = 1/\sin\theta$", 34).move_to(M0 + DOWN * (S * YM + 0.45))
        fL = da.mtex(r"$a = 1/\sin\theta,\ \ b = \sin\theta$", 34).move_to(L0 + DOWN * (S + 0.5))

        def nums(name):
            def make():
                a, b = semi_axes(name, theta())
                return da.label(f"a = {a:.2f}  b = {b:.2f}  area ×{a * b:.2f}", 32).move_to(
                    (fM if name == "mercator" else fL).get_center() + DOWN * 0.58)
            return always_redraw(make)

        nM, nL = nums("mercator"), nums("lambert")

        # 작은 globe: 같은 sphere 위 원이 위도를 따라 움직인다
        view = View(elev=12, azim=np.rad2deg(PHI0), scale=1.05, center=np.array([2.3, -1.9, 0]))
        globe = VGroup(sphere_disk(view), graticule(view, n_lat=5, n_lon=12, hidden=None))

        def small_circle():
            t0 = theta()
            p = sph(t0, PHI0)
            et = np.array([np.cos(t0) * np.cos(PHI0), np.cos(t0) * np.sin(PHI0), -np.sin(t0)])
            ep = np.array([-np.sin(PHI0), np.cos(PHI0), 0.0])
            s = np.linspace(0, 2 * np.pi, 60)[:, None]
            P = np.cos(EPS) * p + np.sin(EPS) * (np.cos(s) * et + np.sin(s) * ep)
            m = Polygon(*view.p(P), stroke_color=C["tangent"], stroke_width=2.5, fill_color=C["region"],
                        fill_opacity=0.75)
            return m

        circS = always_redraw(small_circle)
        latlab = always_redraw(lambda: da.label(f"latitude {lat.get_value():.0f}°".replace("-", "−"), 34).next_to(
            view.center + RIGHT * view.scale, RIGHT, buff=0.35))

        self.add(frames, tM, tL, globe, circS, circM, circL, fM, fL, nM, nL, latlab)
        self.wait(2.0)
        self.play(lat.animate.set_value(60), run_time=4.0, rate_func=smooth)
        self.wait(2.5)
        self.play(lat.animate.set_value(70), run_time=2.0, rate_func=smooth)
        self.wait(1.8)
        self.play(lat.animate.set_value(0), run_time=3.0, rate_func=smooth)
        self.play(lat.animate.set_value(-60), run_time=3.0, rate_func=smooth)
        self.wait(1.5)

        # 끝: Figure 0.5.1처럼 여러 위도의 indicatrix를 한꺼번에 깔고, 결론을 쓴다
        grid = VGroup()
        for name in ("mercator", "lambert"):
            for la in (-60, -30, 0, 30, 60):
                for k in range(6):
                    grid.add(self.tissot(name, np.deg2rad(90 - la), k * np.pi / 3 + np.pi / 6, 0.45, 1.8))
        msg = da.label("conformal or equal-area, never both", 36).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(grid), FadeIn(msg), run_time=1.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.5)

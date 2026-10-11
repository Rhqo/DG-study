"""Animation 0.6.4: vector field의 flow θ_t는 모든 점을 동시에 흘려보낸다 (0.6절, Example 0.6.6, Figure 0.6.3).

unit sphere S² 위의 두 vector field와 그 flow. 파란 화살표는 X(실제 길이의 0.3배), 검은 격자는 처음에 적도 근처
(colatitude 75°–105°, 경도 폭 40°)에 놓은 점들이고, 주황 곡선은 격자 가운데 점 p(적도 위)의
integral curve t ↦ θ_t(p)다. (1)에서는 격자를 longitude −135° 둘레에, (2)에서는 −90° 둘레에 놓았다.
  (1) X(x, y, z) = (−y, x, 0) (Example 0.6.6, Figure 0.6.3): θ_t는 z축 둘레의 각 t 회전. 격자는 모양을 유지하며 돈다.
      t = 0 → π/2.
  (2) X = grad z = e₃ − z·p (Exercise 0.8.4의 height function의 gradient): integral curve는 meridian이고
      z(t) = tanh(t + artanh z₀). 격자는 북극 쪽으로 몰리며 줄어든다(θ_t는 diffeomorphism이지만 회전이 아니다).
      t = 0 → 1.6.
두 field 모두 두 극에서 0이다. 그곳의 점은 움직이지 않는다.
자기검사: 두 X가 S²에 접함, 닫힌 식의 flow와 dgnum.rk4의 해가 일치, flow가 S²에 머묾, θ_s∘θ_t = θ_{s+t}.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-4-sphere-flows.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

import dgnum  # noqa: E402
from _helpers_b import View, arrow, graticule, polyline, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C
E3 = np.array([0.0, 0.0, 1.0])


def X_rot(x):
    return np.array([-x[..., 1], x[..., 0], np.zeros_like(x[..., 0])]).T if x.ndim > 1 else np.array([-x[1], x[0], 0.0])


def X_grad(x):
    return E3 - np.atleast_2d(x)[:, 2:3] * np.atleast_2d(x) if x.ndim > 1 else E3 - x[2] * x


def flow_rot(t, x):
    c, s = np.cos(t), np.sin(t)
    return x @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def flow_grad(t, x):
    x = np.atleast_2d(x)
    z0 = np.clip(x[:, 2], -0.999999, 0.999999)
    z = np.tanh(t + np.arctanh(z0))
    h = np.sqrt(np.maximum(1 - z0 ** 2, 1e-15))
    k = np.sqrt(np.maximum(1 - z ** 2, 0)) / h
    return np.column_stack([x[:, 0] * k, x[:, 1] * k, z])


# ---------------------------------------------------------------- 자기검사
rng = np.random.default_rng(0)
pts = rng.normal(size=(20, 3))
pts /= np.linalg.norm(pts, axis=1)[:, None]
assert np.allclose(np.einsum("ij,ij->i", X_rot(pts), pts), 0) and np.allclose(np.einsum("ij,ij->i", X_grad(pts), pts), 0)
ts = np.linspace(0, 1.6, 401)
for x0 in pts[:5]:
    for Xf, fl, T in ((X_rot, flow_rot, np.pi / 2), (X_grad, flow_grad, 1.6)):
        tt = np.linspace(0, T, 401)
        num = dgnum.rk4(lambda t, y: Xf(y), x0, tt)
        ex = np.vstack([fl(t, x0) for t in tt])
        assert np.max(np.abs(num - ex)) < 1e-7
        assert np.allclose(np.linalg.norm(ex, axis=1), 1)
        assert np.allclose(fl(0.3, fl(0.5, x0)), fl(0.8, x0))                 # θ_s∘θ_t = θ_{s+t}
for Xf in (X_rot, X_grad):
    assert np.allclose(Xf(E3), 0) and np.allclose(Xf(-E3), 0)                  # 두 극에서 0

VIEW = View(elev=28, azim=-90, scale=2.85, center=np.array([-2.8, -0.35, 0]))
GT = np.deg2rad(np.linspace(75, 105, 5))


def make_grid(lon_c):
    return np.array([[sph(a, b) for b in np.deg2rad(np.linspace(lon_c - 20, lon_c + 20, 5))] for a in GT])


GRIDS = {"rot": make_grid(-135), "grad": make_grid(-90)}          # (5, 5, 3)
P0S = {"rot": sph(np.pi / 2, np.deg2rad(-135)), "grad": sph(np.pi / 2, -np.pi / 2)}
FIELD = np.array([sph(a, b) for a in np.deg2rad([30, 60, 90, 120, 150]) for b in np.deg2rad(np.arange(0, 360, 30))])


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        self.add(sphere_disk(v, fill_opacity=0.3), graticule(v, n_lat=5, n_lon=12, color="#C4C4C4", width=1.0))
        self.add(Dot(v.p(E3), radius=0.07), da.mtex(r"$N$", 34).next_to(v.p(E3), UP, buff=0.08))
        t = ValueTracker(0.0)
        state = {"X": X_rot, "flow": flow_rot, "k": "rot"}

        def field():
            g = VGroup()
            Xv = state["X"](FIELD)
            for x, w in zip(FIELD, Xv):
                if v.depth(x) > 0.05 and np.linalg.norm(w) > 1e-6:
                    g.add(arrow(v.p(x), v.p(x) + v.vec(0.3 * w), C["tangent"], 3.5, 0.13))
            return g

        def grid():
            GRID, P0 = GRIDS[state["k"]], P0S[state["k"]]
            G = state["flow"](t.get_value(), GRID.reshape(-1, 3)).reshape(GRID.shape)
            g = VGroup()
            for i in range(5):
                g.add(sphere_curve(v, G[i], C["main"], 3, hidden=None))
                g.add(sphere_curve(v, G[:, i], C["main"], 3, hidden=None))
            for q in G.reshape(-1, 3):
                if v.depth(q) > 0:
                    g.add(Dot(v.p(q), radius=0.045, color=C["main"]))
            tr = np.vstack([state["flow"](s, P0) for s in np.linspace(0, t.get_value(), 60)])
            g.add(sphere_curve(v, tr, C["accent"], 6, hidden="faint"))
            g.add(Dot(v.p(tr[-1]), radius=0.08, color=C["accent"]))
            return g

        F = always_redraw(field)
        Gm = always_redraw(grid)
        self.add(F, Gm)
        X0 = 0.75
        title = da.label("A vector field moves every point at once", 34).to_edge(UP, buff=0.3)
        h1 = da.mtex(r"$X = (-y,\ x,\ 0)$", 38).move_to([X0, 2.2, 0], aligned_edge=LEFT)
        s1 = da.label("rotation about the z-axis", 30).move_to([X0, 1.55, 0], aligned_edge=LEFT)
        tl = always_redraw(lambda: da.label(f"t = {t.get_value():.2f}", 34).move_to([X0, 0.6, 0], aligned_edge=LEFT))
        self.add(title, h1, s1, tl)
        self.wait(1.5)
        self.play(t.animate.set_value(np.pi / 2), run_time=5.0, rate_func=linear)
        n1 = da.label("shape kept: the flow rotates", 30).move_to([X0, -0.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(n1), run_time=0.5)
        self.wait(1.5)

        # (2) height function의 gradient
        h2 = da.mtex(r"$X = \mathrm{grad}\,z = e_3 - z\,p$", 38).move_to([X0, 2.2, 0], aligned_edge=LEFT)
        s2 = da.label("flows up the meridians", 30).move_to([X0, 1.55, 0], aligned_edge=LEFT)
        self.play(FadeOut(n1), FadeOut(h1), FadeOut(s1), run_time=0.5)
        state.update(X=X_grad, flow=flow_grad, k="grad")
        t.set_value(0.0)
        self.play(FadeIn(h2), FadeIn(s2), run_time=0.6)
        self.wait(1.0)
        self.play(t.animate.set_value(1.6), run_time=5.5, rate_func=linear)
        n2 = VGroup(da.label("squeezed toward N:", 30), da.label("a diffeomorphism,", 30),
                    da.label("not a rotation", 30)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12).move_to([X0, -0.65, 0], aligned_edge=LEFT)
        n3 = da.label("X = 0 at the poles", 30).move_to([X0, -2.15, 0], aligned_edge=LEFT)
        self.play(FadeIn(n2), FadeIn(n3), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.0)

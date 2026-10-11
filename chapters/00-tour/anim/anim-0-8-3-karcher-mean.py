"""Animation 0.8.3: Karcher(Fréchet) mean의 반복 log → 평균 → exp (0.8절, In computer vision 상자 "회전의 평균").

상자의 반복을 SO(3) 대신 그림으로 볼 수 있는 unit sphere S²에서 한다. exp와 log는 (0.8.5)다:
  exp_p(v) = cos|v| p + sin|v| v/|v|,   log_p(q) = θ/sin θ (q − cos θ p),  θ = arccos⟨p, q⟩.
데이터 7개(검은 점)는 c = x(50°, −60°) 둘레에 퍼뜨린 점이고(T_cS²의 극좌표 (반지름, 각)을 정해 exp_c로 보냄),
처음 추정 x̄₀(주황)은 c에서 0.8만큼 떨어진 점이다. 화면은 c와 x̄₀의 가운데를 정면에서 본다. 한 번의 반복:
  (1) tangent plane T_x̄S²(하늘색)에 δ_i = log_x̄(x_i)(파랑)를 그린다.
  (2) 평균 δ = (1/n) Σ δ_i(검은 굵은 화살표).
  (3) x̄ ← exp_x̄(δ)(주황 geodesic을 따라 이동).
|δ|가 0.790 → 0.015 → 0.0003 → … 로 빠르게 줄고, Σ d(x̄, x_i)²도 5.270 → 0.810 → 0.808 → … 로 줄어든다.
처음 세 번을 보여 준다. 8번째 반복에서 |δ| < 10⁻¹²이다.
자기검사: exp_p(log_p q) = q, 반복이 수렴(|δ| < 1e-10), 수렴점에서 Σ log = 0이고 주변 어느 방향으로 움직여도 Σ d²가 커짐,
Σ d²가 반복마다 줄어듦.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-3-karcher-mean.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, graticule, great_arc, polyline, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C


def exp_p(p, v):
    r = np.linalg.norm(v)
    return p.copy() if r < 1e-15 else np.cos(r) * p + np.sin(r) * v / r


def log_p(p, q):
    th = np.arccos(np.clip(p @ q, -1, 1))
    return np.zeros(3) if th < 1e-15 else th / np.sin(th) * (q - np.cos(th) * p)


def basis(p):
    a = np.array([0, 0, 1.0]) if abs(p[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(a, p)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(p, e1)


def cost(p, X):
    return float(sum(np.arccos(np.clip(p @ x, -1, 1)) ** 2 for x in X))


CEN = sph(np.deg2rad(50), np.deg2rad(-60))
b1, b2 = basis(CEN)
_R = [0.30, 0.45, 0.25, 0.50, 0.20, 0.40, 0.10]
_A = np.deg2rad([15, 70, 130, 190, 250, 310, 100])
DATA = np.array([exp_p(CEN, r * (np.cos(a) * b1 + np.sin(a) * b2)) for r, a in zip(_R, _A)])
X0 = exp_p(CEN, 0.8 * (np.cos(np.deg2rad(200)) * b1 + np.sin(np.deg2rad(200)) * b2))

ITER = [X0]
DELTA = []
for _ in range(40):
    d = np.mean([log_p(ITER[-1], x) for x in DATA], axis=0)
    DELTA.append(d)
    ITER.append(exp_p(ITER[-1], d))
    if np.linalg.norm(d) < 1e-12:
        break

# ---------------------------------------------------------------- 자기검사
for x in DATA:
    assert np.allclose(exp_p(X0, log_p(X0, x)), x)
assert np.linalg.norm(DELTA[-1]) < 1e-10
xm = ITER[-1]
assert np.linalg.norm(np.sum([log_p(xm, x) for x in DATA], axis=0)) < 1e-9
e1, e2 = basis(xm)
for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    assert cost(exp_p(xm, 0.02 * (np.cos(a) * e1 + np.sin(a) * e2)), DATA) > cost(xm, DATA)
costs = [cost(p, DATA) for p in ITER[:4]]
assert all(costs[k + 1] < costs[k] for k in range(3))
NORMS = [np.linalg.norm(d) for d in DELTA]
assert NORMS[0] > 0.7 and NORMS[2] < 1e-3 and all(NORMS[k + 1] < NORMS[k] for k in range(5))

_m = (CEN + X0) / np.linalg.norm(CEN + X0)
VIEW = View(elev=np.rad2deg(np.arcsin(_m[2])) + 8, azim=np.rad2deg(np.arctan2(_m[1], _m[0])) + 10, scale=2.75,
            center=np.array([-3.25, -0.15, 0]))          # 아래 90px(control bar)를 비운다(§12.8)
assert all(VIEW.depth(x) > 0.45 for x in list(DATA) + [X0])                # 모든 점이 앞쪽에 보인다


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        self.add(sphere_disk(v, fill_opacity=0.25), graticule(v, n_lat=5, n_lon=12, color="#C4C4C4", width=1.0))
        self.add(*[Dot(v.p(x), radius=0.075, color=C["main"]) for x in DATA])
        title = da.label("Karcher mean on S²: log → average → exp", 34).to_edge(UP, buff=0.25)
        self.add(title)
        XT = 0.55
        state = {"x": ITER[0]}
        est = always_redraw(lambda: Dot(v.p(state["x"]), radius=0.11, color=C["accent"]))
        lab = da.mtex(r"$\bar x_0$", 36, color=C["accent"]).next_to(v.p(ITER[0]), DOWN, buff=0.1)
        self.add_foreground_mobject(est)            # 추정점이 tangent plane 조각과 화살표에 가리지 않도록
        self.add(lab)
        info = VGroup()
        self.wait(1.5)
        trail = VGroup()
        self.add(trail)
        for k in range(3):
            x = ITER[k]
            e1, e2 = basis(x)
            # tangent plane 조각: x와 모든 log 화살표의 끝을 담는 사각형(여백 0.15). 화살표가 조각 밖으로 나가지 않는다
            uv = np.array([[0.0, 0.0]] + [[log_p(x, q) @ e1, log_p(x, q) @ e2] for q in DATA])
            (a0, b0), (a1, b1) = uv.min(axis=0) - 0.15, uv.max(axis=0) + 0.15
            sq = np.array([x + a * e1 + b * e2 for a, b in ((a1, b1), (a0, b1), (a0, b0), (a1, b0))])
            plane = Polygon(*v.p(sq), stroke_color=C["tangent"], stroke_width=2, fill_color=C["region"],
                            fill_opacity=0.25)
            logs = VGroup(*[arrow(v.p(x), v.p(x + log_p(x, q)), C["tangent"], 4, 0.15) for q in DATA])
            arcs = VGroup(*[DashedVMobject(polyline(v.p(great_arc(x, q, 40)), C["aux"], 2), num_dashes=10)
                            for q in DATA])
            mean = arrow(v.p(x), v.p(x + DELTA[k]), C["main"], 8, 0.25)
            new_info = VGroup(da.label(f"iteration {k + 1}", 34),
                              da.label(f"|δ| = {NORMS[k]:.4f}", 32),
                              da.label(f"Σ d(x̄, xᵢ)² = {cost(x, DATA):.4f}", 32)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.15).move_to([XT, 2.0, 0], aligned_edge=UL)
            fast = k >= 1
            self.play(FadeIn(plane), FadeIn(arcs), Create(logs), FadeOut(info), FadeIn(new_info),
                      run_time=0.7 if fast else 1.2)
            info = new_info
            step = VGroup(da.label("(1) log: into the tangent plane", 30, color=C["tangent"])).move_to(
                [XT, -0.2, 0], aligned_edge=UL)
            if k == 0:
                self.play(FadeIn(step), run_time=0.4)
                self.wait(1.0)
            else:
                self.wait(0.4 if fast else 0.7)
            s2 = da.label("(2) average: δ", 30).move_to([XT, -0.75, 0], aligned_edge=UL)
            self.play(GrowFromPoint(mean, v.p(x)), *([FadeIn(s2)] if k == 0 else []), run_time=0.6 if fast else 0.9)
            self.wait(0.3 if fast else 0.7)
            s3 = da.label("(3) exp: move x̄ by δ", 30, color=C["accent"]).move_to([XT, -1.3, 0], aligned_edge=UL)
            path = great_arc(x, ITER[k + 1], 40) if np.linalg.norm(DELTA[k]) > 1e-6 else np.array([x, x])
            seg = polyline(v.p(path), C["accent"], 5)
            tr = ValueTracker(0.0)

            def mover(path=path, tr=tr):
                i = int(round(tr.get_value() * (len(path) - 1)))
                state["x"] = path[i]
                return VGroup()

            m = always_redraw(mover)
            self.add(m)
            self.play(tr.animate.set_value(1.0), Create(seg), *([FadeIn(s3)] if k == 0 else []),
                      run_time=0.7 if fast else 1.3)
            self.remove(m)
            state["x"] = ITER[k + 1]
            trail.add(seg)
            self.play(FadeOut(plane), FadeOut(logs), FadeOut(arcs), FadeOut(mean), run_time=0.4)
            if k == 0:
                self.step_group = VGroup(step, s2, s3)
        done = VGroup(da.label(f"|δ| < 10⁻¹² after {len(DELTA)} steps", 32),
                      da.label("at the mean: Σ logs = 0", 32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(
            [XT, -1.95, 0], aligned_edge=UL)
        lab2 = da.mtex(r"$\bar x$", 36, color=C["accent"]).next_to(v.p(ITER[-1]), UP + RIGHT, buff=0.06)
        self.play(FadeIn(done), FadeIn(lab2), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=3.8)

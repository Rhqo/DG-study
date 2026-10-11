"""Animation 0.8.4: sphere 위의 Riemannian gradient descent — projection과 retraction (0.8절, In computer vision 상자
"Riemannian optimization", (0.8.3) 아래 문단).

DLT처럼 |x| = 1 아래에서 f(x) = xᵀ A x를 최소화한다(상자의 ‖A h‖²와 같은 꼴, A는 symmetric positive definite).
A = Q diag(1, 2.5, 4) Qᵀ, Q의 첫 열 q = x(55°, −50°)가 최소점(가장 작은 eigenvalue의 eigenvector)이다.
회색 곡선은 f의 등위선(f = 1.3, 1.8, 2.5, 3.2, 3.7)이다. 한 걸음:
  Euclidean gradient ∇f = 2Ax(회색 화살표, 오른쪽 단면에서 0.05배)는 sphere 밖으로 비스듬하다.
  T_xS²로 projection한 grad f = ∇f − ⟨∇f, x⟩x(파랑, 0.05배)가 Riemannian gradient다.
  v = −α grad f(α = 0.12)만큼 tangent plane에서 움직인 뒤 retraction (x + v)/|x + v|로 sphere에 되돌린다(주황).
  비교: 보통의 Euclidean step x − α∇f(회색 점)는 sphere를 벗어난다.
오른쪽 패널은 x₀와 grad f가 펼치는 평면으로 자른 단면이다(∇f, grad f, v, x + v, x − α∇f가 모두 이 평면 안에 있다).
주홍은 ∇f의 normal 성분이다. 첫 걸음을 단면에서 나누어 보여 준 뒤 왼쪽 sphere에서 13걸음을 더 간다. f는 매 걸음 줄고, x는 q로 간다.
자기검사: q가 A의 가장 작은 eigenvalue의 eigenvector, grad f ⟂ x, retraction이 |x| = 1을 지킴, f가 단조 감소,
25걸음 뒤 d(x, q) < 1e-3, Euclidean step의 길이 ≠ 1.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-8-4-riemannian-gd.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, polyline, sph, sphere_curve, sphere_disk  # noqa: E402

C = da.C
q1 = sph(np.deg2rad(55), np.deg2rad(-50))
_t = np.cross(q1, [0, 0, 1.0])
q2 = _t / np.linalg.norm(_t)
q3 = np.cross(q1, q2)
Q = np.column_stack([q1, q2, q3])
A = Q @ np.diag([1.0, 2.5, 4.0]) @ Q.T
ALPHA = 0.12


def f(x):
    return float(x @ A @ x)


def egrad(x):
    return 2 * A @ x


def rgrad(x):
    g = egrad(x)
    return g - (g @ x) * x


def retr(x, v):
    y = x + v
    return y / np.linalg.norm(y)


X0 = retr(q1, 1.25 * (np.cos(np.deg2rad(150)) * q2 + np.sin(np.deg2rad(150)) * q3))
PATH = [X0]
for _ in range(40):
    PATH.append(retr(PATH[-1], -ALPHA * rgrad(PATH[-1])))

# ---------------------------------------------------------------- 자기검사
w, V = np.linalg.eigh(A)
assert np.isclose(w[0], 1) and np.isclose(abs(V[:, 0] @ q1), 1)
for x in PATH[:10]:
    assert abs(rgrad(x) @ x) < 1e-12 and np.isclose(np.linalg.norm(x), 1)
fs = [f(x) for x in PATH]
assert all(fs[k + 1] < fs[k] for k in range(15))
assert np.arccos(min(1, abs(PATH[25] @ q1))) < 1e-3
assert abs(np.linalg.norm(X0 - ALPHA * egrad(X0)) - 1) > 0.2          # Euclidean step은 sphere를 벗어난다


def dist(x):
    return float(np.arccos(np.clip(abs(x @ q1), -1, 1)))


VIEW = View(elev=35 + 8, azim=-50 + 18, scale=2.45, center=np.array([-3.9, -0.45, 0]))


def contours():
    """f의 등위선을 (θ, φ) 격자에서 matplotlib contour로 구해 sphere 위로 옮긴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = np.linspace(1e-3, np.pi - 1e-3, 181)
    ph = np.linspace(-np.pi, np.pi, 361)
    T, P = np.meshgrid(th, ph, indexing="ij")
    Xs = sph(T, P)
    F = np.einsum("...i,ij,...j->...", Xs, A, Xs)
    fig = plt.figure()
    cs = plt.contour(P, T, F, levels=[1.3, 1.8, 2.5, 3.2, 3.7])
    out = []
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg) > 5:
                out.append(sph(seg[:, 1], seg[:, 0]))
    plt.close(fig)
    return out


CONT = contours()
assert len(CONT) >= 5
for c in CONT:
    vals = np.einsum("ij,jk,ik->i", c, A, c)
    assert np.ptp(vals) < 0.05


# 오른쪽 단면: 큰 원의 중심과 반지름(화면). x₀ 근처만 보인다. x₀는 화면 높이 1.65.
# 아래 90px(멈춤 상태의 control bar)를 비우려고 반지름을 5.2로 했다(§12.8)
SL_C, SL_R = np.array([3.5, 1.65 - 5.2, 0]), 5.2


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        self.add(sphere_disk(v, fill_opacity=0.25))
        self.add(*[sphere_curve(v, c, "#9A9A9A", 2, hidden=None) for c in CONT])
        self.add(Dot(v.p(q1), radius=0.08, color=C["main"]), da.mtex(r"$q$", 34).next_to(v.p(q1), DOWN + RIGHT, buff=0.04))
        title = da.label("Gradient descent on the sphere: project, step, retract", 34).to_edge(UP, buff=0.25)
        self.add(title)
        x = X0
        dotx = Dot(v.p(x), radius=0.1, color=C["accent"])
        labx = da.mtex(r"$x_0$", 36, color=C["accent"]).next_to(v.p(x), LEFT, buff=0.1)
        self.add(dotx, labx)

        # 오른쪽: x₀, grad f가 펼치는 평면으로 자른 단면 (모든 벡터가 이 평면 안에 있다)
        g, gr = egrad(x), rgrad(x)
        u = gr / np.linalg.norm(gr)

        def Sl(y):
            """ℝ³의 점 y = a x₀ + b u (+ 수직 성분 0) → 단면 화면 좌표"""
            return SL_C + SL_R * np.array([y @ u, y @ x, 0])

        arc = polyline([SL_C + SL_R * np.array([np.sin(t), np.cos(t), 0]) for t in np.linspace(-0.5, 0.5, 80)],
                       C["main"], 4)
        # T_{x₀}S²는 하늘색(region)으로 그린다: 그 위에 놓이는 파란 grad f와 v가 보이도록
        tline = Line(Sl(x - 0.5 * u), Sl(x + 0.5 * u), stroke_color=C["region"], stroke_width=3)
        cap = VGroup(da.label("slice through", 30), da.label("x₀ and grad f", 30)).arrange(DOWN, aligned_edge=RIGHT, buff=0.1).move_to([6.9, 0.2, 0], aligned_edge=RIGHT)
        tl = da.mtex(r"$T_{x_0}S^2$", 32, color=C["tangent"]).next_to(Sl(x - 0.5 * u), UP, buff=0.08)
        sl_x = Dot(Sl(x), radius=0.09, color=C["accent"])
        self.add(arc, tline, cap, tl, sl_x, da.mtex(r"$S^2$", 32).next_to(arc.get_start(), DOWN, buff=0.1))
        self.wait(1.2)
        SCG = 0.05
        a_e = arrow(Sl(x), Sl(x + SCG * g), "#808080", 6, 0.2)
        a_n = arrow(Sl(x), Sl(x + SCG * (g @ x) * x), C["normal"], 5, 0.18)
        a_r = arrow(Sl(x), Sl(x + SCG * gr), C["tangent"], 6, 0.2)
        t1 = da.label("∇f = 2Ax: normal part + tangent part", 30).move_to([-1.2, -1.4, 0], aligned_edge=UL)
        self.play(GrowFromPoint(a_e, Sl(x)), FadeIn(t1), run_time=0.9)
        self.wait(0.8)
        self.play(GrowFromPoint(a_n, Sl(x)), GrowFromPoint(a_r, Sl(x)), run_time=0.9)
        t2 = VGroup(da.label("keep the tangent part: grad f", 30, color=C["tangent"]), da.label("(the Riemannian gradient)", 30, color=C["tangent"])).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-1.2, -1.4, 0], aligned_edge=UL)
        self.play(FadeOut(t1), FadeOut(a_n), FadeOut(a_e), FadeIn(t2), run_time=0.6)
        self.wait(1.2)
        step = -ALPHA * gr
        x1 = retr(x, step)
        a_s = arrow(Sl(x), Sl(x + step), C["tangent"], 8, 0.22)
        t3 = VGroup(da.label("step v = −α grad f,", 30), da.label("then retract (x + v)/|x + v|", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-1.2, -1.4, 0], aligned_edge=UL)
        self.play(FadeOut(a_r), GrowFromPoint(a_s, Sl(x)), FadeOut(t2), FadeIn(t3), run_time=0.8)
        self.wait(0.6)
        off = Dot(Sl(x + step), radius=0.07, color=C["tangent"])
        ray = DashedLine(Sl(x + step), Sl(x1), stroke_color=C["accent"], stroke_width=3, dash_length=0.06)
        sl_x1 = Dot(Sl(x1), radius=0.09, color=C["accent"])
        self.play(FadeIn(off), Create(ray), FadeIn(sl_x1), dotx.animate.move_to(v.p(x1)),
                  Create(polyline(v.p(np.array([x, x1])), C["accent"], 4)), run_time=1.0)
        self.wait(0.8)
        eu = x - ALPHA * g
        a_eu = DashedLine(Sl(x), Sl(eu), stroke_color="#808080", stroke_width=2.5, dash_length=0.08)
        eud = Dot(Sl(eu), radius=0.08, color="#808080")
        t4 = VGroup(da.label("plain step x − α∇f leaves S²", 30, color="#606060").move_to(
                        [-1.2, -2.45, 0], aligned_edge=UL),
                    da.label(f"|x − α∇f| = {np.linalg.norm(eu):.2f}", 30, color="#606060").next_to(
                        Sl(eu), RIGHT, buff=0.2).shift(UP * 0.15))
        self.play(Create(a_eu), FadeIn(eud), FadeIn(t4), run_time=0.8)
        self.wait(2.0)

        # 이어서
        trail = VGroup()
        self.add(trail)
        k = {"v": 1}
        info = always_redraw(lambda: VGroup(
            da.label(f"step {k['v']}:  f(x) = {f(PATH[k['v']]):.4f}", 32),
            da.label(f"distance to q = {dist(PATH[k['v']]):.4f}", 32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .to_corner(UL, buff=0.35).shift(DOWN * 0.6))
        self.play(FadeOut(t3), FadeOut(t4), run_time=0.4)
        self.add(info)
        for j in range(1, 14):
            a, b = PATH[j], PATH[j + 1]
            st = arrow(v.p(a), v.p(a - ALPHA * rgrad(a)), C["tangent"], 5, 0.14)
            self.add(st)
            seg = polyline(v.p(np.array([a, b])), C["accent"], 4)
            k["v"] = j + 1
            self.play(dotx.animate.move_to(v.p(b)), Create(seg), run_time=0.55 if j < 5 else 0.3)
            self.remove(st)
            trail.add(seg)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.5)

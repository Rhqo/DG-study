"""Animation 0.6.6: Euler angle R = R_z(α) R_y(β) R_x(γ)의 gimbal lock (0.6절, In computer vision 상자 "카메라 자세").

gimbal 세 고리: 바깥(α, 검정)은 world z축 둘레로, 가운데(β, 초록)는 축 a_β = R_z(α) e_y 둘레로, 안쪽(γ, 주황)은
축 a_γ = R_z(α) R_y(β) e_x 둘레로 돈다. 고리의 평면은 차례로 span{e_z, a_β}, span{a_β, a_γ}, span{a_γ, R e_z}이고
반지름은 1.9, 1.6, 1.3이다. 안쪽 고리에 붙은 물체(하늘색 frustum, 파란 화살표 = R e_x)가 R로 돈다.
세 축의 각속도 기여는 ω = α̇ e_z + β̇ a_β + γ̇ a_γ이므로 differential의 rank는 det[e_z, a_β, a_γ] = −cos β로 정해진다.
  1부: β = 0에서 α와 γ를 차례로 ±25° 흔든다. 물체가 서로 다른 축(z축, 앞 방향) 둘레로 돈다.
  2부: β를 0 → 90°로. 가운데 고리가 바깥 고리와 같은 평면으로 접히고 |cos β| → 0.
  3부: β = 90°에서 α와 γ를 차례로 흔든다. 두 경우 모두 물체가 z축 둘레로 돈다(R = R_z(α − γ) R_y(π/2)). 자유도 하나가 사라진다.
자기검사: R ∈ SO(3), det[e_z, a_β, a_γ] = −cos β, 수치 미분한 differential(각속도 성분 3×3)의 singular value가
β = 0에서 모두 1, β = π/2에서 가장 작은 것이 0(rank 2), R(α, π/2, γ) = R_z(α − γ) R_y(π/2).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-6-6-gimbal-lock.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import View, arrow, polyline  # noqa: E402

C = da.C


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def euler(a, b, g):
    return Rz(a) @ Ry(b) @ Rx(g)


def axes_of(a, b):
    e = np.eye(3)
    return e[2], Rz(a) @ e[1], Rz(a) @ Ry(b) @ e[0]


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def ang_vel_matrix(a, b, g, h=1e-6):
    """열 k: k번째 각을 미분했을 때의 world 각속도 vee(dR Rᵀ)."""
    R = euler(a, b, g)
    cols = []
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        dR = (euler(*(np.array([a, b, g]) + d)) - euler(*(np.array([a, b, g]) - d))) / (2 * h)
        cols.append(vee(dR @ R.T))
    return np.column_stack(cols)


# ---------------------------------------------------------------- 자기검사
rng = np.random.default_rng(2)
for a, b, g in rng.uniform(-np.pi, np.pi, size=(6, 3)):
    R = euler(a, b, g)
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    A = np.column_stack(axes_of(a, b))
    assert np.isclose(np.linalg.det(A), -np.cos(b))
    assert np.allclose(ang_vel_matrix(a, b, g), A, atol=1e-6)
    assert np.allclose(euler(a, np.pi / 2, g), Rz(a - g) @ Ry(np.pi / 2))
assert np.allclose(np.linalg.svd(ang_vel_matrix(0.3, 0.0, 0.2), compute_uv=False), 1, atol=1e-6)
sv = np.linalg.svd(ang_vel_matrix(0.3, np.pi / 2, 0.2), compute_uv=False)
assert sv[-1] < 1e-6 and sv[1] > 0.5                                                  # rank 2

VIEW = View(elev=20, azim=-35, scale=1.55, center=np.array([-3.6, -0.45, 0]))
RAD = (1.9, 1.6, 1.3)


class Anim(da.Scene2D):
    def construct(self):
        v = VIEW
        al, be, ga = ValueTracker(np.deg2rad(20)), ValueTracker(0.0), ValueTracker(0.0)

        def ring(u, w, r, col, width):
            s = np.linspace(0, 2 * np.pi, 121)
            P = r * (np.cos(s)[:, None] * u + np.sin(s)[:, None] * w)
            front = v.depth(P) >= 0
            g = VGroup()
            for mask, op in ((~front, 0.35), (front, 1.0)):
                idx = np.where(mask)[0]
                if len(idx) < 2:
                    continue
                splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                for sp_ in splits:
                    if len(sp_) >= 2:
                        g.add(polyline(v.p(P[sp_]), col, width, op))
            return g

        def gimbal():
            a, b, gg = al.get_value(), be.get_value(), ga.get_value()
            aa, ab, ag = axes_of(a, b)
            R = euler(a, b, gg)
            G = VGroup()
            # 축 (점선)
            for ax, col in ((aa, C["main"]), (ab, C["third"]), (ag, C["accent"])):
                G.add(DashedVMobject(polyline(v.p(np.array([-2.3 * ax, 2.3 * ax])), col, 2.2), num_dashes=22))
            G.add(ring(aa, ab, RAD[0], C["main"], 7))
            G.add(ring(ab, ag, RAD[1], C["third"], 7))
            G.add(ring(ag, R @ np.array([0, 0, 1.0]), RAD[2], C["accent"], 7))
            ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
            rect = [0.95 * ex + sy * 0.38 * ey + sz * 0.27 * ez for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            G.add(Polygon(*v.p(np.array(rect)), stroke_color=C["main"], stroke_width=2, fill_color=C["region"],
                          fill_opacity=0.5))
            for c in rect:
                G.add(polyline(v.p(np.array([[0, 0, 0], c])), C["main"], 2))
            G.add(arrow(v.p([0, 0, 0]), v.p(1.15 * ex), C["tangent"], 7, 0.22))
            G.add(Dot(v.p([0, 0, 0]), radius=0.06))
            return G

        self.add(always_redraw(gimbal))
        title = VGroup(da.label("Euler angles", 34), da.mtex(r"$R_z(\alpha)\,R_y(\beta)\,R_x(\gamma)$", 36),
                       da.label("and gimbal lock", 34)).arrange(RIGHT, buff=0.25).to_edge(UP, buff=0.25)
        self.add(title)
        X0 = 0.45
        leg = VGroup(*[VGroup(Line(ORIGIN, RIGHT * 0.55, color=col, stroke_width=7), da.label(txt, 30)).arrange(RIGHT)
                       for col, txt in ((C["main"], "α ring (about z)"), (C["third"], "β ring"),
                                        (C["accent"], "γ ring (about the nose)"))])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([X0, 2.35, 0], aligned_edge=UL)
        self.add(leg)

        def nums():
            g = VGroup(da.label((f"α = {np.rad2deg(al.get_value()):.0f}°   β = {np.rad2deg(be.get_value()):.0f}°   "
                                 f"γ = {np.rad2deg(ga.get_value()):.0f}°").replace("-", "−"), 32),
                       da.label(f"axes volume |cos β| = {abs(np.cos(be.get_value())):.2f}", 32))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([X0, 0.2, 0], aligned_edge=UL)

        self.add(always_redraw(nums))

        def wiggle(tr, amp=np.deg2rad(25), rt=2.4):
            x0 = tr.get_value()
            self.play(tr.animate.set_value(x0 + amp), run_time=rt / 4, rate_func=smooth)
            self.play(tr.animate.set_value(x0 - amp), run_time=rt / 2, rate_func=smooth)
            self.play(tr.animate.set_value(x0), run_time=rt / 4, rate_func=smooth)

        def note(txt, col=C["main"]):
            return da.label(txt, 30, color=col).move_to([X0, -1.45, 0], aligned_edge=UL)

        self.wait(1.5)
        n = note("β = 0: wiggle α, then γ")
        self.play(FadeIn(n), run_time=0.4)
        wiggle(al)
        wiggle(ga)
        n2 = VGroup(da.label("two different axes", 30)).move_to([X0, -2.05, 0], aligned_edge=UL)
        self.play(FadeIn(n2), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(n), FadeOut(n2), run_time=0.4)
        self.play(be.animate.set_value(np.pi / 2), run_time=4.0, rate_func=smooth)
        self.wait(1.0)
        n = note("β = 90°: wiggle α, then γ")
        self.play(FadeIn(n), run_time=0.4)
        wiggle(al)
        wiggle(ga)
        n2 = VGroup(da.label("same axis: one freedom lost", 30),
                    da.label("(only α − γ matters)", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        n2.move_to([X0, -2.05, 0], aligned_edge=UL)
        self.play(FadeIn(n2), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.0)

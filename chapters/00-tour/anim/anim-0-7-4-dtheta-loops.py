"""Animation 0.7.4: ω = dθ는 구멍을 도는 고리에서 2π, 구멍을 감싸지 않는 고리에서 0 (0.7절, Example 0.7.4, (0.7.7), Figure 0.7.4).

ω = (−y dx + x dy)/(x² + y²)를 ℝ² ∖ {0} 위의 등위선 묶음, 곧 반직선 θ = 2πk/16(자주색)으로 그렸다(Figure 0.7.4와 같다).
  1부: γ₁(중심 0, 반지름 1.5, 반시계 방향)을 따라 점이 돈다. 지금까지의 ∫ω(= 연속으로 이어 잰 각의 변화)와
       가로지른 반직선의 부호 있는 개수를 함께 보인다. 한 바퀴 뒤 2π, 16개.
  2부: γ₂(중심 (1.9, 1.1), 반지름 0.6)를 따라 돈다. 값이 올라갔다가 다시 내려와 0이 된다(같은 반직선을 앞으로, 뒤로).
  3부: γ₂를 원점 쪽으로 밀어 (0.25, 0.15)까지 옮긴다. 고리가 원점을 넘지 않는 동안 ∮ω = 0이고,
       고리가 원점을 지나는 순간(ω가 정의되지 않는 곳) 값이 2π로 바뀐다. closed이지만 exact가 아니다.
자기검사: 수치 적분 ∮_{γ₁} ω = 2π, ∮_{γ₂} ω = 0(Figure 0.7.4의 검사와 같다), dω = 0(수치 curl),
밀어 옮기는 동안의 ∮ω가 원점을 감싸는지에 따라 0 또는 2π, 반직선 개수 16과 0.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-7-4-dtheta-loops.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import polyline  # noqa: E402

C = da.C
NR = 16


def omega(x, y, dx, dy):
    return (-y * dx + x * dy) / (x ** 2 + y ** 2)


def loop(c, r, t):
    t = np.asarray(t, float)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], -1)


def start_angle(c):
    """γ₂ 꼴의 고리는 원점에서 가장 먼 점(원점 → 중심 방향)에서 출발한다. 원점이 안에 있으면 0."""
    return np.arctan2(c[1], c[0]) if np.hypot(*c) > 1e-9 else 0.0


def loop_integral(c, r, n=20000):
    t = np.linspace(0, 2 * np.pi, n + 1)
    P = loop(c, r, t)
    val = omega(P[:, 0], P[:, 1], -r * np.sin(t), r * np.cos(t))
    return float(np.sum((val[1:] + val[:-1]) / 2 * np.diff(t)))


def running(c, r, s, n=400):
    """출발점에서 매개변수 s까지의 ∫ω = 연속으로 이은 극각의 변화, 그리고 가로지른 반직선의 부호 있는 개수."""
    a0 = start_angle(c)
    t = a0 + np.linspace(0, s, n)
    P = loop(c, r, t)
    th = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    k = np.floor(th / (2 * np.pi / NR) + 1e-9)
    return th[-1] - th[0], int(k[-1] - k[0])


G1, G2 = ((0.0, 0.0), 1.5), ((1.9, 1.1), 0.6)
CEND = np.array([0.25, 0.15])

# ---------------------------------------------------------------- 자기검사
assert abs(loop_integral(*G1) - 2 * np.pi) < 1e-8 and abs(loop_integral(*G2)) < 1e-8
h = 1e-5
for x, y in ((0.7, 0.3), (-1.2, 0.5), (0.2, -0.9)):
    P_y = ((-(y + h) / (x * x + (y + h) ** 2)) - (-(y - h) / (x * x + (y - h) ** 2))) / (2 * h)
    Q_x = (((x + h) / ((x + h) ** 2 + y * y)) - ((x - h) / ((x - h) ** 2 + y * y))) / (2 * h)
    assert abs(Q_x - P_y) < 1e-6                                                  # dω = 0
assert np.isclose(running(*G1, 2 * np.pi)[0], 2 * np.pi) and running(*G1, 2 * np.pi)[1] == NR
assert abs(running(*G2, 2 * np.pi)[0]) < 1e-9 and running(*G2, 2 * np.pi)[1] == 0
for lam in np.linspace(0, 1, 21):
    c = (1 - lam) * np.array(G2[0]) + lam * CEND
    d = np.hypot(*c)
    if abs(d - G2[1]) > 0.02:
        want = 2 * np.pi if d < G2[1] else 0.0
        assert abs(loop_integral(c, G2[1]) - want) < 1e-6

SC = 1.25
O = np.array([-3.45, -0.3, 0])


def S(P):
    P = np.atleast_2d(P)
    return O + SC * np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])


class Anim(da.Scene2D):
    def construct(self):
        rays = VGroup()
        for k in range(NR):
            a = 2 * np.pi * k / NR
            rays.add(Line(S([0.12 * np.cos(a), 0.12 * np.sin(a)])[0], S([3.3 * np.cos(a), 3.3 * np.sin(a)])[0],
                          stroke_color=C["covector"], stroke_width=2))
        # 오른쪽 글자 영역으로 넘어간 반직선은 흰 사각형으로 가린다
        hole = Circle(radius=0.12 * SC, stroke_color=C["main"], stroke_width=2.5, fill_color=WHITE,
                      fill_opacity=1).move_to(O)
        mask_r = Rectangle(width=8, height=9, stroke_width=0, fill_color=WHITE, fill_opacity=1).move_to(
            [O[0] + 3.2 + 4, 0, 0])
        self.add(rays, mask_r, hole)
        title = da.mtex(r"$\omega = d\theta = \frac{-y\,dx + x\,dy}{x^2+y^2}$ on $\mathbb{R}^2\setminus\{0\}$",
                        34).move_to([3.55, 3.3, 0])            # 가장자리 여백 0.2 이상(§12.8)
        self.add(title)
        X0 = 0.3                     # 가장 긴 줄(γ₂ 결과)도 오른쪽 여백 0.2 이상(§12.8)
        cur = {"c": np.array(G1[0]), "r": G1[1], "col": C["tangent"]}
        s = ValueTracker(0.0)

        def drawn():
            c, r = cur["c"], cur["r"]
            a0 = start_angle(c)
            full = polyline(S(loop(c, r, np.linspace(0, 2 * np.pi, 160))), cur["col"], 2.5, 0.35)
            ss = s.get_value()
            part = loop(c, r, a0 + np.linspace(0, max(ss, 1e-4), 160))
            return VGroup(full, polyline(S(part), cur["col"], 6), Dot(S(part[-1])[0], radius=0.1, color=cur["col"]))

        def readout():
            val, cnt = running(cur["c"], cur["r"], s.get_value())
            val = 0.0 if abs(val) < 5e-4 else val
            return VGroup(da.label(f"∫ω so far = {val:.3f}".replace("-", "−"), 32),
                          da.label(f"rays crossed (signed): {f'{cnt:+d}' if cnt else '0'}".replace("-", "−"), 32)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.15).move_to([X0, 1.85, 0], aligned_edge=LEFT)

        D = always_redraw(drawn)
        R = always_redraw(readout)
        self.add(D, R)
        lab1 = da.mtex(r"$\gamma_1$", 38, color=C["tangent"]).move_to(S([-1.25, -1.35])[0])
        self.add(lab1)
        self.wait(1.5)
        self.play(s.animate.set_value(2 * np.pi), run_time=6.0, rate_func=linear)
        r1 = da.label("∮ω over γ₁ = 2π (hole inside)", 30, color=C["tangent"]).move_to(
            [X0, 0.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(r1), run_time=0.5)
        self.wait(1.5)

        # 2부
        self.remove(D)
        self.add(polyline(S(loop(*G1, np.linspace(0, 2 * np.pi, 160))), C["tangent"], 3, 0.45))
        cur.update(c=np.array(G2[0]), r=G2[1], col=C["accent"])
        s.set_value(0.0)
        D = always_redraw(drawn)
        lab2 = da.mtex(r"$\gamma_2$", 38, color=C["accent"]).move_to(S([2.55, 1.75])[0])
        self.add(D, lab2)
        self.wait(0.5)
        self.play(s.animate.set_value(2 * np.pi), run_time=5.0, rate_func=linear)
        r2 = da.label("∮ω over γ₂ = 0 (no hole inside)", 30, color=C["accent"]).move_to(
            [X0, 0.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(r2), run_time=0.5)
        self.wait(1.5)

        # 3부: 고리를 밀어 원점을 넘긴다
        self.play(FadeOut(R), FadeOut(lab2), run_time=0.4)
        lam = ValueTracker(0.0)
        self.remove(D)
        self.add(polyline(S(loop(*G2, np.linspace(0, 2 * np.pi, 160))), C["accent"], 3, 0.45))

        def slid():
            c = (1 - lam.get_value()) * np.array(G2[0]) + lam.get_value() * CEND
            return polyline(S(loop(c, G2[1], np.linspace(0, 2 * np.pi, 160))), C["third"], 6)

        def slid_val():
            c = (1 - lam.get_value()) * np.array(G2[0]) + lam.get_value() * CEND
            d = np.hypot(*c)
            if abs(d - G2[1]) < 0.03:
                txt = "loop hits the hole: undefined"
            else:
                v = loop_integral(c, G2[1], 4000)
                txt = f"∮ω = {abs(v):.3f}" + ("  = 2π" if v > 1 else "")
            return da.label(txt, 32, color=C["third"]).move_to([X0, -1.25, 0], aligned_edge=LEFT)

        Sd, Sv = always_redraw(slid), always_redraw(slid_val)
        h3 = da.label("slide γ₂ toward the hole:", 32).move_to([X0, -0.7, 0], aligned_edge=LEFT)
        self.play(FadeIn(Sd), FadeIn(Sv), FadeIn(h3), run_time=0.5)
        self.wait(0.8)
        self.play(lam.animate.set_value(1.0), run_time=6.0, rate_func=smooth)
        fin = VGroup(da.label("dω = 0 (closed), yet", 32), da.label("not df for any f (not exact)", 32)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12).move_to([X0, -2.5, 0], aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.8)

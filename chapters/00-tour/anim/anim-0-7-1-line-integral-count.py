"""Animation 0.7.1: 1-form df를 등위선 묶음으로 보고, 곡선이 가로지르는 선의 개수를 세면 line integral이 된다
(0.7절, Intuition 상자 "covector는 등고선의 묶음이다", (0.7.3), (0.7.4), Figure 0.7.1).

f(x, y) = x² + y²(Figure 0.7.1과 같은 f), ω = df. 자주색 원은 등위선 f = k·h이다.
곡선을 따라 점이 움직일 때 바깥쪽으로(f가 커지는 쪽으로) 선을 넘으면 +1, 안쪽으로 넘으면 −1을 센다.
세어진 수 × h는 ∫_γ df = f(b) − f(a)의 근사이고, 오차는 h보다 작다((0.7.4)).
a = (0.4 cos 15°, 0.4 sin 15°), b = (1.45 cos 50°, 1.45 sin 50°)이므로 f(b) − f(a) = 2.1025 − 0.16 = 1.9425.
  1부: h = 1/4, 앞뒤로 흔들리는 곡선 γ₁. 몇 번은 안쪽으로 되돌아가므로 −1도 나온다. 합 8 → 8h = 2.00.
  2부: h = 1/16로 묶음을 촘촘하게 하고 γ₁을 다시: 합 31 → 31h = 1.9375.
  3부: 같은 a, b를 잇는 다른 곡선 γ₂(선분): 같은 31. 4부: closed curve γ₃(중심 (1, 0.75)): 0.
자기검사: 셈 = ⌊f(b)/h⌋ − ⌊f(a)/h⌋, |셈·h − (f(b) − f(a))| < h, γ₁에는 −1 crossing이 있음, closed curve는 0,
선적분의 수치 적분(사다리꼴) = f(b) − f(a).

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-7-1-line-integral-count.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import polyline  # noqa: E402

C = da.C


def f(P):
    P = np.asarray(P, float)
    return P[..., 0] ** 2 + P[..., 1] ** 2


def polar(r, deg):
    return np.array([r * np.cos(np.deg2rad(deg)), r * np.sin(np.deg2rad(deg))])


A, B = polar(0.4, 15), polar(1.45, 50)
TS = np.linspace(0, 1, 2001)


def gamma1(t):
    t = np.asarray(t, float)
    r = 0.4 + 1.05 * t + 0.2 * np.sin(3 * np.pi * t)
    a = 15 + 35 * t + 18 * np.sin(2 * np.pi * t)
    return np.stack([r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))], -1)


def gamma2(t):
    t = np.asarray(t, float)[..., None]
    return (1 - t) * A + t * B


def gamma3(t):
    """closed curve: 중심 (1.0, 0.75), 반지름 0.38(1 + 0.25 sin 2πt)을 시계 반대 방향으로 한 바퀴"""
    t = np.asarray(t, float)
    ang = -np.pi * 0.75 + 2 * np.pi * t
    rr = 0.38 * (1 + 0.25 * np.sin(2 * np.pi * t))
    return np.stack([1.0 + rr * np.cos(ang), 0.75 + rr * np.sin(ang)], -1)


def count(path, h, t1=1.0):
    k = np.floor(f(path(TS[TS <= t1])) / h)
    return int(k[-1] - k[0]), np.diff(k)


# ---------------------------------------------------------------- 자기검사
exact = f(B) - f(A)
assert np.isclose(exact, 1.9425)
for path in (gamma1, gamma2):
    assert np.allclose(path(0.0), A) and np.allclose(path(1.0), B)
    P = path(TS)
    dP = np.gradient(P, TS, axis=0)
    integrand = 2 * P[:, 0] * dP[:, 0] + 2 * P[:, 1] * dP[:, 1]          # df(γ') = 2x x' + 2y y'
    assert np.isclose(np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(TS)), exact, atol=1e-4)
    for h in (0.25, 1 / 16):
        n, steps = count(path, h)
        assert n == int(np.floor(f(B) / h) - np.floor(f(A) / h))
        assert abs(n * h - exact) < h
        assert np.all(np.abs(steps) <= 1)                                 # 한 걸음에 한 선까지만
assert count(gamma1, 0.25)[0] == 8 and count(gamma1, 1 / 16)[0] == 31
assert np.any(count(gamma1, 0.25)[1] < 0)                                  # 안쪽으로 되돌아가는 crossing
assert np.allclose(gamma3(0.0), gamma3(1.0)) and count(gamma3, 1 / 16)[0] == 0
assert np.any(count(gamma3, 1 / 16)[1] < 0) and np.any(count(gamma3, 1 / 16)[1] > 0)

SC = 3.0
O = np.array([-6.5, -2.85, 0])          # 아래 90px(멈춤 상태의 control bar)를 비운다(§12.8)


def S(P):
    P = np.atleast_2d(P)
    return O + SC * np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])


class Anim(da.Scene2D):
    def stack(self, h, width):
        g = VGroup()
        s = np.linspace(0, np.pi / 2, 90)
        k = 1
        while np.sqrt(k * h) < 1.9:
            r = np.sqrt(k * h)
            g.add(polyline(S(np.column_stack([r * np.cos(s), r * np.sin(s)])), C["covector"], width))
            k += 1
        return g

    def construct(self):
        axes = VGroup(Arrow(S([0, 0])[0], S([2.0, 0])[0], buff=0, stroke_width=2, color=C["aux"], tip_length=0.15),
                      Arrow(S([0, 0])[0], S([0, 2.0])[0], buff=0, stroke_width=2, color=C["aux"], tip_length=0.15))
        self.add(axes)
        st = self.stack(0.25, 2.5)
        self.add(st)
        title = da.label("A 1-form is a stack: count the crossings", 34).to_edge(UP, buff=0.25)
        X0 = -0.15
        eq = da.mtex(r"$\omega = df,\ \ f = x^2 + y^2$", 36).move_to([X0, 2.45, 0], aligned_edge=LEFT)
        self.add(title, eq)
        h = {"v": 0.25}
        hl = always_redraw(lambda: da.label(f"lines f = k·h,  h = {'1/4' if h['v'] == 0.25 else '1/16'}", 32,
                                            color=C["covector"]).move_to([X0, 1.75, 0], aligned_edge=LEFT))
        self.add(hl)
        dots = VGroup(Dot(S(A)[0], radius=0.08), Dot(S(B)[0], radius=0.08),
                      da.mtex(r"$a$", 36).next_to(S(A)[0], DOWN + LEFT, buff=0.06),
                      da.mtex(r"$b$", 36).next_to(S(B)[0], UP + RIGHT, buff=0.06))
        self.add(dots)

        t = ValueTracker(0.0)
        cur = {"path": gamma1, "col": C["tangent"]}

        def trail():
            tt = t.get_value()
            P = cur["path"](np.linspace(0, max(tt, 1e-4), 200))
            return VGroup(polyline(S(P), cur["col"], 5), Dot(S(P[-1])[0], radius=0.1, color=cur["col"]))

        def counter():
            n, _ = count(cur["path"], h["v"], max(t.get_value(), 1e-6))
            val = f"{n * h['v']:.4f}".rstrip("0").rstrip(".")
            sn = f"{n:+d}".replace("-", "−") if n else "0"                       # Unicode minus, 0에는 부호 없음
            return VGroup(da.label(f"signed crossings: {sn}", 32),
                          da.label(f"× h = {val}".replace("-", "−"), 32)).arrange(
                DOWN, aligned_edge=LEFT, buff=0.15).move_to([X0, 0.55, 0], aligned_edge=LEFT)

        tr = always_redraw(trail)
        cn = always_redraw(counter)
        ex = da.mtex(r"$\int_\gamma df = f(b) - f(a) = 1.9425$", 36).move_to([X0, -0.8, 0], aligned_edge=LEFT)
        self.add(tr, cn)
        self.wait(1.5)
        self.play(t.animate.set_value(1.0), run_time=7.0, rate_func=linear)
        self.play(FadeIn(ex), run_time=0.5)
        self.wait(2.0)

        # 2부: 촘촘한 묶음
        st2 = self.stack(1 / 16, 1.4)
        h["v"] = 1 / 16
        t.set_value(0.0)
        self.play(FadeOut(st), FadeIn(st2), run_time=0.8)
        self.play(t.animate.set_value(1.0), run_time=4.5, rate_func=linear)
        self.wait(1.5)

        # 3부: 다른 곡선
        ghost = polyline(S(gamma1(np.linspace(0, 1, 300))), C["tangent"], 3, 0.5)
        self.add(ghost)
        cur.update(path=gamma2, col=C["accent"])
        t.set_value(0.0)
        lab2 = VGroup(da.label("another curve from a to b:", 30, color=C["accent"]), da.label("the same count", 30)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.1).move_to([X0, -2.0, 0], aligned_edge=LEFT)
        self.play(FadeIn(lab2), run_time=0.4)
        self.play(t.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        self.wait(1.5)

        # 4부: closed curve
        ghost2 = polyline(S(gamma2(np.linspace(0, 1, 50))), C["accent"], 3, 0.5)
        self.add(ghost2)
        cur.update(path=gamma3, col=C["third"])
        t.set_value(0.0)
        lab3 = da.label("a closed curve: 0", 30, color=C["third"]).move_to([X0, -2.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(lab3), ex.animate.set_opacity(0.3), run_time=0.4)   # a → b의 식은 closed curve와 무관하다
        self.play(t.animate.set_value(1.0), run_time=4.0, rate_func=linear)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.5)

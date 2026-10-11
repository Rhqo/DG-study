"""Animation 0.2.4: curvature scale space — smoothing할수록 inflection point가 짝지어 사라진다
(Section 0.2, In computer vision 상자, Figure 0.2.4, [Mokhtarian86]의 방법을 다시 계산).

Figure 0.2.4와 같은 contour: r(φ) = 1 + 0.22 cos 3φ + 0.1 sin(5φ + 0.7) + noise (seed 0), arc length에 비례하는
매개변수 u ∈ [0, 1)로 N = 1024점. x(u), y(u)를 폭 σ(둘레에 대한 비율)인 주기 Gaussian으로 smoothing하고(FFT),
κ_s = (x'y'' − y'x'')/(x'² + y'²)^{3/2} ((0.2.3))의 부호가 바뀌는 u를 inflection point로 찾는다.
σ를 0.0015에서 0.08까지 로그 눈금으로 키운다. Figure 0.2.4의 세 값 σ = 0.002, 0.01, 0.04에서 멈춘다.
왼쪽: 지금의 σ로 smoothing한 contour, inflection point(주황), u = 0인 출발점(파란 네모).
오른쪽: (u, σ) 평면에 지금까지의 σ에서 찾은 inflection point를 찍어 나간다(curvature scale space image).
자기검사: 같은 FFT 미분으로 원의 κ_s = 1/r, inflection point의 개수는 언제나 짝수, 가장 큰 σ에서 0개(convex),
가장 작은 σ에서의 개수가 중간 σ보다 많다, ∫κ_s ds = 2π (Theorem 0.2.7).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from manim import (LEFT, RIGHT, WHITE, DashedLine, Dot, Line, Rectangle, Square, ValueTracker, VGroup, always_redraw,
                   linear)

import _helpers_a as ha
import dganim as da

C = da.C
N = 1024
rng = np.random.default_rng(0)


def fft_deriv(f, order, sigma=0.0):
    k = np.fft.fftfreq(len(f), d=1.0 / len(f))
    F = np.fft.fft(f) * np.exp(-0.5 * (2 * np.pi * k * sigma) ** 2)
    return np.real(np.fft.ifft(F * (2j * np.pi * k) ** order))


def curvature(x, y, sigma):
    x1, y1 = fft_deriv(x, 1, sigma), fft_deriv(y, 1, sigma)
    x2, y2 = fft_deriv(x, 2, sigma), fft_deriv(y, 2, sigma)
    return (x1 * y2 - y1 * x2) / (x1 ** 2 + y1 ** 2) ** 1.5


def zero_crossings(k):
    s = np.sign(k)
    return np.nonzero(s != np.roll(s, -1))[0]


u = np.arange(N) / N
for rr in (0.5, 2.0):
    assert np.allclose(curvature(rr * np.cos(2 * np.pi * u), rr * np.sin(2 * np.pi * u), 0.0), 1 / rr, atol=1e-8)

# Figure 0.2.4와 같은 contour (같은 seed, 같은 순서의 난수)
M = 8192
ph = np.arange(M) / M * 2 * np.pi
r = 1 + 0.22 * np.cos(3 * ph) + 0.1 * np.sin(5 * ph + 0.7)
for kk in range(8, 40):
    r += 0.012 * rng.normal() * np.cos(kk * ph + rng.uniform(0, 2 * np.pi)) / np.sqrt(kk / 8)
X0, Y0 = r * np.cos(ph), r * np.sin(ph)
seg = np.hypot(np.diff(np.append(X0, X0[0])), np.diff(np.append(Y0, Y0[0])))
s_ = np.concatenate([[0], np.cumsum(seg)])
x = np.interp(u * s_[-1], s_, np.append(X0, X0[0]))
y = np.interp(u * s_[-1], s_, np.append(Y0, Y0[0]))
assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0      # 반시계 방향

SMIN, SMAX = 0.0015, 0.08
LEVELS = np.geomspace(SMIN, SMAX, 420)
CSS = [zero_crossings(curvature(x, y, sg)) for sg in LEVELS]
counts = np.array([len(z) for z in CSS])
assert np.all(counts % 2 == 0)
assert counts[-1] == 0 and counts[0] > 6 and counts[0] > counts[len(counts) // 2]
total = np.sum(curvature(x, y, 0.01) * np.hypot(fft_deriv(x, 1, 0.01), fft_deriv(y, 1, 0.01))) / N
assert abs(total - 2 * np.pi) < 1e-6                                     # ∫κ_s ds = 2π

# 화면 배치
CL, SCL = np.array([-3.75, 0.35, 0.0]), 1.95
UX0, UX1 = 0.85, 6.3
SY0, SY1 = -2.0, 2.85                 # u축의 0, 1 라벨과 아래 개수 라벨이 poster의 아래 20% 위에 오도록


def sy(sig):
    return SY0 + (SY1 - SY0) * (np.log(sig) - np.log(SMIN)) / (np.log(SMAX) - np.log(SMIN))


def ux(uu):
    return UX0 + (UX1 - UX0) * uu


def frame(lsig):
    sig = float(np.exp(lsig))
    g = VGroup()
    xs, ys = fft_deriv(x, 0, sig), fft_deriv(y, 0, sig)
    P = CL + SCL * np.stack([xs, ys, 0 * xs], axis=1)
    g.add(ha.polyline(np.vstack([P, P[:1]]), C["main"], 4.5))
    zc = zero_crossings(curvature(x, y, sig))
    for i in zc:
        g.add(Dot(P[i], radius=0.075, color=C["accent"]))
    g.add(Square(0.17, color=C["tangent"], fill_color=C["tangent"], fill_opacity=1, stroke_width=0).move_to(P[0]))
    # CSS image: 지금의 σ보다 위쪽은 흰 사각형으로 가린다(점들은 처음에 한 번 만든다)
    top = SY1 + 0.12
    if sy(sig) + 0.03 < top:
        # 왼쪽 끝(u = 0)의 점 반쪽이 남지 않도록 축보다 0.06 왼쪽부터 가린다(축은 나중에 그린다)
        g.add(Rectangle(width=UX1 - UX0 + 0.26, height=top - sy(sig) - 0.03, stroke_width=0, fill_color=WHITE,
                        fill_opacity=1).move_to([0.5 * (UX0 + UX1) + 0.07, 0.5 * (top + sy(sig) + 0.03), 0]))
    g.add(Line([UX0, sy(sig), 0], [UX1, sy(sig), 0], color=C["tangent"], stroke_width=3))
    g.add(ha.row(r"$\sigma =$", f"{sig:.4f}", size=32).move_to([-5.2, 3.35, 0], aligned_edge=LEFT))
    g.add(da.label(f"inflection points: {len(zc)}", 30, C["accent"]).move_to([-3.75, -2.3, 0]))
    return g


class Anim(da.Scene2D):
    def construct(self):
        ls = ValueTracker(np.log(SMIN))
        axes = VGroup(Line([UX0, SY0, 0], [UX1, SY0, 0], color=C["main"], stroke_width=2.5),
                      Line([UX0, SY0, 0], [UX0, SY1 + 0.15, 0], color=C["main"], stroke_width=2.5))
        for sg in (0.002, 0.01, 0.04):
            axes.add(DashedLine([UX0, sy(sg), 0], [UX1, sy(sg), 0], color=C["aux"], stroke_width=1.5))
            axes.add(da.label(f"{sg:g}", 30, C["aux"]).next_to([UX0, sy(sg), 0], LEFT, buff=0.12))
        axes.add(da.mtex(r"$u$", 32).move_to([UX1 + 0.3, SY0, 0]))
        axes.add(da.label("0", 30, C["aux"]).move_to([UX0, SY0 - 0.35, 0]))
        axes.add(da.label("1", 30, C["aux"]).move_to([UX1, SY0 - 0.35, 0]))
        axes.add(da.mtex(r"$\sigma$", 34).move_to([UX0 - 0.15, SY1 + 0.5, 0]))
        axes.add(da.label("(log scale)", 30, C["aux"]).move_to([UX0 + 1.45, SY1 + 0.5, 0]))
        css_pts = [(ux(u[i]), sy(LEVELS[j])) for j in range(len(LEVELS)) for i in CSS[j]]
        css = ha.dots_mobject(css_pts, size=0.045, color=C["accent"])
        dyn = always_redraw(lambda: frame(ls.get_value()))
        self.add(css, dyn, axes)
        self.wait(1.0)
        for sg, run in ((0.002, 1.5), (0.01, 4.0), (0.04, 3.5), (SMAX, 2.5)):
            self.play(ls.animate.set_value(np.log(sg)), run_time=run, rate_func=linear)
            self.wait(1.8)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)

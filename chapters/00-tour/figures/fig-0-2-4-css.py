"""Figure 0.2.4: curvature scale space (0.2절, In computer vision 상자, [Mokhtarian86]의 방법을 다시 계산).

닫힌 contour: r(φ) = 1 + 0.22 cos 3φ + 0.1 sin(5φ + 0.7) + noise (seed 0, 진폭 0.012의 무작위 Fourier 항)를
arc length에 비례하는 매개변수 u ∈ [0, 1)로 다시 샘플링한다(N = 1024점).
x(u), y(u)를 폭 σ(둘레에 대한 비율)인 주기 Gaussian으로 smoothing하고(FFT), 일반 매개변수 공식
κ_s = (x'y'' − y'x'')/(x'² + y'²)^{3/2}로 signed curvature를 계산해 부호가 바뀌는 u(inflection point)를 찾는다.
위: σ = 0.002, 0.01, 0.04일 때의 contour와 inflection point(주황).
아래: (u, σ) 평면에 inflection point의 위치를 찍은 curvature scale space image.
자기검사: 같은 FFT 미분으로 원의 κ_s = 1/r을 재현한다. inflection point의 개수는 짝수이고,
σ가 가장 클 때 0개(convex)이며, 가장 작은 σ에서의 개수보다 적다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-2-4-css.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
N = 1024
rng = np.random.default_rng(0)


def fft_deriv(f, order, sigma=0.0):
    """주기 [0, 1) 위 함수의 Gaussian smoothing과 미분 (FFT). sigma는 u 단위."""
    k = np.fft.fftfreq(len(f), d=1.0 / len(f))           # 정수 진동수
    F = np.fft.fft(f) * np.exp(-0.5 * (2 * np.pi * k * sigma) ** 2)
    return np.real(np.fft.ifft(F * (2j * np.pi * k) ** order))


def curvature(x, y, sigma):
    x1, y1 = fft_deriv(x, 1, sigma), fft_deriv(y, 1, sigma)
    x2, y2 = fft_deriv(x, 2, sigma), fft_deriv(y, 2, sigma)
    return (x1 * y2 - y1 * x2) / (x1 ** 2 + y1 ** 2) ** 1.5


def smooth(f, sigma):
    return fft_deriv(f, 0, sigma)


def zero_crossings(k):
    s = np.sign(k)
    return np.nonzero(s != np.roll(s, -1))[0]


# 자기검사: 원 ------------------------------------------------------------------------------
u = np.arange(N) / N
for rr in (0.5, 2.0):
    kc = curvature(rr * np.cos(2 * np.pi * u), rr * np.sin(2 * np.pi * u), 0.0)
    assert np.allclose(kc, 1 / rr, atol=1e-8)

# contour 만들기 ---------------------------------------------------------------------------
M = 8192
ph = np.arange(M) / M * 2 * np.pi
r = 1 + 0.22 * np.cos(3 * ph) + 0.1 * np.sin(5 * ph + 0.7)
for kk in range(8, 40):
    r += 0.012 * rng.normal() * np.cos(kk * ph + rng.uniform(0, 2 * np.pi)) / np.sqrt(kk / 8)
X0, Y0 = r * np.cos(ph), r * np.sin(ph)
seg = np.hypot(np.diff(np.append(X0, X0[0])), np.diff(np.append(Y0, Y0[0])))
s = np.concatenate([[0], np.cumsum(seg)])
L = s[-1]
x = np.interp(u * L, s, np.append(X0, X0[0]))
y = np.interp(u * L, s, np.append(Y0, Y0[0]))
# 반시계 방향인지 확인 (signed area > 0)
assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0

sigmas = np.geomspace(0.0015, 0.08, 140)
css_u, css_s, counts = [], [], []
for sg in sigmas:
    zc = zero_crossings(curvature(x, y, sg))
    counts.append(len(zc))
    css_u += list(u[zc])
    css_s += [sg] * len(zc)
counts = np.array(counts)
assert np.all(counts % 2 == 0)
assert counts[-1] == 0 and counts[0] > 6 and counts[0] > counts[len(counts) // 2]
total = np.sum(curvature(x, y, 0.01) * np.hypot(fft_deriv(x, 1, 0.01), fft_deriv(y, 1, 0.01))) / N
assert abs(total - 2 * np.pi) < 1e-6                     # rotation index theorem: ∫κ_s ds = 2π

fig = plt.figure(figsize=(6.4, 5.0))
axT = fig.add_axes([0.0, 0.5, 1.0, 0.5])
axB = fig.add_axes([0.11, 0.08, 0.86, 0.38])
show = (0.002, 0.01, 0.04)
for j, sg in enumerate(show):
    off = np.array([2.75 * j, 0.0])
    xs, ys = smooth(x, sg), smooth(y, sg)
    axT.plot(np.append(xs, xs[0]) + off[0], np.append(ys, ys[0]), color=C["main"], lw=1.3)
    zc = zero_crossings(curvature(x, y, sg))
    axT.plot(xs[zc] + off[0], ys[zc], "o", color=C["accent"], ms=4, zorder=5)
    axT.plot(xs[0] + off[0], ys[0], "s", color=C["tangent"], ms=4, zorder=5)
    axT.text(off[0], 1.55, rf"$\sigma = {sg:g}$", fontsize=11, ha="center")
    axB.axhline(sg, color=C["aux"], lw=0.7, ls=(0, (3, 3)))
dgfig.schematic_axes(axT, (-1.45, 2 * 2.75 + 1.45), (-1.45, 1.8))

axB.plot(css_u, css_s, "o", color=C["accent"], ms=1.6, mew=0, rasterized=True)
axB.set_yscale("log")
axB.set_xlim(0, 1)
axB.set_ylim(sigmas[0], sigmas[-1])
axB.set_yticks([0.002, 0.01, 0.04])
axB.set_yticks([], minor=True)
axB.set_yticklabels([r"$0.002$", r"$0.01$", r"$0.04$"])
axB.set_xlabel(r"$u$")
axB.set_ylabel(r"$\sigma$", rotation=0, labelpad=10)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

dgfig.save(fig, __file__)

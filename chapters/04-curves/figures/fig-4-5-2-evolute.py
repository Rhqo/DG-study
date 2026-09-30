"""그림 4.5.2: 타원의 접촉원과 축폐선 (4.5절, 예 4.5.2(c), 비고 4.5.8, 예 4.5.12).

타원 γ(t) = (2 cos t, sin t) (A = 2, B = 1). 부호곡률 κ_s = AB/(A² sin²t + B² cos²t)^{3/2} (연습 4.4.1).
곡률중심 c(t) = γ + n_s/κ_s = ((A² − B²)/A · cos³t, −(A² − B²)/B · sin³t) = (1.5 cos³t, −3 sin³t).
접촉원: t = 0 (꼭짓점, 반지름 B²/A = 0.5), t = π/6 (반지름 ≈ 1.158, 꼭짓점이 아니므로 곡선을 가로지른다),
t = 5π/6 (반지름 ≈ 1.158). 축폐선(곡률중심의 자취, 청록)은 네 꼭짓점에 대응하는 네 뾰족점을 갖는다. 접촉원은 주황.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-5-2-evolute.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW
A, B = 2.0, 1.0
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def gamma(t):
    return np.stack([A * np.cos(t), B * np.sin(t)], axis=-1)


def d1(t):
    return np.stack([-A * np.sin(t), B * np.cos(t)], axis=-1)


def d2(t):
    return np.stack([-A * np.cos(t), -B * np.sin(t)], axis=-1)


def kappa(t):
    v = d1(t)
    return np.einsum("...i,...i->...", d2(t), v @ J.T) / np.linalg.norm(v, axis=-1) ** 3


def center(t):
    v = d1(t)
    n = (v / np.linalg.norm(v, axis=-1, keepdims=True)) @ J.T
    return gamma(t) + n / kappa(t)[..., None]


ts = np.linspace(0, 2 * np.pi, 1201)
E = center(ts)
# 자기검사: 닫힌 식, 꼭짓점의 반지름, 비꼭짓점에서 가로지름
assert np.allclose(E, np.stack([(A ** 2 - B ** 2) / A * np.cos(ts) ** 3, -(A ** 2 - B ** 2) / B * np.sin(ts) ** 3], 1))
assert abs(1 / kappa(np.array(0.0)) - B ** 2 / A) < 1e-12 and abs(1 / kappa(np.array(np.pi / 2)) - A ** 2 / B) < 1e-12
T_OSC = (0.0, np.pi / 6, 5 * np.pi / 6)


def f_C(t, t0):
    """정리 4.5.5의 f_C = |γ(t) − c|² − ρ²."""
    c0, r0 = center(np.array(t0)), 1 / abs(kappa(np.array(t0)))
    return np.sum((gamma(t) - c0) ** 2, axis=-1) - r0 ** 2


for t0 in T_OSC:
    eps = np.array([-1e-2, 1e-2])
    vals = f_C(t0 + eps, t0)
    if t0 == 0.0:
        assert np.all(vals > 0)                     # 꼭짓점: 곡선이 접촉원 밖에 머문다 (가로지르지 않음)
    else:
        assert vals[0] * vals[1] < 0                # 꼭짓점이 아니면 가로지른다 (비고 4.5.8)

fig, ax = plt.subplots(figsize=(4.6, 4.8))
dgfig.schematic_axes(ax, (-3.3, 3.3), (-3.45, 3.45))
P = gamma(ts)
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=LW["main"], zorder=4)
ax.plot(E[:, 0], E[:, 1], color=C["third"], lw=1.4, zorder=3)
ang = np.linspace(0, 2 * np.pi, 400)
for t0 in T_OSC:
    c0 = center(np.array(t0))
    r0 = 1 / abs(kappa(np.array(t0)))
    p0 = gamma(np.array(t0))
    ax.plot(c0[0] + r0 * np.cos(ang), c0[1] + r0 * np.sin(ang), color=C["accent"], lw=1.1, zorder=2)
    ax.plot(*np.stack([p0, c0]).T, color=C["normal"], lw=1.0, zorder=3)
    ax.plot(*p0, "o", color=C["main"], ms=4.5, zorder=6)
    ax.plot(*c0, "o", color=C["third"], ms=4.5, zorder=6)
p = gamma(np.array(np.pi / 6))
ax.text(p[0] + 0.08, p[1] + 0.12, r"$\gamma(\pi/6)$", fontsize=10.5)
p = gamma(np.array(0.0))
ax.text(p[0] + 0.1, p[1] + 0.1, r"$\gamma(0)$", fontsize=10.5)
ax.text(0.15, -3.25, r"$c(\pi/2)$", fontsize=10.5, color=C["third"])
ax.text(0.15, 3.15, r"$c(3\pi/2)$", fontsize=10.5, color=C["third"])
ax.text(-1.25, 1.6, r"$\gamma$", fontsize=12)
dgfig.save(fig, __file__)

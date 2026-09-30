"""그림 2.1.1: 전미분은 한 점 근처의 선형근사다 (2.1절, 예 2.1.14).

극좌표 사상 P(r, θ) = (r cos θ, r sin θ)와 점 p = (3/2, π/6).
왼쪽: (r, θ) 평면에서 p를 중심으로 한 정사각형 [1, 2] × [π/6 - 1/2, π/6 + 1/2]의 좌표격자.
오른쪽: 그 격자의 P에 의한 상(검정, 휜 격자)과, 아핀 근사
    L(q) = P(p) + DP(p)(q - p)
에 의한 상(주황 점선, 평행사변형 격자). p 가까이에서 두 격자가 거의 겹친다.
파랑 화살표: 왼쪽은 (1/2)e_1, (1/2)e_2, 오른쪽은 그 상 DP(p)((1/2)e_1), DP(p)((1/2)e_2).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-1-1-linear-approximation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
LW = dgfig.LW

r0, t0 = 1.5, np.pi / 6          # 예 2.1.14의 점 p
p = np.array([r0, t0])
d = 0.5                          # 정사각형의 반폭


def P(r, t):
    return np.array([r * np.cos(t), r * np.sin(t)])


DP = np.array([[np.cos(t0), -r0 * np.sin(t0)],
               [np.sin(t0), r0 * np.cos(t0)]])


def L(r, t):
    return P(*p) + DP @ (np.array([r, t]) - p)


# 자기검사 (§12.1): 식 (2.1.4)의 값, det DP = r, 선형근사의 오차가 |h|^2 정도
assert np.allclose(DP, [[np.sqrt(3) / 2, -0.75], [0.5, 3 * np.sqrt(3) / 4]])
assert abs(np.linalg.det(DP) - r0) < 1e-12
for s in (1e-1, 1e-2, 1e-3):
    h = s * np.array([1.0, 1.0])
    err = np.linalg.norm(P(*(p + h)) - L(*(p + h)))
    assert err / np.linalg.norm(h) < 0.1 * s / 1e-1 + 1e-12   # 오차/|h| → 0 (대략 s에 비례)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.3), gridspec_kw={"width_ratios": [1.15, 1.25]})

# 왼쪽: (r, θ) 평면 -----------------------------------------------------------
ax = axL
ax.set_aspect("equal")
ax.set_axis_off()
rs = np.linspace(r0 - d, r0 + d, 6)
ts = np.linspace(t0 - d, t0 + d, 6)
sq = np.array([[r0 - d, t0 - d], [r0 + d, t0 - d], [r0 + d, t0 + d], [r0 - d, t0 + d]])
ax.add_patch(Polygon(sq, closed=True, color=C["region"], alpha=0.25, lw=0, zorder=1))
for r in rs:
    ax.plot([r, r], [t0 - d, t0 + d], color=C["main"], lw=0.6, zorder=2)
for t in ts:
    ax.plot([r0 - d, r0 + d], [t, t], color=C["main"], lw=0.6, zorder=2)
# 축
ax.annotate("", xy=(2.35, 0.0), xytext=(-0.05, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0.0, 1.35), xytext=(0.0, -0.05),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(2.37, 0.0, r"$r$", fontsize=12, va="center")
ax.text(0.0, 1.39, r"$\theta$", fontsize=12, ha="center", va="bottom")
for vec in (np.array([d, 0.0]), np.array([0.0, d])):
    ax.annotate("", xy=p + vec, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=5)
ax.plot(*p, "o", color=C["main"], ms=4, zorder=6)
ax.text(p[0] - 0.06, p[1] - 0.06, r"$p$", fontsize=12, ha="right", va="top", zorder=7,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
ax.text(p[0] + d + 0.04, p[1] + 0.05, r"$\frac{1}{2}e_1$", color=C["tangent"], fontsize=11)
ax.text(p[0] - 0.04, p[1] + d - 0.02, r"$\frac{1}{2}e_2$", color=C["tangent"], fontsize=11, ha="right")
ax.set_xlim(-0.15, 2.55)
ax.set_ylim(-0.15, 1.5)

# 오른쪽: (x, y) 평면 ---------------------------------------------------------
ax = axR
ax.set_aspect("equal")
ax.set_axis_off()
ss = np.linspace(-d, d, 80)
# 휜 격자의 상 영역
edge = np.concatenate([
    np.array([P(r0 - d + (s + d), t0 - d) for s in ss]),
    np.array([P(r0 + d, t0 + s) for s in ss]),
    np.array([P(r0 + d - (s + d), t0 + d) for s in ss]),
    np.array([P(r0 - d, t0 - s) for s in ss]),
])
ax.add_patch(Polygon(edge, closed=True, color=C["region"], alpha=0.25, lw=0, zorder=1))
for r in rs:
    pts = np.array([P(r, t0 + s) for s in ss])
    ax.plot(*pts.T, color=C["main"], lw=0.6, zorder=2)
for t in ts:
    pts = np.array([P(r0 + s, t) for s in ss])
    ax.plot(*pts.T, color=C["main"], lw=0.6, zorder=2)
# 아핀 근사의 격자 (평행사변형)
for r in rs:
    a, b = L(r, t0 - d), L(r, t0 + d)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C["accent"], lw=0.9, ls=(0, (3, 2)), zorder=3)
for t in ts:
    a, b = L(r0 - d, t), L(r0 + d, t)
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C["accent"], lw=0.9, ls=(0, (3, 2)), zorder=3)
# 축
ax.annotate("", xy=(2.25, 0.0), xytext=(-0.1, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0.0, 2.05), xytext=(0.0, -0.3),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(2.28, 0.0, r"$x$", fontsize=12, va="center")
ax.text(0.0, 2.09, r"$y$", fontsize=12, ha="center", va="bottom")
q0 = P(*p)
for k, vec in enumerate((np.array([d, 0.0]), np.array([0.0, d]))):
    ax.annotate("", xy=q0 + DP @ vec, xytext=q0,
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=5)
ax.plot(*q0, "o", color=C["main"], ms=4, zorder=6)
ax.text(q0[0] + 0.06, q0[1] - 0.2, r"$P(p)$", fontsize=11, zorder=7,
        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
tip1 = q0 + DP @ np.array([d, 0.0])
tip2 = q0 + DP @ np.array([0.0, d])
ax.text(tip1[0] + 0.05, tip1[1] - 0.05, r"$DP(p)\,\frac{1}{2}e_1$", color=C["tangent"], fontsize=10)
ax.text(tip2[0] - 0.9, tip2[1] + 0.02, r"$DP(p)\,\frac{1}{2}e_2$", color=C["tangent"], fontsize=10)
ax.set_xlim(-0.35, 2.55)
ax.set_ylim(-0.4, 2.2)

dgfig.map_arrow(fig, axL, axR, r"$P$", xy_from=(0.88, 0.86), xy_to=(0.12, 0.9), rad=-0.3,
                label_offset=(0.0, 0.03))
fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.97, bottom=0.03)
dgfig.save(fig, __file__)

"""그림 2.3.1: 극좌표 사상은 r ≠ 0에서 국소적으로 가역이지만 대역적으로는 단사가 아니다 (2.3절, 예 2.3.10).

P(r, θ) = (r cos θ, r sin θ), det DP = r.
왼쪽 (r, θ) 평면:
    R_1 = [1, 1.7] × [0.3, 1.1],  R_2 = R_1 + (0, 2π)   (하늘색)
    R_0 = [0, 0.6] × [-0.6, 0.6]                       (주황, 왼쪽 변 r = 0은 굵은 주황 선)
오른쪽 (x, y) 평면: P(R_1) = P(R_2) (하늘색 부채꼴 띠), P(R_0) (주황 부채꼴), 변 r = 0은 원점 한 점으로 간다.
θ 축은 r 축보다 1/2로 줄여 그렸다 (매개변수 평면이므로 비율이 의미를 갖지 않는다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-3-1-polar-local-inverse.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
LW = dgfig.LW
TS = 0.5                                  # 왼쪽 그림에서 θ 축의 축척


def P(r, t):
    return np.array([r * np.cos(t), r * np.sin(t)])


R1 = (1.0, 1.7, 0.3, 1.1)
R2 = (1.0, 1.7, 0.3 + 2 * np.pi, 1.1 + 2 * np.pi)
R0 = (0.0, 0.6, -0.6, 0.6)


def rect(ax, R, color, alpha, ls="-", lw=0.9):
    r0, r1, t0, t1 = R
    pts = np.array([[r0, t0], [r1, t0], [r1, t1], [r0, t1]])
    pts[:, 1] *= TS
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=alpha, lw=0, zorder=1))
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=C["main"], lw=lw, ls=ls, zorder=2))


def image(R, n=120):
    r0, r1, t0, t1 = R
    s = np.linspace(0, 1, n)
    edge = np.concatenate([
        np.array([P(r0 + (r1 - r0) * u, t0) for u in s]),
        np.array([P(r1, t0 + (t1 - t0) * u) for u in s]),
        np.array([P(r1 - (r1 - r0) * u, t1) for u in s]),
        np.array([P(r0, t1 - (t1 - t0) * u) for u in s]),
    ])
    return edge


# 자기검사 (§12.1)
e1, e2 = image(R1), image(R2)
assert np.allclose(e1, e2)                                   # P(R_1) = P(R_2): 대역적으로 단사가 아님
assert np.allclose(image(R0)[3 * 120:], 0.0)                 # r = 0인 변은 원점으로
for (r, t) in ((1.3, 0.7), (0.3, 0.2), (1.0, 5.0)):
    J = np.array([[np.cos(t), -r * np.sin(t)], [np.sin(t), r * np.cos(t)]])
    assert abs(np.linalg.det(J) - r) < 1e-12                  # det DP = r

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.2, 3.7), gridspec_kw={"width_ratios": [0.9, 1.0]})

ax = axL
ax.set_aspect("equal")
ax.set_axis_off()
ax.annotate("", xy=(2.1, 0.0), xytext=(-0.25, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0.0, TS * 7.9), xytext=(0.0, -TS * 0.9),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(2.13, 0.0, r"$r$", fontsize=12, va="center")
ax.text(0.0, TS * 7.95, r"$\theta$", fontsize=12, ha="center", va="bottom")
for val, lab in ((np.pi, r"$\pi$"), (2 * np.pi, r"$2\pi$")):
    ax.plot([-0.05, 0.05], [TS * val] * 2, color=C["aux"], lw=0.8)
    ax.text(-0.1, TS * val, lab, fontsize=10, ha="right", va="center", color=C["aux"])
rect(ax, R1, C["region"], 0.35)
rect(ax, R2, C["region"], 0.35, ls="--")
rect(ax, R0, C["accent"], 0.25)
ax.plot([0, 0], [TS * R0[2], TS * R0[3]], color=C["accent"], lw=2.6, zorder=4, solid_capstyle="butt")
ax.text(1.75, TS * 0.7, r"$R_1$", fontsize=12, va="center")
ax.text(1.75, TS * (0.7 + 2 * np.pi), r"$R_2$", fontsize=12, va="center")
ax.text(0.65, TS * (-0.45), r"$R_0$", fontsize=12, va="center", color=C["main"])
ax.set_xlim(-0.45, 2.3)
ax.set_ylim(-TS * 1.0, TS * 8.4)

ax = axR
ax.set_aspect("equal")
ax.set_axis_off()
ax.annotate("", xy=(1.95, 0.0), xytext=(-0.45, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0.0, 1.85), xytext=(0.0, -0.75),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(1.98, 0.0, r"$x$", fontsize=12, va="center")
ax.text(0.0, 1.9, r"$y$", fontsize=12, ha="center", va="bottom")
ax.add_patch(Polygon(e1, closed=True, facecolor=C["region"], alpha=0.35, lw=0, zorder=1))
ax.add_patch(Polygon(e1, closed=True, fill=False, edgecolor=C["main"], lw=0.9, zorder=2))
e0 = image(R0)
ax.add_patch(Polygon(e0, closed=True, facecolor=C["accent"], alpha=0.25, lw=0, zorder=1))
ax.add_patch(Polygon(e0, closed=True, fill=False, edgecolor=C["main"], lw=0.9, zorder=2))
ax.plot(0, 0, "o", color=C["accent"], ms=6, zorder=5)
ax.text(1.05, 1.45, r"$P(R_1) = P(R_2)$", fontsize=11)
ax.text(0.62, -0.55, r"$P(R_0)$", fontsize=11)
ax.set_xlim(-0.6, 2.2)
ax.set_ylim(-0.85, 2.05)

dgfig.map_arrow(fig, axL, axR, r"$P$", xy_from=(0.86, 0.6), xy_to=(0.18, 0.72), rad=-0.3,
                label_offset=(0.0, 0.02))
fig.subplots_adjust(wspace=0.1, left=0.01, right=0.99, top=0.97, bottom=0.02)
dgfig.save(fig, __file__)

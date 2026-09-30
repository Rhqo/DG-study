"""그림 2.6.1: 극좌표 사상은 작은 칸의 넓이를 |det DP| = r배 한다 (2.6절, 예 2.6.10).

왼쪽: (r, θ) 평면의 직사각형 [0, 2] × [0, π/2]를 Δr = 1/4, Δθ = π/16 칸으로 나눈 격자.
      강조한 직사각형 Q = [1, 3/2] × [π/8, 3π/8] (하늘색, 격자 칸 2 × 4개), 중심 c = (5/4, π/4).
오른쪽: 격자의 P에 의한 상(사분원판의 극좌표 격자)과 P(Q)(하늘색 부채꼴 칸).
      주황 점선: 중심에서의 선형근사 P(c) + DP(c)(Q - c) (평행사변형).
      넓이: area P(Q) = (r2² - r1²)(θ2 - θ1)/2 = (5/4)(1/2)(π/4) = 5π/32 = |det DP(c)| vol Q.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-6-1-polar-area.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
LW = dgfig.LW
TS = 1.0

dr, dt = 0.25, np.pi / 16
r1, r2 = 1.0, 1.5
t1, t2 = 2 * dt, 6 * dt
c = np.array([(r1 + r2) / 2, (t1 + t2) / 2])


def P(r, t):
    return np.array([r * np.cos(t), r * np.sin(t)])


DPc = np.array([[np.cos(c[1]), -c[0] * np.sin(c[1])], [np.sin(c[1]), c[0] * np.cos(c[1])]])


def shoelace(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# 부채꼴 칸의 경계를 촘촘히 잡아 넓이 확인 (자기검사, §12.1)
s = np.linspace(0, 1, 4000)
cell = np.concatenate([
    np.array([P(r1 + (r2 - r1) * a, t1) for a in s]),
    np.array([P(r2, t1 + (t2 - t1) * a) for a in s]),
    np.array([P(r2 - (r2 - r1) * a, t2) for a in s]),
    np.array([P(r1, t2 - (t2 - t1) * a) for a in s]),
])
exact = 5 * np.pi / 32
assert abs(shoelace(cell) - exact) < 1e-6
assert abs((r2 ** 2 - r1 ** 2) * (t2 - t1) / 2 - exact) < 1e-15
assert abs(abs(np.linalg.det(DPc)) * (r2 - r1) * (t2 - t1) - exact) < 1e-15
corners = np.array([[r1, t1], [r2, t1], [r2, t2], [r1, t2]])
par = np.array([P(*c) + DPc @ (q - c) for q in corners])
assert abs(shoelace(par) - exact) < 1e-12
s = np.linspace(0, 1, 60)                     # 그리기용 (가벼운 SVG)
cell = np.concatenate([
    np.array([P(r1 + (r2 - r1) * a, t1) for a in s]),
    np.array([P(r2, t1 + (t2 - t1) * a) for a in s]),
    np.array([P(r2 - (r2 - r1) * a, t2) for a in s]),
    np.array([P(r1, t2 - (t2 - t1) * a) for a in s]),
])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw={"width_ratios": [1.05, 1.0]})

ax = axL
ax.set_aspect("equal")
ax.set_axis_off()
ax.annotate("", xy=(2.3, 0), xytext=(-0.05, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0, np.pi / 2 + 0.3), xytext=(0, -0.05),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(2.33, 0, r"$r$", fontsize=12, va="center")
ax.text(0, np.pi / 2 + 0.34, r"$\theta$", fontsize=12, ha="center", va="bottom")
ax.plot([-0.04, 0.04], [np.pi / 2] * 2, color=C["aux"], lw=0.8)
ax.text(-0.08, np.pi / 2, r"$\pi/2$", fontsize=10, ha="right", va="center", color=C["aux"])
ax.text(2.0, -0.09, r"$2$", fontsize=10, ha="center", va="top", color=C["aux"])
ax.add_patch(Polygon(corners, closed=True, facecolor=C["region"], alpha=0.5, lw=0, zorder=1))
for rr in np.arange(0, 2.0 + 1e-9, dr):
    ax.plot([rr, rr], [0, np.pi / 2], color=C["main"], lw=0.5, zorder=2)
for tt in np.arange(0, np.pi / 2 + 1e-9, dt):
    ax.plot([0, 2], [tt, tt], color=C["main"], lw=0.5, zorder=2)
ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=C["main"], lw=1.4, zorder=3))
ax.text(r2 + 0.05, (t1 + t2) / 2, r"$Q$", fontsize=12, va="center")
ax.set_xlim(-0.35, 2.45)
ax.set_ylim(-0.3, np.pi / 2 + 0.55)

ax = axR
ax.set_aspect("equal")
ax.set_axis_off()
ax.annotate("", xy=(2.3, 0), xytext=(-0.05, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(0, 2.3), xytext=(0, -0.05),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(2.33, 0, r"$x$", fontsize=12, va="center")
ax.text(0, 2.34, r"$y$", fontsize=12, ha="center", va="bottom")
ax.add_patch(Polygon(cell, closed=True, facecolor=C["region"], alpha=0.5, lw=0, zorder=1))
tt = np.linspace(0, np.pi / 2, 200)
for rr in np.arange(dr, 2.0 + 1e-9, dr):
    ax.plot(rr * np.cos(tt), rr * np.sin(tt), color=C["main"], lw=0.5, zorder=2)
for ang in np.arange(0, np.pi / 2 + 1e-9, dt):
    ax.plot([0, 2 * np.cos(ang)], [0, 2 * np.sin(ang)], color=C["main"], lw=0.5, zorder=2)
ax.add_patch(Polygon(cell, closed=True, fill=False, edgecolor=C["main"], lw=1.4, zorder=3))
ax.add_patch(Polygon(par, closed=True, fill=False, edgecolor=C["accent"], lw=1.2, ls=(0, (3, 2)), zorder=4))
pc = P(*c)
ax.text(pc[0] + 0.62, pc[1] + 0.12, r"$P(Q)$", fontsize=11, va="center", bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.4))
ax.set_xlim(-0.2, 2.45)
ax.set_ylim(-0.2, 2.45)

dgfig.map_arrow(fig, axL, axR, r"$P$", xy_from=(0.9, 0.85), xy_to=(0.12, 0.9), rad=-0.3,
                label_offset=(0.0, 0.02))
fig.subplots_adjust(wspace=0.1, left=0.01, right=0.99, top=0.96, bottom=0.02)
dgfig.save(fig, __file__)

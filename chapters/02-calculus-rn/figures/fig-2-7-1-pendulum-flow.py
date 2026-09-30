"""그림 2.7.1: 진자 방정식의 해와 초기값에 대한 매끄러운 의존성 (2.7절, 예 2.7.10).

x'' = -sin x 를 1계 연립 (x, v)' = V(x, v) = (v, -sin x)로 쓴다.
검정 가는 선: 에너지 E = v²/2 - cos x 의 등위곡선 (해의 자취). 주황 굵은 선: 분리선 E = 1.
하늘색: 초기값의 원판 D = {|(x, v) - (-1.9, 0)| ≤ 0.25}와 그 흐름 θ_t(D), t = 1, 2, 3 (dgnum.rk4로 계산).
파랑 점선: 원판 중심에서 출발한 해 θ(t, (-1.9, 0)), 0 ≤ t ≤ 3.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-7-1-pendulum-flow.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

import dgnum

C = dgfig.COLORS
LW = dgfig.LW


def V(t, y):
    return np.array([y[1], -np.sin(y[0])])


def E(x, v):
    return 0.5 * v ** 2 - np.cos(x)


def flow(y0, T, n=600):
    ts = np.linspace(0.0, T, n + 1)
    return dgnum.rk4(V, y0, ts)


c0 = np.array([-1.9, 0.0])
rad = 0.25
angs = np.linspace(0, 2 * np.pi, 161)[:-1]
boundary = c0 + rad * np.stack([np.cos(angs), np.sin(angs)], axis=1)
times = (1.0, 2.0, 3.0)
imgs = {0.0: boundary}
for T in times:
    imgs[T] = np.array([flow(b, T)[-1] for b in boundary])
center_path = flow(c0, 3.0, 900)

# 자기검사 (§12.1): 에너지 보존, 흐름의 군 성질 θ_1∘θ_2 = θ_3, 넓이 보존(수치)
for b in boundary[::20]:
    traj = flow(b, 3.0)
    assert np.max(np.abs(E(traj[:, 0], traj[:, 1]) - E(*b))) < 1e-8
y12 = flow(flow(c0, 2.0)[-1], 1.0)[-1]
assert np.allclose(y12, flow(c0, 3.0)[-1], atol=1e-9)


def shoelace(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


a0 = shoelace(imgs[0.0])
for T in times:
    assert abs(shoelace(imgs[T]) - a0) < 2e-3 * a0

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.set_aspect("equal")
ax.set_axis_off()
X, Y = np.meshgrid(np.linspace(-4.0, 4.0, 400), np.linspace(-2.6, 2.6, 300))
levels = [-0.9, -0.6, -0.2, 0.3, 0.7, 1.5, 2.2, 3.0]
ax.contour(X, Y, E(X, Y), levels=levels, colors=C["main"], linewidths=0.5, linestyles="solid", zorder=1)
ax.contour(X, Y, E(X, Y), levels=[1.0], colors=C["accent"], linewidths=1.6, zorder=2)
ax.annotate("", xy=(4.15, 0), xytext=(-4.1, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.annotate("", xy=(0, 2.75), xytext=(0, -2.7),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.text(4.2, 0, r"$x$", fontsize=11, va="center")
ax.text(0, 2.8, r"$v$", fontsize=11, ha="center", va="bottom")
for xv, lab in ((np.pi, r"$\pi$"), (-np.pi, r"$-\pi$")):
    ax.plot([xv, xv], [-0.06, 0.06], color=C["aux"], lw=0.8)
    ax.text(xv, -0.12, lab, fontsize=9, ha="center", va="top", color=C["aux"])
ax.plot(*center_path.T, color=C["tangent"], lw=1.0, ls=(0, (3, 2)), zorder=3)
for T, lab, off in ((0.0, r"$D$", (-0.55, -0.35)), (1.0, r"$\theta_1(D)$", (0.15, -0.55)),
                    (2.0, r"$\theta_2(D)$", (-0.35, 0.32)), (3.0, r"$\theta_3(D)$", (0.1, 0.35))):
    pts = imgs[T]
    ax.add_patch(Polygon(pts, closed=True, facecolor=C["region"], alpha=0.45, lw=0, zorder=4))
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=C["main"], lw=0.9, zorder=5))
    cx, cy = pts.mean(axis=0)
    ax.text(cx + off[0], cy + off[1], lab, fontsize=10, zorder=6)
ax.set_xlim(-4.3, 4.45)
ax.set_ylim(-2.75, 3.0)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

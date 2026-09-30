"""그림 2.2.1: 같은 속도로 p를 지나는 두 곡선은 같은 속도로 F(p)를 지나는 곡선으로 간다 (2.2절, 따름정리 2.2.2).

극좌표 사상 P(r, θ) = (r cos θ, r sin θ), p = (3/2, π/6), v = (2/5, 3/5).
    α(t) = p + t v                 (직선)
    β(t) = p + t v + t^2 w,  w = (-4/5, 3/5)   (포물선)
둘 다 α(0) = β(0) = p, α'(0) = β'(0) = v. 상 P∘α, P∘β는 P(p)를 같은 속도 DP(p)v로 지난다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-2-1-curves-velocity.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

p = np.array([1.5, np.pi / 6])
v = np.array([0.4, 0.6])
w = np.array([-0.8, 0.6])


def P(q):
    q = np.atleast_2d(q)
    return np.stack([q[:, 0] * np.cos(q[:, 1]), q[:, 0] * np.sin(q[:, 1])], axis=1)


DP = np.array([[np.cos(p[1]), -p[0] * np.sin(p[1])],
               [np.sin(p[1]), p[0] * np.cos(p[1])]])

ts = np.linspace(-0.7, 0.7, 301)
alpha = p + np.outer(ts, v)
beta = p + np.outer(ts, v) + np.outer(ts ** 2, w)
Pa, Pb = P(alpha), P(beta)

# 자기검사 (§12.1): 두 상 곡선의 t = 0에서의 속도가 모두 DP(p)v (중심차분)
eps = 1e-6
for curve in (lambda t: P(p + t * v)[0], lambda t: P(p + t * v + t ** 2 * w)[0]):
    vel = (curve(eps) - curve(-eps)) / (2 * eps)
    assert np.allclose(vel, DP @ v, atol=1e-8)
assert np.allclose(DP @ v, [0.4 * np.sqrt(3) / 2 - 0.45, 0.2 + 0.45 * np.sqrt(3)], atol=1e-12)


def arrow(ax, base, vec, color, lw=None, z=6):
    ax.annotate("", xy=base + vec, xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def axes_arrows(ax, x0, x1, y0, y1, xl, yl):
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
    ax.annotate("", xy=(x0, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
    ax.text(x1 + 0.03, y0, xl, fontsize=12, va="center")
    ax.text(x0, y1 + 0.03, yl, fontsize=12, ha="center", va="bottom")


fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1.1]})

ax = axL
ax.set_aspect("equal")
ax.set_axis_off()
axes_arrows(ax, 0.0, 2.3, 0.0, 1.45, r"$r$", r"$\theta$")
ax.plot(*alpha.T, color=C["main"], lw=LW["main"], zorder=3)
ax.plot(*beta.T, color=C["accent"], lw=LW["main"], zorder=3)
arrow(ax, p, v, C["tangent"])
ax.plot(*p, "o", color=C["main"], ms=4, zorder=7)
ax.text(p[0] + 0.07, p[1] - 0.12, r"$p$", fontsize=12)
ax.text(p[0] + v[0] + 0.03, p[1] + v[1] - 0.02, r"$v$", color=C["tangent"], fontsize=12)
ax.text(*(alpha[-1] + [0.03, -0.05]), r"$\alpha$", fontsize=12)
ax.text(*(beta[-1] + [0.03, 0.0]), r"$\beta$", color=C["accent"], fontsize=12)
ax.set_xlim(-0.1, 2.45)
ax.set_ylim(-0.1, 1.55)

ax = axR
ax.set_aspect("equal")
ax.set_axis_off()
axes_arrows(ax, 0.0, 2.2, 0.0, 1.95, r"$x$", r"$y$")
ax.plot(*Pa.T, color=C["main"], lw=LW["main"], zorder=3)
ax.plot(*Pb.T, color=C["accent"], lw=LW["main"], zorder=3)
q0 = P(p)[0]
arrow(ax, q0, DP @ v, C["tangent"])
ax.plot(*q0, "o", color=C["main"], ms=4, zorder=7)
ax.text(q0[0] + 0.08, q0[1] - 0.12, r"$P(p)$", fontsize=11)
tip = q0 + DP @ v
ax.text(tip[0] - 0.62, tip[1] + 0.03, r"$DP(p)v$", color=C["tangent"], fontsize=11)
ax.text(*(Pa[-1] + [0.05, -0.02]), r"$P\circ\alpha$", fontsize=11)
ax.text(*(Pb[-1] + [-0.35, 0.06]), r"$P\circ\beta$", color=C["accent"], fontsize=11)
ax.set_xlim(-0.2, 2.45)
ax.set_ylim(-0.2, 2.1)

dgfig.map_arrow(fig, axL, axR, r"$P$", xy_from=(0.88, 0.88), xy_to=(0.1, 0.92), rad=-0.3,
                label_offset=(0.0, 0.03))
fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.95, bottom=0.03)
dgfig.save(fig, __file__)

"""그림 11.1.1: 위상다양체의 차트 (11.1절, 정의 11.1.1, 정의 11.1.13). (개념도)

왼쪽: 얼룩 모양의 위상공간 M, 점 p, p의 열린 근방 U(하늘색).
오른쪽: R^n (n = 2로 그림)의 열린집합 Û = φ(U)(하늘색)와 φ(p), 좌표격자.
φ(U)의 좌표격자를 φ^{-1}로 되돌린 곡선들을 U 안에 그린다. 이때 φ^{-1}은 원판 Û에서 U로 가는
명시적인 위상동형사상(극좌표로 반지름 방향을 늘이는 사상)으로 정한다. 두 영역의 모양은 개념도다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-1-1-chart.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
LW = dgfig.LW

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]})

# ---- 왼쪽: M과 U
ax = axL
dgfig.schematic_axes(ax, (-1.45, 1.45), (-1.4, 1.25))
dgfig.blob(ax, center=(0, 0), radius=1.05, seed=11, amp=0.16, fill="surface", fill_alpha=0.35)

# U: 점 c0 둘레의 얼룩. 경계는 rho(t) = r0 * (1 + 0.18 cos 3t + 0.08 sin 2t)로 준다.
c0 = np.array([0.28, 0.05])
r0 = 0.52


def rho(t):
    return r0 * (1 + 0.18 * np.cos(3 * t + 0.4) + 0.08 * np.sin(2 * t))


def phi_inv(w):
    """Û = 단위원판(반지름 1) → U. 극좌표 (s, t)에서 반지름을 rho(t)배 한다 (연속 전단사, 역도 연속)."""
    w = np.asarray(w, float)
    s = np.hypot(w[..., 0], w[..., 1])
    t = np.arctan2(w[..., 1], w[..., 0])
    return np.stack([c0[0] + s * rho(t) * np.cos(t), c0[1] + s * rho(t) * np.sin(t)], axis=-1)


tt = np.linspace(0, 2 * np.pi, 400)
Ub = phi_inv(np.stack([np.cos(tt), np.sin(tt)], axis=1) * 0.999)
ax.add_patch(Polygon(Ub, closed=True, facecolor=C["region"], alpha=0.35, lw=0, zorder=3))
ax.add_patch(Polygon(Ub, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=4))
# 좌표격자를 U로 되돌린 곡선
for a in np.linspace(-0.75, 0.75, 7):
    h = np.sqrt(max(1 - a * a, 0)) * 0.999
    s = np.linspace(-h, h, 120)
    ax.plot(*phi_inv(np.stack([np.full_like(s, a), s], axis=1)).T, color=C["tangent"], lw=0.45, zorder=5)
    ax.plot(*phi_inv(np.stack([s, np.full_like(s, a)], axis=1)).T, color=C["tangent"], lw=0.45, zorder=5)
p_img = np.array([0.15, -0.2])
p = phi_inv(p_img)
ax.plot(*p, "o", ms=4, color=C["main"], zorder=7)
ax.text(p[0] + 0.05, p[1] + 0.06, r"$p$", fontsize=12, zorder=8)
ax.text(c0[0] - 0.62, c0[1] + 0.36, r"$U$", fontsize=13, zorder=8)
ax.text(-1.2, 0.72, r"$M$", fontsize=14)

# ---- 오른쪽: R^n의 Û
ax = axR
dgfig.schematic_axes(ax, (-1.55, 1.55), (-1.45, 1.4))
ax.annotate("", xy=(1.5, -1.15), xytext=(-1.45, -1.15),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.annotate("", xy=(-1.45, 1.3), xytext=(-1.45, -1.15),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.8, mutation_scale=9))
ax.text(1.5, -1.28, r"$x^1$", fontsize=11, ha="right", va="top")
ax.text(-1.38, 1.3, r"$x^2$", fontsize=11, va="top")
disk = np.stack([np.cos(tt), np.sin(tt)], axis=1)
ax.add_patch(Polygon(disk, closed=True, facecolor=C["region"], alpha=0.35, lw=0, zorder=1))
ax.add_patch(Polygon(disk, closed=True, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5)), zorder=2))
for a in np.linspace(-0.75, 0.75, 7):
    h = np.sqrt(max(1 - a * a, 0))
    ax.plot([a, a], [-h, h], color=C["tangent"], lw=0.45, zorder=3)
    ax.plot([-h, h], [a, a], color=C["tangent"], lw=0.45, zorder=3)
ax.plot(*p_img, "o", ms=4, color=C["main"], zorder=7)
ax.text(p_img[0] + 0.06, p_img[1] + 0.07, r"$\varphi(p)$", fontsize=11, zorder=8)
ax.text(0.62, 0.86, r"$\hat U = \varphi(U)$", fontsize=11)
ax.text(1.05, -0.95, r"$\mathbb{R}^n$", fontsize=12)

# 자기검사 (§12.1): phi_inv는 원판의 경계를 U의 경계로, 중심을 c0로 보낸다. 서로 다른 점은 서로 다른 점으로.
assert np.allclose(phi_inv(np.array([0.0, 0.0])), c0)
assert np.all(rho(tt) > 0)
w = np.random.default_rng(0).uniform(-0.7, 0.7, size=(500, 2))
w = w[np.hypot(*w.T) < 0.99]
img = phi_inv(w)
d_in = np.linalg.norm(w[:, None] - w[None], axis=-1)
d_out = np.linalg.norm(img[:, None] - img[None], axis=-1)
assert np.all((d_in < 1e-12) == (d_out < 1e-12))          # 단사 (표본)

dgfig.map_arrow(fig, axL, axR, r"$\varphi$", xy_from=(0.78, 0.72), xy_to=(0.25, 0.78), rad=-0.3)
fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.95, bottom=0.02)
dgfig.save(fig, __file__)

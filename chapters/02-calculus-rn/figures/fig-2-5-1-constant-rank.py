"""그림 2.5.1: 계수가 1로 일정한 사상 F(x, y) = (cos(x + y), sin(x + y)) (2.5절, 예 2.5.5).

왼쪽: 정의역의 올(fiber) x + y = c (c = -2.5, -2, ..., 2.5, 회색 실선)과 p = (0.3, 0.2)를 지나는 올
      F^{-1}(F(p)) ∩ 그림 영역 = {x + y = 0.5} (주황). 파랑: 올에 수직인 단위벡터 (1, 1)/√2의 0.6배.
      오른쪽 파랑은 그 상 DF(p)((1, 1)/√2)의 0.6배.
오른쪽: 상은 단위원(검정). 옅은 점선은 극좌표 격자(반직선과 동심원)로, 표적 쪽 좌표 ψ = (각, 반지름 - 1)의 좌표선이다.
      F(p) = (cos 0.5, sin 0.5).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-5-1-constant-rank.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

p = np.array([0.3, 0.2])
s0 = p.sum()


def F(x, y):
    return np.array([np.cos(x + y), np.sin(x + y)])


def DF(x, y):
    s = x + y
    return np.array([[-np.sin(s), -np.sin(s)], [np.cos(s), np.cos(s)]])


# 자기검사 (§12.1): 모든 점에서 계수 1, 올 위에서 F가 상수, F∘φ^{-1}(u, v) = P(1, u)
rng = np.random.default_rng(0)
for xy in rng.uniform(-3, 3, (50, 2)):
    assert np.linalg.matrix_rank(DF(*xy), tol=1e-10) == 1
for t in np.linspace(-1, 1, 5):
    assert np.allclose(F(p[0] + t, p[1] - t), F(*p))
for u, v in rng.uniform(-1, 1, (20, 2)):
    assert np.allclose(F(u - v, v), [np.cos(u), np.sin(u)])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.4, 3.2))

ax = axL
L = 1.6
dgfig.schematic_axes(ax, (-L - 0.1, L + 0.3), (-L - 0.1, L + 0.3))
ax.annotate("", xy=(L + 0.1, 0), xytext=(-L, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.annotate("", xy=(0, L + 0.1), xytext=(0, -L),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.text(L + 0.14, 0, r"$x$", fontsize=11, va="center")
ax.text(0, L + 0.14, r"$y$", fontsize=11, ha="center", va="bottom")
xs = np.linspace(-L, L, 200)
for c in np.arange(-2.5, 2.51, 0.5):
    ys = c - xs
    m = np.abs(ys) <= L
    col, lw = (C["accent"], 2.0) if abs(c - s0) < 1e-9 else ("#9A9A9A", 0.7)
    ax.plot(np.where(m, xs, np.nan), np.where(m, ys, np.nan), color=col, lw=lw, zorder=2 if lw > 1 else 1)
ax.annotate("", xy=p + 0.6 * np.array([1, 1]) / np.sqrt(2), xytext=p,
            arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=LW["vector"], mutation_scale=11,
                            shrinkA=0, shrinkB=0), zorder=5)
ax.plot(*p, "o", color=C["main"], ms=4, zorder=6)
ax.text(p[0] + 0.08, p[1] - 0.25, r"$p$", fontsize=12)
ax.text(-1.55, 1.75, r"$F^{-1}(F(p))$", fontsize=10, color=C["accent"])

ax = axR
dgfig.schematic_axes(ax, (-1.75, 1.95), (-1.75, 1.85))
ax.annotate("", xy=(1.7, 0), xytext=(-1.6, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.annotate("", xy=(0, 1.7), xytext=(0, -1.6),
            arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
ax.text(1.74, 0, r"$x$", fontsize=11, va="center")
ax.text(0, 1.74, r"$y$", fontsize=11, ha="center", va="bottom")
tt = np.linspace(0, 2 * np.pi, 400)
for rr in (0.6, 1.4):
    ax.plot(rr * np.cos(tt), rr * np.sin(tt), color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=1)
for ang in np.arange(0, 2 * np.pi, np.pi / 6):
    ax.plot([0.25 * np.cos(ang), 1.55 * np.cos(ang)], [0.25 * np.sin(ang), 1.55 * np.sin(ang)],
            color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.plot(np.cos(tt), np.sin(tt), color=C["main"], lw=LW["main"], zorder=3)
q = F(*p)
v = DF(*p) @ (np.array([1, 1]) / np.sqrt(2))
ax.annotate("", xy=q + 0.6 * v, xytext=q,
            arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=LW["vector"], mutation_scale=11,
                            shrinkA=0, shrinkB=0), zorder=5)
ax.plot(*q, "o", color=C["accent"], ms=5, zorder=6)
ax.text(q[0] + 0.1, q[1] - 0.15, r"$F(p)$", fontsize=11)
ax.text(-1.2, -1.2, r"$F(\mathbb{R}^2)$", fontsize=11, ha="right")

dgfig.map_arrow(fig, axL, axR, r"$F$", xy_from=(0.9, 0.82), xy_to=(0.1, 0.85), rad=-0.3,
                label_offset=(0.0, 0.02))
fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.96, bottom=0.02)
dgfig.save(fig, __file__)

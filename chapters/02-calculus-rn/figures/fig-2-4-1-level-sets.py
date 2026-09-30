"""그림 2.4.1: 등위집합과 음함수 정리 (2.4절, 예 2.4.5, 비예 2.4.3, 예 2.4.11).

(a) 원 F(x, y) = x^2 + y^2 = 1.
    A = (0.6, 0.8): ∂_y F(A) = 1.6 ≠ 0 이므로 A 근처에서 y = √(1 - x^2) (주황, x ∈ [0.2, 0.8]).
    B = (1, 0): ∂_y F(B) = 0 이지만 ∂_x F(B) = 2 ≠ 0 이므로 B 근처에서 x = √(1 - y^2) (주황, y ∈ [-0.4, 0.4]).
    주홍 화살표: 기울기 grad F의 1/4배.
(b) 안장면의 높이함수 f(x, y) = x^2 - y^2의 등위곡선 c = -2, -1, 0, 1, 2.
    q = (√2, 1) (c = 1): 주홍 = grad f(q)의 1/4배, 파랑 = ker Df(q)의 벡터 (1, √2)의 1/2배.
    원점: grad f = 0, 등위집합 c = 0 은 두 직선 y = ±x (그래프가 아니다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/02-calculus-rn/figures/fig-2-4-1-level-sets.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW
S_GRAD = 0.25     # 기울기 벡터의 배율 (캡션에 적음)


def arrow(ax, base, vec, color, lw=None, z=6):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def axes(ax, lim, xlim=None):
    xlim = xlim or lim
    ax.annotate("", xy=(xlim, 0), xytext=(-lim, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
    ax.annotate("", xy=(0, lim), xytext=(0, -lim),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8), zorder=0)
    ax.text(xlim + 0.04, 0, r"$x$", fontsize=11, va="center")
    ax.text(0, lim + 0.04, r"$y$", fontsize=11, ha="center", va="bottom")


# 자기검사
A = np.array([0.6, 0.8]); B = np.array([1.0, 0.0]); q = np.array([np.sqrt(2), 1.0])
assert abs(A @ A - 1) < 1e-12 and abs(B @ B - 1) < 1e-12
assert abs(q[0] ** 2 - q[1] ** 2 - 1) < 1e-12
grad_q = np.array([2 * q[0], -2 * q[1]])
tan_q = np.array([1.0, np.sqrt(2)])
assert abs(grad_q @ tan_q) < 1e-12                      # grad ⊥ 접선 (명제 2.4.10)
hq = lambda x: np.sqrt(x ** 2 - 1)
assert abs((hq(q[0] + 1e-6) - hq(q[0] - 1e-6)) / 2e-6 - np.sqrt(2)) < 1e-6   # h'(√2) = √2

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 3.3))

# (a) 원 ------------------------------------------------------------------------
ax = axA
dgfig.schematic_axes(ax, (-1.45, 1.95), (-1.35, 1.45))
axes(ax, 1.35, xlim=1.7)
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color=C["main"], lw=1.2, zorder=2)
xs = np.linspace(0.2, 0.8, 100)
ax.plot(xs, np.sqrt(1 - xs ** 2), color=C["accent"], lw=3.2, alpha=0.9, zorder=3, solid_capstyle="round")
ys = np.linspace(-0.4, 0.4, 100)
ax.plot(np.sqrt(1 - ys ** 2), ys, color=C["accent"], lw=3.2, alpha=0.9, zorder=3, solid_capstyle="round")
for P0 in (A, B):
    arrow(ax, P0, S_GRAD * 2 * P0, C["normal"])
    ax.plot(*P0, "o", color=C["main"], ms=4, zorder=7)
ax.text(A[0] - 0.2, A[1] - 0.22, r"$A$", fontsize=12)
ax.text(B[0] - 0.22, B[1] - 0.2, r"$B$", fontsize=12)
ax.text(0.05, 1.12, r"$y = g(x)$", fontsize=10, color=C["accent"])
ax.text(1.08, -0.52, r"$x = \tilde g(y)$", fontsize=10, color=C["accent"])
ax.text(-1.4, 1.3, "(a)", fontsize=11)

# (b) 안장면의 등위곡선 ---------------------------------------------------------------
ax = axB
lim = 2.0
dgfig.schematic_axes(ax, (-lim - 0.1, lim + 0.25), (-lim - 0.1, lim + 0.35))
axes(ax, lim)
for c in (-2.0, -1.0, 1.0, 2.0):
    s = np.linspace(-1.6, 1.6, 300)
    if c > 0:
        for sgn in (1, -1):
            X, Y = sgn * np.sqrt(c) * np.cosh(s), np.sqrt(c) * np.sinh(s)
            m = (np.abs(X) <= lim) & (np.abs(Y) <= lim)
            ax.plot(np.where(m, X, np.nan), np.where(m, Y, np.nan), color=C["main"], lw=1.0, zorder=2)
    else:
        for sgn in (1, -1):
            X, Y = np.sqrt(-c) * np.sinh(s), sgn * np.sqrt(-c) * np.cosh(s)
            m = (np.abs(X) <= lim) & (np.abs(Y) <= lim)
            ax.plot(np.where(m, X, np.nan), np.where(m, Y, np.nan), color=C["main"], lw=1.0, ls=(0, (4, 2)), zorder=2)
u = np.linspace(-lim, lim, 10)
ax.plot(u, u, color=C["accent"], lw=1.6, zorder=3)
ax.plot(u, -u, color=C["accent"], lw=1.6, zorder=3)
ax.plot(0, 0, "o", color=C["accent"], ms=5, zorder=7)
arrow(ax, q, S_GRAD * grad_q, C["normal"])
arrow(ax, q, 0.5 * tan_q, C["tangent"])
ax.plot(*q, "o", color=C["main"], ms=4, zorder=7)
ax.text(q[0] - 0.3, q[1] + 0.02, r"$q$", fontsize=12)
ax.text(0.5, -0.33, r"$c=1$", fontsize=10)
ax.text(0.08, 1.08, r"$c=-1$", fontsize=10)
ax.text(-2.1, 2.15, "(b)", fontsize=11)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.98, bottom=0.02)
dgfig.save(fig, __file__)

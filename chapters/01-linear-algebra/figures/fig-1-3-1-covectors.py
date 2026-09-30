"""그림 1.3.1: 여벡터 = 평행한 등위선들의 모임 (1.3절, 예 1.3.2와 예 1.3.6).

(a) ω(x) = x^1 + x^2의 등위선 {ω = c}, c = -2, …, 4. v = (1, 2)는 등위선 3칸을 건너므로 ω(v) = 3.
    b_2 = (-1, 1)은 등위선 {ω = 0}을 따라 놓여 ω(b_2) = 0.
(b) 기준 기저 B = (b_1, b_2)의 쌍대기저 β^1 = (x^1 + x^2)/3, β^2 = (-x^1 + 2x^2)/3의 등위선.
    β^1의 등위선은 b_2와 평행하고 b_1의 끝이 {β^1 = 1} 위에 있다. v = b_1 + b_2는 {β^1 = 1}과 {β^2 = 1}의 교점.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-3-1-covectors.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

b1 = np.array([2.0, 1.0])
b2 = np.array([-1.0, 1.0])
v = b1 + b2
B = np.column_stack([b1, b2])
Binv = np.linalg.inv(B)
beta1, beta2 = Binv[0], Binv[1]          # 명제 1.3.7: B^{-1}의 행
omega = np.array([1.0, 1.0])             # ω = e^1 + e^2

# 자기검사
assert np.allclose(beta1, [1 / 3, 1 / 3]) and np.allclose(beta2, [-1 / 3, 2 / 3])
assert np.allclose([beta1 @ b1, beta1 @ b2, beta2 @ b1, beta2 @ b2], [1, 0, 0, 1])
assert np.isclose(omega @ v, 3) and np.isclose(omega @ b2, 0)
assert np.allclose(omega, 3 * beta1)     # ω = 3β^1 (예 1.4.8에서 다시 쓴다)


def arrow(ax, base, vec, color, lw=None, z=6):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def level_lines(ax, a, cs, color, lw=0.9, ls="-", xlim=(-4, 4)):
    """선형함수 a·x = c의 등위선 (a는 행벡터)."""
    a = np.asarray(a, float)
    d = np.array([-a[1], a[0]])                       # 등위선 방향
    base0 = a / (a @ a)                               # a·x = 1인 점
    for c in cs:
        P = c * base0 + np.outer([-10, 10], d / np.linalg.norm(d))
        ax.plot(P[:, 0], P[:, 1], color=color, lw=lw, ls=ls, zorder=2)


def frame(ax, xlim, ylim, tag):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=7)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)


XL, YL = (-2.3, 3.2), (-1.4, 3.1)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.95))

# (a) 여벡터 ω = x^1 + x^2 -----------------------------------------------------
ax = axs[0]
frame(ax, XL, YL, "(a)")
level_lines(ax, omega, range(-2, 5), C["normal"])
arrow(ax, (0, 0), v, C["main"], lw=1.8)
arrow(ax, (0, 0), b2, C["tangent"])
WB = dict(facecolor="white", edgecolor="none", pad=0.4)
for c in (1, 2, 3):
    q = c * v / 3 + np.array([0.14, -0.2])          # v가 {ω = c}를 건너는 점의 오른쪽
    ax.text(q[0], q[1], rf"$\omega = {c}$", color=C["normal"], fontsize=9.5, ha="left", va="center",
            bbox=WB, zorder=4)
ax.text(0.55, -0.75, r"$\omega = 0$", color=C["normal"], fontsize=9.5, ha="left", va="center", bbox=WB, zorder=4)
ax.text(v[0] - 0.55, v[1] + 0.1, r"$v$", color=C["main"], fontsize=12)
ax.text(b2[0] - 0.35, b2[1] + 0.12, r"$b_2$", color=C["tangent"], fontsize=12)
ax.set_xlim(*XL); ax.set_ylim(*YL)

# (b) 쌍대기저 β^1, β^2 --------------------------------------------------------
ax = axs[1]
frame(ax, XL, YL, "(b)")
level_lines(ax, beta1, range(-2, 4), C["normal"])
level_lines(ax, beta2, range(-2, 4), C["third"], ls=(0, (4, 2)))
arrow(ax, (0, 0), b1, C["tangent"])
arrow(ax, (0, 0), b2, C["tangent"])
arrow(ax, (0, 0), v, C["main"], lw=1.8)
ax.plot(*v, "o", color=C["main"], ms=3.5, zorder=8)
ax.text(b1[0] - 0.45, b1[1] - 0.72, r"$b_1$", color=C["tangent"], fontsize=12)
ax.text(b2[0] - 0.4, b2[1] + 0.22, r"$b_2$", color=C["tangent"], fontsize=12, bbox=WB, zorder=5)
ax.text(v[0] - 0.42, v[1] + 0.12, r"$v$", color=C["main"], fontsize=12, bbox=WB, zorder=5)
# 등위선 라벨: {β^1 = 1}은 b_1을 지나고 b_2와 평행, {β^2 = 1}은 b_2를 지나고 b_1과 평행
q1 = b1 - 0.6 * b2
ax.text(q1[0], q1[1], r"$\beta^1 = 1$", color=C["normal"], fontsize=10, rotation=-45,
        ha="center", va="center", bbox=WB, zorder=4)
q2 = b2 + 1.5 * b1
ax.text(q2[0], q2[1], r"$\beta^2 = 1$", color=C["third"], fontsize=10,
        rotation=np.degrees(np.arctan2(b1[1], b1[0])), ha="center", va="center", bbox=WB, zorder=4)
ax.set_xlim(*XL); ax.set_ylim(*YL)

fig.subplots_adjust(wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

"""그림 1.5.1: 그람-슈미트와 내적이 정하는 단위원 (1.5절, 예 1.5.2, 예 1.5.11, 연습 없음).

(a) 유클리드 내적으로 기준 기저 b_1 = (2,1), b_2 = (-1,1)에 그람-슈미트를 적용한다.
    u_1 = b_1/|b_1| = (2,1)/√5,  b_2 - <b_2,u_1>u_1 = (-3/5, 6/5),  u_2 = (-1,2)/√5.
(b) 내적 g(x, y) = x^T G y, G = [[2,1],[1,2]]의 단위원 {g(x,x) = 1}(타원)과 유클리드 단위원(점선),
    그리고 (e_1, e_2)에 g로 그람-슈미트를 적용해 얻은 g-정규직교기저
    u_1 = e_1/√2,  u_2 = (-1/2, 1)/√(3/2).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-5-1-inner-products.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

# (a) 유클리드 그람-슈미트
b1 = np.array([2.0, 1.0])
b2 = np.array([-1.0, 1.0])
u1 = b1 / np.linalg.norm(b1)
proj = (b2 @ u1) * u1
w2 = b2 - proj
u2 = w2 / np.linalg.norm(w2)
assert np.allclose(proj, [-0.4, -0.2]) and np.allclose(w2, [-0.6, 1.2])
assert np.allclose(u2, np.array([-1.0, 2.0]) / np.sqrt(5))
assert abs(u1 @ u2) < 1e-12 and np.isclose(u1 @ u1, 1) and np.isclose(u2 @ u2, 1)

# (b) 내적 G
G = np.array([[2.0, 1.0], [1.0, 2.0]])
g = lambda x, y: x @ G @ y
e1, e2 = np.eye(2)
f1 = e1 / np.sqrt(g(e1, e1))
h2 = e2 - g(e2, f1) * f1
f2 = h2 / np.sqrt(g(h2, h2))
assert np.allclose(h2, [-0.5, 1.0])
assert np.isclose(g(f1, f1), 1) and np.isclose(g(f2, f2), 1) and abs(g(f1, f2)) < 1e-12
assert abs(f1 @ f2) > 0.1   # 유클리드로는 직교가 아니다
assert np.all(np.linalg.eigvalsh(G) > 0)


def arrow(ax, base, vec, color, lw=None, z=6):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def frame(ax, xlim, ylim, tag):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=7)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)


def right_angle(ax, p, d1, d2, s=0.13, color=C["aux"]):
    d1 = d1 / np.linalg.norm(d1); d2 = d2 / np.linalg.norm(d2)
    pts = np.array([p + s * d1, p + s * (d1 + d2), p + s * d2])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=0.8, zorder=5)


ts = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(ts), np.sin(ts)], axis=1)
WB = dict(facecolor="white", edgecolor="none", pad=0.3)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1))

# (a) ------------------------------------------------------------------------
ax = axs[0]
XL, YL = (-1.5, 2.4), (-1.1, 1.55)
frame(ax, XL, YL, "(a)")
ax.plot(circle[:, 0], circle[:, 1], color=C["aux"], lw=0.8, ls=(0, (3, 2.5)), zorder=1)
arrow(ax, (0, 0), b1, C["tangent"])
arrow(ax, (0, 0), b2, C["tangent"])
ax.plot(*np.array([[0, 0], proj]).T, color=C["aux"], lw=1.2, zorder=3)
ax.plot(*np.array([proj, b2]).T, color=C["aux"], lw=0.9, ls="--", zorder=3)
arrow(ax, (0, 0), w2, C["accent"], lw=1.0)
arrow(ax, (0, 0), u1, C["accent"], lw=2.0, z=8)
arrow(ax, (0, 0), u2, C["accent"], lw=2.0, z=8)
right_angle(ax, np.zeros(2), u1, u2)
right_angle(ax, proj, -proj, b2 - proj)
ax.text(*(b1 + [0.05, -0.28]), r"$b_1$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(b2 + [-0.35, 0.02]), r"$b_2$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(u1 + [0.02, -0.32]), r"$u_1$", color=C["accent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(u2 + [0.1, -0.12]), r"$u_2$", color=C["accent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(w2 + [0.1, 0.08]), r"$b_2 - \langle b_2, u_1\rangle u_1$", color=C["accent"], fontsize=9.5,
        bbox=WB, zorder=9, ha="left")
ax.set_xlim(*XL); ax.set_ylim(*YL)

# (b) ------------------------------------------------------------------------
ax = axs[1]
XL, YL = (-1.35, 1.35), (-1.0, 1.25)
frame(ax, XL, YL, "(b)")
ax.plot(circle[:, 0], circle[:, 1], color=C["aux"], lw=0.8, ls=(0, (3, 2.5)), zorder=1)
# g(x,x) = 1: x = L^{-T} (cos, sin) with G = L L^T (Cholesky)
L = np.linalg.cholesky(G)
ell = circle @ np.linalg.inv(L)       # 행벡터 x = c^T L^{-1} → x G x^T = |c|^2 = 1
assert np.allclose(np.einsum("ij,jk,ik->i", ell, G, ell), 1.0)
ax.fill(ell[:, 0], ell[:, 1], color=C["region"], alpha=0.25, lw=0, zorder=1)
ax.plot(ell[:, 0], ell[:, 1], color=C["main"], lw=1.4, zorder=2)
arrow(ax, (0, 0), f1, C["accent"], lw=2.0)
arrow(ax, (0, 0), f2, C["accent"], lw=2.0)
ax.text(*(f1 + [-0.05, -0.2]), r"$u_1$", color=C["accent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(f2 + [0.06, 0.02]), r"$u_2$", color=C["accent"], fontsize=12, bbox=WB, zorder=9)
ax.text(0.62, 0.78, r"$g(x,x) = 1$", color=C["main"], fontsize=10, bbox=WB, zorder=9)
ax.text(0.72, -0.95, r"$\langle x,x\rangle = 1$", color=C["aux"], fontsize=10, bbox=WB, zorder=9)
ax.set_xlim(*XL); ax.set_ylim(*YL)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

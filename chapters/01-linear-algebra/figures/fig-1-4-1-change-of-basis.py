"""그림 1.4.1: 기저를 바꾸면 성분이 바뀐다 (1.4절, 예 1.4.10).

세 패널 모두 같은 벡터 v = (1, 2)(검정)와 같은 여벡터 ω(x) = x^1 + x^2의 등위선(주홍)을 그린다.
바뀌는 것은 기저(파랑)와 그 격자(회색 점선)뿐이다.
(a) 표준기저 E = (e_1, e_2):        [v] = (1, 2),   (ω_1, ω_2) = (1, 1)
(b) 늘인 기저 (2e_1, e_2):           [v] = (1/2, 2), (ω_1, ω_2) = (2, 1)   ← b_1을 2배 하면 v^1은 1/2배, ω_1은 2배
(c) 기준 기저 B = ((2,1), (-1,1)):  [v] = (1, 1),   (ω_1, ω_2) = (3, 0)
어느 기저에서나 Σ ω_i v^i = ω(v) = 3.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-4-1-change-of-basis.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

v = np.array([1.0, 2.0])
omega = np.array([1.0, 1.0])
bases = [
    ("(a)", np.array([[1.0, 0.0], [0.0, 1.0]]), (r"$e_1$", r"$e_2$"),
     r"$[v] = (1,\,2),\ \ (\omega_1, \omega_2) = (1,\,1)$"),
    ("(b)", np.array([[2.0, 0.0], [0.0, 1.0]]), (r"$\tilde b_1 = 2e_1$", r"$\tilde b_2 = e_2$"),
     r"$[v] = (\frac{1}{2},\,2),\ \ (\omega_1, \omega_2) = (2,\,1)$"),
    ("(c)", np.array([[2.0, -1.0], [1.0, 1.0]]), (r"$b_1$", r"$b_2$"),
     r"$[v] = (1,\,1),\ \ (\omega_1, \omega_2) = (3,\,0)$"),
]

# 자기검사: 성분과 불변량 (정리 1.4.5, 1.4.6, 명제 1.4.8)
expected = [((1, 2), (1, 1)), ((0.5, 2), (2, 1)), ((1, 1), (3, 0))]
for (_, Bm, _, _), (vc, oc) in zip(bases, expected):
    vt = np.linalg.solve(Bm, v)            # ṽ = B^{-1} v
    ot = omega @ Bm                        # ω̃ = ω B
    assert np.allclose(vt, vc) and np.allclose(ot, oc)
    assert np.isclose(ot @ vt, 3.0)        # Σ ω̃_j ṽ^j = ω(v)


def arrow(ax, base, vec, color, lw=None, z=6):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


XL, YL = (-1.6, 2.7), (-0.9, 2.7)
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.95))
WB = dict(facecolor="white", edgecolor="none", pad=0.3)
for ax, (tag, Bm, labels, caption) in zip(axs, bases):
    dgfig.schematic_axes(ax, XL, YL)
    # 기저의 격자
    ts = np.array([-8.0, 8.0])
    for k in range(-5, 6):
        for P0, d in ((k * Bm[:, 0], Bm[:, 1]), (k * Bm[:, 1], Bm[:, 0])):
            pts = P0 + np.outer(ts, d)
            ax.plot(pts[:, 0], pts[:, 1], color=C["aux"], lw=0.45, ls=(0, (3, 2.5)), zorder=1)
    # ω의 등위선 (기저와 무관)
    for c in range(-3, 6):
        P = np.array([[c + 5, -5], [c - 5, 5]], float)
        ax.plot(P[:, 0], P[:, 1], color=C["normal"], lw=0.8, zorder=2)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=7)
    arrow(ax, (0, 0), Bm[:, 0], C["tangent"])
    arrow(ax, (0, 0), Bm[:, 1], C["tangent"])
    arrow(ax, (0, 0), v, C["main"], lw=1.8)
    ax.text(v[0] - 0.4, v[1] + 0.05, r"$v$", color=C["main"], fontsize=12, bbox=WB, zorder=8)
    p1 = Bm[:, 0] + (np.array([-0.15, -0.42]) if tag != "(c)" else np.array([-0.35, -0.55]))
    ax.text(*p1, labels[0], color=C["tangent"], fontsize=10.5, bbox=WB, zorder=8,
            ha="center" if tag != "(a)" else "left")
    p2 = Bm[:, 1] + np.array([-0.12, 0.12]) if tag != "(c)" else Bm[:, 1] + np.array([-0.1, 0.18])
    ax.text(*p2, labels[1], color=C["tangent"], fontsize=10.5, bbox=WB, zorder=8, ha="right")
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11, bbox=WB, zorder=9)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.99, bottom=0.12)
dgfig.save(fig, __file__)

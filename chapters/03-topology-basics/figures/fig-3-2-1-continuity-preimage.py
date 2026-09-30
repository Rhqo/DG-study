"""그림 3.2.1: 열린집합의 역상으로 본 연속 (3.2절, 정의 3.2.1, 비예 3.2.9).

(a) f(x) = 1 + 0.6 sin x. f(p) (p = 0.5)를 포함하는 열린구간 V = (f(p) - 0.25, f(p) + 0.25)의 역상
    f^{-1}(V)는 그림 범위 [-0.5, 3.5] 안에서 열린구간 두 개의 합집합이고, p는 그 한 조각 안에 있다.
(b) 계단함수 H(x) = 0 (x < 0), 1 (x >= 0). V = (1/2, 3/2)의 역상은 [0, ∞)이고, 점 0을 포함하지만
    0을 포함하는 어떤 열린구간도 포함하지 않는다. 따라서 H는 연속이 아니다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-2-1-continuity-preimage.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

f = lambda x: 1 + 0.6 * np.sin(x)
p = 0.5
h = 0.25
V = (f(p) - h, f(p) + h)

xs = np.linspace(-0.5, 3.5, 40001)
inV = (f(xs) > V[0]) & (f(xs) < V[1])
# 역상 f^{-1}(V)의 조각(연속한 참 구간)들의 끝점
edges = np.flatnonzero(np.diff(inV.astype(int)))
starts = [xs[i + 1] for i in edges if not inV[i]]
ends = [xs[i] for i in edges if inV[i]]
pieces = list(zip(starts, ends))
# 자기검사: 그림 범위 안에서 역상은 열린구간 두 개 (a, b), (π - b, π - a)이고
# a = arcsin((f(p) - h - 1)/0.6), b = arcsin((f(p) + h - 1)/0.6), 첫 조각이 p를 포함한다
a_ex = np.arcsin((V[0] - 1) / 0.6)
b_ex = np.arcsin((V[1] - 1) / 0.6)
assert len(pieces) == 2
assert np.allclose(pieces[0], (a_ex, b_ex), atol=1e-3)
assert np.allclose(pieces[1], (np.pi - b_ex, np.pi - a_ex), atol=1e-3)
a, b = pieces[0]
assert a < p < b

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8))

# (a) --------------------------------------------------------------------------
ax = axs[0]
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-0.25, 2.0)
ax.set_axis_off()
ax.plot([-0.6, 3.6], [0, 0], color=C["aux"], lw=0.6)
ax.plot([0, 0], [-0.2, 1.95], color=C["aux"], lw=0.6)
ax.axhspan(V[0], V[1], xmin=0, xmax=1, color=C["region"], alpha=0.25, lw=0)
ax.plot([0, 0], V, color=C["tangent"], lw=3.2, solid_capstyle="butt")
ax.plot(xs, f(xs), color=C["main"], lw=LW["main"])
for a0, b0 in pieces:
    ax.plot([a0, b0], [0, 0], color=C["accent"], lw=3.2, solid_capstyle="butt")
    for x0 in (a0, b0):
        ax.plot([x0, x0], [0, f(x0)], color=C["aux"], lw=0.7, ls=(0, (3, 2)))
        ax.plot(x0, 0, "o", ms=4.5, mfc="white", mec=C["accent"], mew=1.2, zorder=6)
ax.plot(p, 0, "o", color=C["main"], ms=3, zorder=7)
ax.plot(p, f(p), "o", color=C["main"], ms=3, zorder=7)
ax.text(p, -0.2, r"$p$", ha="center", fontsize=12)
ax.text(-0.08, f(p), r"$V$", ha="right", va="center", fontsize=12, color=C["tangent"])
ax.text(2.9, 1.72, r"$f$", fontsize=12)
ax.text((b + pieces[1][0]) / 2, 0.1, r"$f^{-1}(V)$", fontsize=11, ha="center", color=C["accent"])
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) --------------------------------------------------------------------------
ax = axs[1]
ax.set_xlim(-2.1, 2.1); ax.set_ylim(-0.25, 2.0)
ax.set_axis_off()
ax.plot([-2.1, 2.1], [0, 0], color=C["aux"], lw=0.6)
ax.plot([0, 0], [-0.2, 1.95], color=C["aux"], lw=0.6)
Vb = (0.5, 1.5)
H = lambda x: np.where(x < 0, 0.0, 1.0)
# 자기검사: H^{-1}(V) = [0, ∞) (표본)
tt = np.linspace(-2, 2, 4001)
assert np.array_equal((H(tt) > Vb[0]) & (H(tt) < Vb[1]), tt >= 0)
ax.axhspan(*Vb, color=C["region"], alpha=0.25, lw=0)
ax.plot([0, 0], Vb, color=C["tangent"], lw=3.2, solid_capstyle="butt")
ax.plot([-2.0, 0], [0.0, 0.0], color=C["main"], lw=LW["main"])
ax.plot([0, 2.0], [1.0, 1.0], color=C["main"], lw=LW["main"])
ax.plot(0, 0, "o", ms=5, mfc="white", mec=C["main"], mew=1.2, zorder=6)
ax.plot(0, 1, "o", ms=5, color=C["main"], zorder=6)
ax.plot([0, 2.0], [0, 0], color=C["accent"], lw=3.2, solid_capstyle="butt", zorder=4)
ax.plot(0, 0.0, "o", ms=5.5, color=C["accent"], zorder=7)
ax.text(-0.08, 1.0, r"$V$", ha="right", va="center", fontsize=12, color=C["tangent"])
ax.text(1.2, 1.12, r"$H$", fontsize=12)
ax.text(0.6, 0.1, r"$H^{-1}(V)$", fontsize=11, color=C["accent"])
ax.text(0.0, -0.2, r"$0$", ha="center", fontsize=11)
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

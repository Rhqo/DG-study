"""그림 3.6.2: 르베그 수 (3.6절, 보조정리 3.6.17, 예 3.6.18).

[0, 1]의 열린덮개 U_1 = (-0.1, 0.4), U_2 = (0.3, 0.75), U_3 = (0.6, 1.1).
δ = 0.05이면 [0, 1]의 모든 점 x에서 (x - δ, x + δ) ∩ [0, 1]이 어느 한 U_i에 들어간다.
겹치는 부분 (0.3, 0.4)와 (0.6, 0.75) 가운데 좁은 쪽의 폭의 절반이 이 덮개의 가장 큰 르베그 수다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-6-2-lebesgue-number.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

cover = [(-0.1, 0.4), (0.3, 0.75), (0.6, 1.1)]
delta = 0.05


def works(d, xs=np.linspace(0, 1, 20001)):
    """모든 x에 대해 [x-d, x+d] ∩ [0,1] ⊆ 어떤 U_i (열린구간 안에 닫힌구간이 들어가는지)."""
    for x in xs:
        lo, hi = max(0.0, x - d), min(1.0, x + d)
        if not any(a < lo and hi < b for a, b in cover):
            return False
    return True


# 자기검사: δ = 0.05보다 조금 작은 값은 통하고, 0.05보다 조금 큰 값은 통하지 않는다
assert works(delta - 1e-3) and not works(delta + 1e-3)

fig, ax = plt.subplots(figsize=(5.4, 1.9))
ax.set_xlim(-0.16, 1.16); ax.set_ylim(-0.42, 0.62)
ax.set_axis_off()
ax.plot([0, 1], [0, 0], color=C["main"], lw=LW["main"])
for x0 in (0, 1):
    ax.plot(x0, 0, "o", ms=4.5, color=C["main"], zorder=5)
cols = [C["tangent"], C["third"], C["accent"]]
for i, ((a, b), col) in enumerate(zip(cover, cols)):
    y = 0.16 + 0.14 * i
    ax.plot([a, b], [y, y], color=col, lw=2.4, solid_capstyle="butt")
    for x0 in (a, b):
        ax.plot(x0, y, "o", ms=4, mfc="white", mec=col, mew=1.1, zorder=5)
    ax.text(b + 0.015, y, r"$U_{%d}$" % (i + 1), fontsize=10, color=col, va="center")
for xc in (0.35, 0.675):
    ax.plot([xc - delta, xc + delta], [-0.12, -0.12], color=C["main"], lw=3, solid_capstyle="butt")
    ax.plot(xc, -0.12, "o", ms=3, color=C["main"])
ax.text(0.35, -0.3, r"$(x-\delta, x+\delta)$", fontsize=9, ha="center")
ax.text(0.0, -0.3, r"$0$", fontsize=10, ha="center")
ax.text(1.0, -0.3, r"$1$", fontsize=10, ha="center")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

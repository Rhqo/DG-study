"""그림 11.2.2: ℝ의 두 차트 (ℝ, id)와 (ℝ, ψ), ψ(x) = x³의 두 좌표변환 (11.2절, 비예 11.2.3).

(a) ψ∘id^{-1}(x) = x³: 매끄럽고 0에서 기울기 0.
(b) id∘ψ^{-1}(u) = u^{1/3}: 연속이지만 0에서 기울기가 무한대(수직 접선).
구간 [-1.5, 1.5]. 점선은 (a)에서 기울기 0인 접선 y = 0, (b)에서 수직 접선 u = 0.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-2-2-cube-chart.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

x = np.linspace(-1.5, 1.5, 601)
cube = x ** 3
cbrt = np.sign(x) * np.abs(x) ** (1 / 3)

# 자기검사: 서로 역, 차분몫의 거동
assert np.allclose(np.sign(cube) * np.abs(cube) ** (1 / 3), x)
for h in (1e-2, 1e-4, 1e-6):
    assert abs(h ** 3 / h) < h                       # (x³)'(0) = 0
    assert (h ** (1 / 3)) / h > h ** (-2 / 3) * 0.999  # u^{1/3}의 차분몫 = h^{-2/3} → ∞

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax, y, lab, ttl in ((a1, cube, r"$u = x^3$", "(a)"), (a2, cbrt, r"$x = u^{1/3}$", "(b)")):
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.75, 1.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_color(C["aux"])
    ax.spines["bottom"].set_color(C["aux"])
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.tick_params(colors=C["aux"], labelsize=9, length=3)
    ax.plot(x, y, color=C["main"], lw=LW["main"], zorder=3)
    ax.plot(0, 0, "o", ms=4, color=C["accent"], zorder=4)
    ax.text(-1.55, 1.6, ttl, fontsize=11)
a1.plot([-1.2, 1.2], [0, 0], color=C["accent"], lw=1.1, ls=(0, (4, 2.5)), zorder=2)
a2.plot([0, 0], [-1.3, 1.3], color=C["accent"], lw=1.1, ls=(0, (4, 2.5)), zorder=2)
a1.text(0.55, 1.35, r"$u = x^3$", fontsize=11)
a1.text(1.52, 0.1, r"$x$", fontsize=11)
a1.text(0.08, 1.62, r"$u$", fontsize=11)
a2.text(0.35, 1.35, r"$x = u^{1/3}$", fontsize=11)
a2.text(1.52, 0.1, r"$u$", fontsize=11)
a2.text(0.08, 1.62, r"$x$", fontsize=11)

fig.subplots_adjust(wspace=0.15, left=0.02, right=0.98, top=0.98, bottom=0.04)
dgfig.save(fig, __file__)

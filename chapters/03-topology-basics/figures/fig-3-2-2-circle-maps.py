"""그림 3.2.2: 원 S^1으로 가는 두 사상 (3.2절, 예 3.2.14, 비예 3.2.15).

(a) e(t) = (cos t, sin t), t ∈ [0, 2π)는 연속 전단사이지만 위상동형사상이 아니다.
    U = [0, 1)은 [0, 2π)의 열린집합인데, e(U)(주황 호)는 S^1의 열린집합이 아니다.
    (1, 0)을 중심으로 하는 공 B_ε((1,0))(ε = 0.35)에는 t가 2π에 가까운 점 e(t)(파랑)가 들어 있고,
    그 점들은 e(U)에 속하지 않는다.
(b) 입체사영 σ(x) = x^1/(1 - x^2) (n = 1). 북극 N = (0, 1)과 x를 잇는 직선이 x^1축과 만나는 점이 σ(x)다.
    x = (3/5, 4/5)이면 σ(x) = 3 (연습 3.2.3).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-2-2-circle-maps.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
LW = dgfig.LW

e = lambda t: np.stack([np.cos(t), np.sin(t)], axis=-1)
eps = 0.35
tU = np.linspace(0, 1, 200)
# 자기검사: 2π - δ (δ 작음)에서 e(t)는 B_eps((1,0)) 안에 있고 e(U)에 없다 (t ∉ [0,1))
for delta in (0.05, 0.15, 0.3):
    pt = e(2 * np.pi - delta)
    assert np.linalg.norm(pt - np.array([1.0, 0.0])) < eps
    assert not (0 <= 2 * np.pi - delta < 1)

fig = plt.figure(figsize=(6.6, 2.9))
ax0 = fig.add_axes([0.0, 0.08, 0.2, 0.84])
ax1 = fig.add_axes([0.2, 0.0, 0.36, 1.0])
ax2 = fig.add_axes([0.6, 0.0, 0.4, 1.0])

# (a) 왼쪽: 구간 [0, 2π)을 세로로 ----------------------------------------------------
ax = ax0
ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.6, 2 * np.pi + 0.6)
ax.set_axis_off()
ax.plot([0, 0], [0, 2 * np.pi], color=C["main"], lw=LW["main"])
ax.plot(0, 0, "o", ms=4.5, color=C["main"], zorder=5)
ax.plot(0, 2 * np.pi, "o", ms=4.5, mfc="white", mec=C["main"], mew=1.2, zorder=5)
ax.plot([0, 0], [0, 1], color=C["accent"], lw=4, solid_capstyle="butt", zorder=4)
ax.plot([0, 0], [2 * np.pi - 0.3, 2 * np.pi - 0.02], color=C["tangent"], lw=4, solid_capstyle="butt", zorder=4)
ax.text(0.15, 0.0, r"$0$", va="center", fontsize=11)
ax.text(0.15, 2 * np.pi, r"$2\pi$", va="center", fontsize=11)
ax.text(0.15, 1.0, r"$1$", va="center", fontsize=11)
ax.text(-0.18, 0.5, r"$U$", ha="right", va="center", fontsize=12, color=C["accent"])
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (a) 오른쪽: 원 -----------------------------------------------------------------
ax = ax1
dgfig.schematic_axes(ax, (-1.35, 1.45), (-1.35, 1.35))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(*e(th).T, color=C["main"], lw=LW["main"] * 0.8, zorder=2)
ax.plot(*e(tU).T, color=C["accent"], lw=4, solid_capstyle="butt", zorder=3)
tB = np.linspace(2 * np.pi - 0.3, 2 * np.pi - 0.02, 50)
ax.plot(*e(tB).T, color=C["tangent"], lw=4, solid_capstyle="butt", zorder=3)
ax.add_patch(Circle((1, 0), eps, fill=False, edgecolor=C["aux"], lw=1.0, ls=(0, (3, 2)), zorder=4))
ax.plot(1, 0, "o", ms=4.5, color=C["main"], zorder=6)
ax.text(1.08, 0.36, r"$e(U)$", fontsize=11, color=C["accent"])
ax.text(0.62, -0.62, r"$B_\varepsilon((1,0))$", fontsize=10, color=C["aux"])
ax.text(-1.05, 0.85, r"$S^1$", fontsize=12, ha="right")
dgfig.map_arrow(fig, ax0, ax1, r"$e$", xy_from=(0.62, 0.62), xy_to=(0.1, 0.72), rad=-0.3)

# (b) 입체사영 ---------------------------------------------------------------------
ax = ax2
dgfig.schematic_axes(ax, (-1.4, 3.4), (-1.3, 1.35))
ax.plot(*e(th).T, color=C["main"], lw=LW["main"] * 0.8, zorder=2)
ax.plot([-1.4, 3.4], [0, 0], color=C["aux"], lw=0.7, zorder=1)
N = np.array([0.0, 1.0])
x = np.array([3 / 5, 4 / 5])
sig = x[0] / (1 - x[1])
assert abs(sig - 3.0) < 1e-12
assert abs(np.linalg.norm(x) - 1) < 1e-12
# 자기검사: N, x, (σ(x), 0)이 한 직선 위에 있다
v1, v2 = x - N, np.array([sig, 0.0]) - N
assert abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-12
ax.plot([N[0], sig], [N[1], 0], color=C["tangent"], lw=1.1, zorder=3)
x2 = np.array([-np.sin(0.9), -np.cos(0.9)])          # 남반구의 점
sig2 = x2[0] / (1 - x2[1])
ax.plot([N[0], sig2], [N[1], 0], color=C["tangent"], lw=1.1, zorder=3)
for q in (N, x, x2):
    ax.plot(*q, "o", ms=4, color=C["main"], zorder=6)
for s in (sig, sig2):
    ax.plot(s, 0, "o", ms=4, color=C["accent"], zorder=6)
ax.text(0.08, 1.08, r"$N$", fontsize=12)
ax.text(x[0] + 0.07, x[1] + 0.05, r"$x$", fontsize=12)
ax.text(sig, -0.25, r"$\sigma(x)$", fontsize=11, ha="center", color=C["accent"])
ax.text(x2[0] - 0.2, x2[1] - 0.12, r"$y$", fontsize=12)
ax.text(sig2 - 0.12, -0.28, r"$\sigma(y)$", fontsize=11, ha="right", color=C["accent"])
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

dgfig.save(fig, __file__)

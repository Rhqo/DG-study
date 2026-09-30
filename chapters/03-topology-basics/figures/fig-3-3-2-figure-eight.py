"""그림 3.3.2: 8자 곡선 — 연속 단사이지만 위상 매장이 아닌 사상 (3.3절, 비예 3.3.11).

β(t) = (sin 2t, sin t), t ∈ (-π, π). β는 단사이고 연속이며, 상 E = β((-π, π))는 8자 모양이다.
U = (-1, 1)은 (-π, π)의 열린집합이고 β(U)(검정 굵은 선)는 원점을 지난다. 그러나 t → ±π일 때 β(t) → (0, 0)이므로
원점을 중심으로 하는 공(점선, 반지름 0.3)에는 t가 ±π에 가까운 점들의 상(주황)이 들어 있고, 그 점들은 β(U)에 없다.
따라서 β(U)는 E의 부분공간 위상에서 열려 있지 않다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-3-2-figure-eight.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
LW = dgfig.LW

beta = lambda t: np.stack([np.sin(2 * t), np.sin(t)], axis=-1)
t = np.linspace(-np.pi, np.pi, 4001)[1:-1]
B = beta(t)

# 자기검사 1: 단사 (표본에서 서로 다른 t의 상이 충분히 떨어져 있지 않은 쌍은 t가 가까운 쌍뿐)
ts = np.linspace(-np.pi, np.pi, 1201)[1:-1]
P = beta(ts)
D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
close_pairs = np.argwhere((D < 1e-9) & (np.abs(ts[:, None] - ts[None, :]) > 1e-9))
assert close_pairs.size == 0
# 자기검사 2: t → ±π이면 β(t) → (0, 0)
assert np.linalg.norm(beta(np.pi - 1e-6)) < 1e-5 and np.linalg.norm(beta(-np.pi + 1e-6)) < 1e-5
# 자기검사 3: 원점 둘레 공 B_0.3(0)에 t ∈ (π - 0.12, π)의 상이 들어간다 (그 t는 U = (-1,1) 밖)
eps = 0.3
tt = np.linspace(np.pi - 0.12, np.pi - 1e-4, 50)
assert np.all(np.linalg.norm(beta(tt), axis=1) < eps) and np.all(np.abs(tt) >= 1)
# 자기검사 4: 상 위의 점은 x^2 = 4y^2(1 - y^2)를 만족 (연습 3.3.6)
assert np.allclose(B[:, 0] ** 2, 4 * B[:, 1] ** 2 * (1 - B[:, 1] ** 2))

fig, ax = plt.subplots(figsize=(3.6, 3.4))
dgfig.schematic_axes(ax, (-1.25, 1.35), (-1.2, 1.25))
ax.plot([-1.2, 1.2], [0, 0], color=C["aux"], lw=0.5, zorder=0)
ax.plot([0, 0], [-1.15, 1.15], color=C["aux"], lw=0.5, zorder=0)
ax.plot(*B.T, color=C["main"], lw=1.0, zorder=2)
tu = np.linspace(-1, 1, 400)
ax.plot(*beta(tu).T, color=C["main"], lw=3.0, zorder=3)
for tb in (np.linspace(np.pi - 0.45, np.pi - 1e-3, 100), np.linspace(-np.pi + 1e-3, -np.pi + 0.45, 100)):
    ax.plot(*beta(tb).T, color=C["accent"], lw=3.0, zorder=4)
ax.add_patch(Circle((0, 0), eps, fill=False, edgecolor=C["aux"], lw=1.0, ls=(0, (3, 2)), zorder=5))
ax.plot(0, 0, "o", ms=4, color=C["main"], zorder=6)
for tm, lab, off in ((1.0, r"$\beta(1)$", (0.07, -0.13)), (-1.0, r"$\beta(-1)$", (-0.5, -0.2))):
    q = beta(np.array(tm))
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=C["main"], mew=1.2, zorder=6)
    ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=10)
ax.text(-0.95, 0.35, r"$t \to \pi$", fontsize=10, color=C["accent"])
ax.text(0.45, -0.42, r"$t \to -\pi$", fontsize=10, color=C["accent"])
ax.text(-1.15, 0.9, r"$E$", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

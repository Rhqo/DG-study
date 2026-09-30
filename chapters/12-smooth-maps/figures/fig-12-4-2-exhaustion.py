"""그림 12.4.2: 구멍 뚫린 평면 M = ℝ² ∖ {0}의 컴팩트 소진 (12.4절, 예 12.4.10).

K_j = {x : 1/(j+1) ≤ |x| ≤ j}, V_j = {x : 1/(j+1) < |x| < j} (j = 1, 2, 3).
K_j는 닫힌 원환(컴팩트)이고 V_j ⊆ K_j ⊆ V_{j+1}이다. j가 커질수록 원점 쪽과 무한대 쪽으로 동시에 커진다.
(a) 전체 (|x| ≤ 3.3), (b) 원점 근처 확대 (|x| ≤ 0.6). 원점(빈 원)은 M에 없다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-4-2-exhaustion.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Annulus, Circle

C = dgfig.COLORS
LW = dgfig.LW

js = (1, 2, 3)
inner = {j: 1.0 / (j + 1) for j in range(1, 6)}
outer = {j: float(j) for j in range(1, 6)}

# 자기검사: V_j ⊆ K_j ⊆ V_{j+1} (반지름 비교), 합집합이 M을 덮음(표본)
for j in range(1, 5):
    assert inner[j + 1] < inner[j] and outer[j] < outer[j + 1]
rng = np.random.default_rng(4)
pts = rng.normal(size=(2000, 2)) * rng.uniform(0.01, 3, (2000, 1))
r = np.linalg.norm(pts, axis=1)
jneed = np.maximum(np.maximum(np.ceil(r), np.ceil(1 / r) - 1), 1)     # x ∈ K_j인 j
assert np.all((1.0 / (jneed + 1) <= r) & (r <= jneed))

cols = {1: C["tangent"], 2: C["third"], 3: C["accent"]}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
for ax, L, title in ((axes[0], 3.3, "(a)"), (axes[1], 0.62, "(b)")):
    ax.set_aspect("equal")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xticks([])
    ax.set_yticks([])
    for j in reversed(js):
        ax.add_patch(Annulus((0, 0), outer[j], outer[j] - inner[j], facecolor=cols[j], alpha=0.22, lw=0))
        for rr in (inner[j], outer[j]):
            ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor=cols[j], lw=1.2))
    ax.plot([0], [0], "o", mfc="white", mec=C["main"], ms=4, zorder=5)
    ax.set_title(title, fontsize=11)
axes[0].text(0.62, 0.55, r"$K_1$", color=cols[1], fontsize=11)
axes[0].text(1.35, 1.25, r"$K_2$", color=cols[2], fontsize=11)
axes[0].text(2.05, 2.0, r"$K_3$", color=cols[3], fontsize=11)
for j, ang, pos in ((1, -0.5, (0.36, -0.52)), (2, -1.0, (0.05, -0.56)), (3, -1.9, (-0.52, -0.56))):
    rr = inner[j]
    axes[1].annotate(rf"$|x|=1/{j + 1}$", xy=(rr * np.cos(ang), rr * np.sin(ang)), xytext=pos, fontsize=9,
                     color=cols[j], arrowprops=dict(arrowstyle="-", color=cols[j], lw=0.6))
fig.tight_layout(w_pad=1.5)
dgfig.save(fig, __file__)

"""그림 12.2.1: F(x) = x³은 (ℝ, 𝒜₃)에서 (ℝ, 𝒜₁)로 가는 미분동형사상이다 (12.2절, 예 12.2.5).

위 직선: 정의역 (ℝ, 𝒜₃). 차트 ψ(x) = x³의 좌표 u가 −1, −0.75, …, 1인 점 x = u^{1/3}에 눈금을 찍는다(파랑).
아래 직선: 공역 (ℝ, 𝒜₁). 표준 좌표 y = −1, −0.75, …, 1에 눈금을 찍는다.
화살표는 F(x) = x³이다. F는 ψ-좌표 u인 점을 표준 좌표 y = u인 점으로 보낸다. 즉 좌표표현 id∘F∘ψ^{-1}은 항등사상이다.
위 직선의 회색 눈금은 보통의 좌표 x = −1, −0.5, 0, 0.5, 1의 위치다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-2-1-cube-diffeomorphism.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

C = dgfig.COLORS
LW = dgfig.LW

us = np.linspace(-1, 1, 9)          # ψ-좌표 (= 공역의 표준 좌표)
xs = np.cbrt(us)                    # 정의역에서의 위치 x = u^{1/3}
F = lambda x: x ** 3

# 자기검사 (§12.1): ψ(x_k) = u_k, F(x_k) = u_k, 좌표표현 id∘F∘ψ^{-1} = id
assert np.allclose(xs ** 3, us) and np.allclose(F(xs), us)
assert np.allclose(F(np.cbrt(np.linspace(-2, 2, 101))), np.linspace(-2, 2, 101))

fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-0.55, 1.45)
ax.set_axis_off()
yT, yB = 1.0, 0.0
for y in (yT, yB):
    ax.annotate("", xy=(1.3, y), xytext=(-1.3, y),
                arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.0, mutation_scale=10))
# 위 직선: 보통 좌표 x의 회색 눈금
for x in (-1, -0.5, 0, 0.5, 1):
    ax.plot([x, x], [yT - 0.03, yT + 0.03], color=C["aux"], lw=0.8)
    ax.text(x, yT + 0.16, f"{x:g}", color=C["aux"], fontsize=8, ha="center")
ax.text(-1.33, yT + 0.16, r"$x$", color=C["aux"], fontsize=9, ha="left")
# ψ-좌표 눈금과 F 화살표
for u, x in zip(us, xs):
    ax.plot([x], [yT], "o", color=C["tangent"], ms=4, zorder=5)
    ax.plot([u], [yB], "o", color=C["tangent"], ms=4, zorder=5)
    ax.add_patch(FancyArrowPatch((x, yT - 0.05), (u, yB + 0.06), arrowstyle="-|>", mutation_scale=8,
                                 color=C["accent"], lw=0.9, shrinkA=0, shrinkB=0, zorder=3))
for u in (-1, -0.5, 0, 0.5, 1):
    ax.text(u, yB - 0.2, f"{u:g}", color=C["tangent"], fontsize=8, ha="center")
    ax.text(np.cbrt(u), yT - 0.22, f"{u:g}", color=C["tangent"], fontsize=8, ha="center")
ax.text(1.33, yT + 0.12, r"$(\mathbb{R},\mathcal{A}_3)$", fontsize=11, ha="right", va="bottom")
ax.text(1.33, yB + 0.12, r"$(\mathbb{R},\mathcal{A}_1)$", fontsize=11, ha="right", va="bottom")
ax.text(-1.33, yT - 0.22, r"$u=\psi(x)$", color=C["tangent"], fontsize=9, ha="left")
ax.text(-1.33, yB - 0.2, r"$y$", color=C["tangent"], fontsize=9, ha="left")
ax.text(0.06, 0.5, r"$F(x)=x^3$", color=C["accent"], fontsize=11, ha="left", va="center")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

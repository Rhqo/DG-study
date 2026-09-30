"""그림 12.2.2: 사영 π_S: S¹ → ℝℙ¹은 국소 미분동형사상이다 (12.2절, 예 12.2.11(a)).

단위원 S¹의 오른쪽 반원 U₁⁺ = {x¹ > 0}(파랑)과 왼쪽 반원 U₁⁻ = {x¹ < 0}(청록).
원점을 지나는 직선 [x]는 서로 대척인 두 점 ±x에서 S¹과 만나고, 세로선 x¹ = 1(주황)과
높이 φ₁[x] = x²/x¹에서 만난다. 방향각 θ = −60°, −45°, …, 60°의 직선 9개를 그린다(회색).
두 반원은 각각 U₁ = ℝℙ¹ ∖ {[0 : 1]} 위로 일대일로 간다. 빠진 점 [0 : 1](세로축)은 점선이다.
좌표표현 φ₁∘π_S∘(φ₁⁺)^{-1}(w) = w/√(1 − w²) (식 (12.1.7), n = 1)을 자기검사로 확인한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/12-smooth-maps/figures/fig-12-2-2-circle-to-rp1.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from dgsym import rpn_charts

C = dgfig.COLORS
LW = dgfig.LW

rp = rpn_charts(1)
X = rp["coords"]
phi1 = sp.lambdify(X, rp["phi"](0)[0], "numpy")        # φ₁[x] = x²/x¹

thetas = np.deg2rad(np.arange(-60, 61, 15))
# 자기검사: 대척점은 같은 φ₁ 값을 갖고, φ₁ = tan θ, 좌표표현은 w/√(1 − w²)
for th in thetas:
    p = np.array([np.cos(th), np.sin(th)])
    assert abs(phi1(*p) - phi1(*(-p))) < 1e-12 and abs(phi1(*p) - np.tan(th)) < 1e-12
    w = p[1]                                            # φ₁⁺(x) = x²
    assert abs(phi1(np.sqrt(1 - w ** 2), w) - w / np.sqrt(1 - w ** 2)) < 1e-12

fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_xlim(-1.5, 2.25)
ax.set_ylim(-2.05, 2.05)
t = np.linspace(-np.pi / 2, np.pi / 2, 300)
ax.plot(np.cos(t), np.sin(t), color=C["tangent"], lw=2.2, zorder=4)
ax.plot(-np.cos(t), np.sin(t), color=C["third"], lw=2.2, zorder=4)
for s in (1, -1):
    ax.plot([0], [s], "o", mfc="white", mec=C["main"], ms=5, zorder=6)
ax.plot([0, 0], [-2.05, 2.05], color=C["aux"], lw=0.8, ls=(0, (4, 3)))
ax.plot([1, 1], [-2.05, 2.05], color=C["accent"], lw=1.8)
for th in thetas:
    s = np.tan(th)
    L = 1.3
    ax.plot([-L * np.cos(th), 1.0], [-L * np.sin(th), s], color=C["aux"], lw=0.7, zorder=2)
    ax.plot([np.cos(th)], [np.sin(th)], "o", color=C["tangent"], ms=3.5, zorder=5)
    ax.plot([-np.cos(th)], [-np.sin(th)], "o", color=C["third"], ms=3.5, zorder=5)
    ax.plot([1.0], [s], "o", color=C["accent"], ms=3.5, zorder=5)
ax.plot([0], [0], "o", color=C["main"], ms=3, zorder=6)
ax.text(0.62, -1.02, r"$U_1^+$", color=C["tangent"], fontsize=12)
ax.text(-1.05, -1.02, r"$U_1^-$", color=C["third"], fontsize=12)
ax.text(1.08, 1.85, r"$x^1=1$", color=C["accent"], fontsize=11)
ax.text(1.12, -0.25, r"$\varphi_1[x]=\dfrac{x^2}{x^1}$", color=C["accent"], fontsize=11)
ax.text(0.08, 1.85, r"$[0:1]$", color=C["aux"], fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

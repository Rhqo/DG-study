"""그림 3.7.2: 위상수학자의 사인 곡선 (3.7절, 비예 3.7.11).

T = {(x, sin(1/x)) : 0 < x <= 1} ∪ ({0} × [-1, 1]). 그래프 부분 A(검정)의 폐포가 T이므로 T는 연결이지만,
세로 선분(주황)의 점과 그래프의 점을 잇는 경로는 없다. x_k = 1/(π/2 + 2πk)에서 sin(1/x_k) = 1,
x'_k = 1/(3π/2 + 2πk)에서 -1이다(k = 1, 2, 3 표시).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-7-2-sine-curve.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

# 1/x를 균등하게 나눠 진동이 빠른 곳에서도 표본이 충분하도록 한다
s = np.linspace(1.0, 400.0, 200001)
x = 1.0 / s
y = np.sin(s)
# 자기검사: 표본점은 그래프 위에 있고, 표시한 점의 값이 ±1
xk = np.array([1 / (np.pi / 2 + 2 * np.pi * k) for k in (1, 2, 3)])
xk2 = np.array([1 / (3 * np.pi / 2 + 2 * np.pi * k) for k in (1, 2, 3)])
assert np.allclose(np.sin(1 / xk), 1) and np.allclose(np.sin(1 / xk2), -1)
# 폐포: 세로 선분의 각 점 (0, y0)에 그래프의 점이 얼마든지 가까이 있다 (y0 = 0.3, 거리 < 1e-3)
y0 = 0.3
cand = [1 / (np.arcsin(y0) + 2 * np.pi * k) for k in range(200, 205)]
assert all(abs(np.sin(1 / c) - y0) < 1e-9 and c < 1e-3 for c in cand)

fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.set_xlim(-0.08, 1.05); ax.set_ylim(-1.3, 1.3)
ax.set_axis_off()
ax.plot([-0.05, 1.03], [0, 0], color=C["aux"], lw=0.5, zorder=0)
ax.plot(x, y, color=C["main"], lw=0.7, zorder=2)
ax.plot([0, 0], [-1, 1], color=C["accent"], lw=2.4, solid_capstyle="butt", zorder=3)
ax.plot(xk, np.ones(3), "o", ms=3.5, color=C["tangent"], zorder=4)
ax.plot(xk2, -np.ones(3), "o", ms=3.5, color=C["third"], zorder=4)
ax.plot(1.0, np.sin(1.0), "o", ms=4, color=C["main"], zorder=5)
ax.text(1.0, np.sin(1.0) + 0.14, r"$(1, \sin 1)$", fontsize=10, ha="center")
ax.text(-0.03, 1.08, r"$\{0\} \times [-1, 1]$", fontsize=10, color=C["accent"])
ax.text(0.5, 1.12, r"$y = \sin(1/x)$", fontsize=10)
ax.text(0.19, 1.12, r"$x_k$", fontsize=10, color=C["tangent"])
ax.text(0.12, -1.22, r"$x'_k$", fontsize=10, color=C["third"])

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

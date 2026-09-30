"""그림 3.4.2: 실사영직선 RP^1과 차트 φ_1 (3.4절, 예 3.4.16).

RP^1의 점은 R^2에서 원점을 지나는 직선이다. 각 직선 ℓ = [x^1 : x^2]는 단위원 S^1과 두 점 ±x에서 만난다(S^1/± 모형).
x^1 ≠ 0이면 ℓ은 세로선 x^1 = 1과 한 점 (1, x^2/x^1)에서 만나고, φ_1[x^1 : x^2] = x^2/x^1은 그 점의 높이(기울기)다.
세로축 x^1 = 0 (점 [0 : 1])만 x^1 = 1과 만나지 않는다. 이 점이 U_1 밖의 유일한 점이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-4-2-rp1-chart.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

angles = np.deg2rad([20.0, 55.0, -35.0])        # 그릴 직선들의 방향각
phi1 = lambda x: x[1] / x[0]


def line_pts(th, L=2.4):
    d = np.array([np.cos(th), np.sin(th)])
    return np.outer([-L, L], d)


# 자기검사: 직선 위의 두 점 x, λx (λ ≠ 0)는 같은 φ_1 값을 준다 (잘 정의됨), 교점 (1, φ_1)은 직선 위에 있다
for th in angles:
    x = np.array([np.cos(th), np.sin(th)])
    for lam in (-2.0, -0.3, 0.7, 3.0):
        assert abs(phi1(lam * x) - phi1(x)) < 1e-12
    q = np.array([1.0, phi1(x)])
    assert abs(q[0] * x[1] - q[1] * x[0]) < 1e-12

fig, ax = plt.subplots(figsize=(4.4, 3.5))
dgfig.schematic_axes(ax, (-1.9, 2.6), (-1.75, 1.95))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=C["main"], lw=1.0, zorder=2)
ax.plot([-1.85, 2.4], [0, 0], color=C["aux"], lw=0.5, zorder=0)
# 세로축 x^1 = 0: U_1 밖의 점 [0 : 1]
ax.plot([0, 0], [-1.7, 1.9], color=C["aux"], lw=1.1, ls=(0, (1.5, 2)), zorder=1)
# 세로선 x^1 = 1 (차트의 "화면")
ax.plot([1, 1], [-1.7, 1.9], color=C["main"], lw=1.3, zorder=1)
cols = [C["tangent"], C["third"], C["accent"]]
for th0, col in zip(angles, cols):
    P = line_pts(th0)
    ax.plot(*P.T, color=col, lw=1.1, zorder=3)
    x = np.array([np.cos(th0), np.sin(th0)])
    for sgn in (1, -1):
        ax.plot(*(sgn * x), "o", ms=4, color=col, zorder=6)
    q = np.array([1.0, phi1(x)])
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=col, mew=1.3, zorder=7)
    ax.text(1.08, q[1], r"$\varphi_1 = %.2f$" % q[1], fontsize=9, color=col, va="center")
x0 = np.array([np.cos(angles[1]), np.sin(angles[1])])
ax.text(*(x0 + [-0.33, 0.06]), r"$x$", fontsize=11, color=cols[1])
ax.text(*(-x0 + [-0.3, -0.18]), r"$-x$", fontsize=11, color=cols[1])
ax.text(0.95, -1.62, r"$x^1 = 1$", fontsize=10, color=C["main"], ha="right")
ax.text(-0.08, 1.75, r"$[0:1]$", fontsize=10, color=C["aux"], ha="right")
ax.text(-1.25, 0.9, r"$S^1$", fontsize=11)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

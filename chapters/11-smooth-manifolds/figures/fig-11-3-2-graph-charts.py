"""그림 11.3.2: S^1의 반원 차트(그래프 차트) (11.3절, 예 11.3.4).

(a) U_2^+ = {x^2 > 0}(위 반원, 파랑)와 φ_2^+(x) = x^1 ∈ (-1, 1). 세로 점선은 x ↦ x^1 사영.
(b) U_1^+ = {x^1 > 0}(오른쪽 반원, 주황)와 φ_1^+(x) = x^2 ∈ (-1, 1). 가로 점선은 x ↦ x^2 사영.
표본점의 각은 (a) 20°, 50°, 80°, 110°, 140°, 160°, (b) -70°, -40°, -10°, 20°, 50°, 80°.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-3-2-graph-charts.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.1))
t = np.linspace(0, 2 * np.pi, 400)
for ax in (a1, a2):
    dgfig.schematic_axes(ax, (-1.45, 1.55), (-1.45, 1.45))
    ax.plot(np.cos(t), np.sin(t), color=C["aux"], lw=0.9, zorder=1)

# (a) 위 반원 → x^1 축의 (-1, 1)
tt = np.linspace(0, np.pi, 200)[1:-1]
a1.plot(np.cos(tt), np.sin(tt), color=C["tangent"], lw=2.4, zorder=3)
a1.plot([-1, 1], [-1.15, -1.15], color=C["tangent"], lw=2.4, zorder=3, solid_capstyle="butt")
for e in (-1, 1):
    a1.plot(e, 0, "o", ms=5, mfc="white", mec=C["main"], zorder=5)
    a1.plot(e, -1.15, "o", ms=5, mfc="white", mec=C["main"], zorder=5)
for deg in (20, 50, 80, 110, 140, 160):
    th = np.deg2rad(deg)
    p = np.array([np.cos(th), np.sin(th)])
    assert p[1] > 0 and abs(np.hypot(*p) - 1) < 1e-12        # U_2^+ 위의 점
    a1.plot([p[0], p[0]], [p[1], -1.15], color=C["aux"], lw=0.7, ls=(0, (3, 2.5)), zorder=2)
    a1.plot(*p, "o", ms=3.5, color=C["main"], zorder=6)
    a1.plot(p[0], -1.15, "o", ms=3.5, color=C["main"], zorder=6)
a1.text(-0.55, 1.02, r"$U_2^+$", fontsize=12, color=C["tangent"])
a1.text(0.0, -1.28, r"$\varphi_2^+(x) = x^1$", fontsize=10, ha="center", va="top")
a1.text(-1.4, 1.3, "(a)", fontsize=11)

# (b) 오른쪽 반원 → x^2 축의 (-1, 1)
tt = np.linspace(-np.pi / 2, np.pi / 2, 200)[1:-1]
a2.plot(np.cos(tt), np.sin(tt), color=C["accent"], lw=2.4, zorder=3)
a2.plot([-1.15, -1.15], [-1, 1], color=C["accent"], lw=2.4, zorder=3, solid_capstyle="butt")
for e in (-1, 1):
    a2.plot(0, e, "o", ms=5, mfc="white", mec=C["main"], zorder=5)
    a2.plot(-1.15, e, "o", ms=5, mfc="white", mec=C["main"], zorder=5)
for deg in (-70, -40, -10, 20, 50, 80):
    th = np.deg2rad(deg)
    p = np.array([np.cos(th), np.sin(th)])
    assert p[0] > 0 and abs(np.hypot(*p) - 1) < 1e-12        # U_1^+ 위의 점
    a2.plot([p[0], -1.15], [p[1], p[1]], color=C["aux"], lw=0.7, ls=(0, (3, 2.5)), zorder=2)
    a2.plot(*p, "o", ms=3.5, color=C["main"], zorder=6)
    a2.plot(-1.15, p[1], "o", ms=3.5, color=C["main"], zorder=6)
a2.text(0.8, 0.85, r"$U_1^+$", fontsize=12, color=C["accent"])
a2.text(-1.25, 0.0, r"$\varphi_1^+(x) = x^2$", fontsize=10, ha="right", va="center", rotation=90)
a2.text(-1.4, 1.3, "(b)", fontsize=11)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)

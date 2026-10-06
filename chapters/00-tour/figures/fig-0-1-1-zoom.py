"""Figure 0.1.1: 확대하면 평평하다 (local linearization) (0.1절).

ellipse γ(t) = (2 cos t, sin t) (0.2절의 Figure 0.2.2와 같은 곡선), 점 p = γ(π/3).
왼쪽: 곡선 전체, tangent line, 반너비 0.5인 상자 A.
오른쪽: 상자 A를 확대한 것. 곡선과 tangent line이 거의 겹치고, 그 안의 반너비 0.1인 상자 B에서는 구별되지 않는다.
자기검사: tangent line과 곡선 사이의 거리는 h²κ/2 + O(h³) (h = p에서 잰 거리).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-1-1-zoom.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C = dgfig.COLORS
A_, B_ = 2.0, 1.0
T0 = np.pi / 3


def gam(t):
    return np.stack([A_ * np.cos(t), B_ * np.sin(t)], axis=-1)


def dgam(t):
    return np.stack([-A_ * np.sin(t), B_ * np.cos(t)], axis=-1)


p = gam(T0)
v = dgam(T0)
tvec = v / np.linalg.norm(v)
nvec = np.array([-tvec[1], tvec[0]])
kappa = A_ * B_ / np.linalg.norm(v) ** 3          # signed curvature det(γ', γ'')/|γ'|³ (> 0, counterclockwise)

# 자기검사: 곡선과 tangent line 사이의 거리 ≈ κ h²/2
for h in (0.05, 0.1):
    tt = np.linspace(T0 - 0.5, T0 + 0.5, 200001)
    pts = gam(tt)
    s = (pts - p) @ tvec
    i = np.argmin(np.abs(s - h))
    dist = abs((pts[i] - p) @ nvec)
    assert abs(dist - kappa * h ** 2 / 2) < 0.5 * kappa * h ** 3 + 1e-6, (h, dist, kappa * h ** 2 / 2)

t = np.linspace(0, 2 * np.pi, 600)
curve = gam(t)

fig = plt.figure(figsize=(6.6, 3.0))
axL = fig.add_axes([0.0, 0.02, 0.52, 0.96])
axR = fig.add_axes([0.58, 0.02, 0.42, 0.96])

# 왼쪽 ------------------------------------------------------------------------------------
axL.plot(*curve.T, color=C["main"], lw=dgfig.LW["main"])
L = 1.6
axL.plot(*np.stack([p - L * tvec, p + L * tvec]).T, color=C["tangent"], lw=1.0)
axL.add_patch(Rectangle(p - 0.5, 1.0, 1.0, fill=False, ec=C["aux"], lw=1.0, ls=(0, (4, 2)), zorder=4))
axL.plot(*p, "o", color=C["main"], ms=4, zorder=6)
axL.text(*(p + np.array([0.08, -0.22])), r"$p$", fontsize=12)
axL.text(*(p + np.array([0.52, 0.32])), r"$A$", fontsize=12, color=C["aux"])
axL.text(-0.25, -0.75, r"$\gamma$", fontsize=13)
dgfig.schematic_axes(axL, (-2.4, 2.4), (-1.3, 1.6))

# 오른쪽: 상자 A 확대 -------------------------------------------------------------------------
axR.plot(*curve.T, color=C["main"], lw=2.2)
axR.plot(*np.stack([p - L * tvec, p + L * tvec]).T, color=C["tangent"], lw=1.2)
axR.add_patch(Rectangle(p - 0.1, 0.2, 0.2, fill=False, ec=C["aux"], lw=1.0, ls=(0, (4, 2)), zorder=4))
axR.add_patch(Rectangle(p - 0.5, 1.0, 1.0, fill=False, ec=C["aux"], lw=1.0, ls=(0, (4, 2)), zorder=4))
axR.plot(*p, "o", color=C["main"], ms=5, zorder=6)
axR.text(*(p + np.array([0.03, -0.09])), r"$p$", fontsize=12)
axR.text(*(p + np.array([0.11, 0.115])), r"$B$", fontsize=12, color=C["aux"])
axR.text(*(p + np.array([0.38, 0.4])), r"$A$", fontsize=12, color=C["aux"])
dgfig.schematic_axes(axR, (p[0] - 0.53, p[0] + 0.53), (p[1] - 0.53, p[1] + 0.53))

dgfig.save(fig, __file__)

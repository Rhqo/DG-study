"""Figure 0.2.1: curvature = unit tangent vector가 도는 빠르기 (0.2절).

곡선 γ(t) = (t, sin t), 0 ≤ t ≤ 2π (inflection point는 t = π).
왼쪽: arc length로 같은 간격인 9개의 점에서 unit tangent vector t (파랑, 길이 0.7배)와 번호.
오른쪽: 같은 9개의 t를 원점에서 그린 것(단위원 위). tangent angle θ가 처음에는 시계 방향(κ_s < 0),
       inflection point를 지난 뒤에는 반시계 방향(κ_s > 0)으로 돈다.
자기검사: |t| = 1, κ_s = det(γ', γ'')/|γ'|³ = −sin t/(1 + cos² t)^{3/2}의 부호가 θ의 증감과 일치한다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-2-1-tangent-turning.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ts = sp.symbols("t", real=True)
gam_s = sp.Matrix([ts, sp.sin(ts), 0])
ks_sym = dgsym.signed_curvature(gam_s, ts)
assert sp.simplify(ks_sym + sp.sin(ts) / (1 + sp.cos(ts) ** 2) ** sp.Rational(3, 2)) == 0
ks = sp.lambdify(ts, ks_sym, "numpy")

t = np.linspace(0, 2 * np.pi, 20001)
P = np.stack([t, np.sin(t)], axis=1)
dP = np.stack([np.ones_like(t), np.cos(t)], axis=1)
speed = np.linalg.norm(dP, axis=1)
s = np.concatenate([[0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(t))])
L = s[-1]
idx = [int(np.argmin(np.abs(s - L * k / 8))) for k in range(9)]
T = dP / speed[:, None]
assert np.allclose(np.linalg.norm(T, axis=1), 1)
theta = np.arctan2(T[:, 1], T[:, 0])
dtheta = np.gradient(theta, s)
assert np.allclose(dtheta[100:-100], ks(t[100:-100]), atol=1e-4)      # κ_s = dθ/ds

fig = plt.figure(figsize=(6.6, 2.7))
axL = fig.add_axes([0.0, 0.0, 0.62, 1.0])
axR = fig.add_axes([0.66, 0.05, 0.34, 0.9])

axL.plot(*P.T, color=C["main"], lw=dgfig.LW["main"])
for k, i in enumerate(idx):
    p, v = P[i], 0.7 * T[i]
    axL.annotate("", xy=p + v, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.5,
                                                        mutation_scale=11, shrinkA=0, shrinkB=0))
    axL.plot(*p, "o", color=C["main"], ms=3, zorder=5)
    off = np.array([0.0, -0.32]) if np.sin(t[i]) >= 0 else np.array([0.0, 0.25])
    if k in (0, 8):
        off = np.array([0.0, -0.3])
    axL.text(*(p + off), f"{k + 1}", fontsize=10, ha="center", va="center", color=C["tangent"])
ip = np.array([np.pi, 0.0])
axL.plot(*ip, "o", ms=6, mfc="white", mec=C["accent"], mew=1.6, zorder=6)
axL.text(np.pi + 0.18, 0.2, r"$\kappa_s = 0$", fontsize=11, color=C["accent"])
axL.text(1.3, 1.25, r"$\kappa_s < 0$", fontsize=11)
axL.text(4.4, -1.38, r"$\kappa_s > 0$", fontsize=11)
dgfig.schematic_axes(axL, (-0.3, 2 * np.pi + 0.8), (-1.6, 1.65))

c = np.linspace(0, 2 * np.pi, 400)
axR.plot(np.cos(c), np.sin(c), color=C["aux"], lw=dgfig.LW["aux"], ls=(0, (4, 3)))
for k, i in enumerate(idx):
    v = T[i]
    axR.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3,
                                                         mutation_scale=10, shrinkA=0, shrinkB=0))
v1, v5 = T[idx[0]], T[idx[4]]
axR.text(*(1.17 * v1), "1, 9", fontsize=10, ha="center", va="center", color=C["tangent"])
axR.text(*(1.2 * v5 + np.array([0.05, 0])), "5", fontsize=10, ha="center", va="center", color=C["tangent"])
a = np.linspace(0, theta[idx[0]], 50)
axR.plot(0.35 * np.cos(a), 0.35 * np.sin(a), color=C["main"], lw=0.9)
axR.text(0.42, 0.17, r"$\theta$", fontsize=12)
axR.plot(0, 0, "o", color=C["main"], ms=3)
dgfig.schematic_axes(axR, (-1.25, 1.45), (-1.25, 1.25))

dgfig.save(fig, __file__)

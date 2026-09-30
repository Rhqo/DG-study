"""그림 5.1.2: 곡률이 0인 점에서 주법선벡터가 뒤집히는 예 (5.1절, 비예 5.1.11).

곡선 γ(t) = (t, t³, 0), -1.07 ≤ t ≤ 1.07 (xy평면에 그린다). 주법선은 t = ±0.32, ±0.62, ±0.95에서 그린다.
t ≠ 0이면 κ > 0이고 주법선벡터는 n = γ''/|γ''|을 호의 길이로 계산한 것이다.
일반 매개변수에서는 n = (dt/dt)/|dt/dt| 이므로 이것을 sympy로 구해 그린다.
t → 0-이면 n → (0, -1), t → 0+이면 n → (0, 1)이어서 t = 0에서 연속으로 정할 수 없다.
벡터는 실제 길이의 0.35배로 그렸다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-1-2-normal-flip.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
t = sp.symbols("t", real=True)
gam = sp.Matrix([t, t ** 3, 0])
Tv, Nv, Bv = dgsym.frenet_frame(gam, t)           # t ≠ 0에서 정의 (γ' × γ'' ≠ 0)
kap, _ = dgsym.curvature_torsion(gam, t)
Nf = sp.lambdify(t, list(Nv), "numpy")
kf = sp.lambdify(t, kap, "numpy")
SC = 0.35

# 자기검사: 주법선은 단위벡터, 접선과 수직, t → 0±에서 극한이 (0, ∓1)·(-1)
for tt in (-1.0, -0.5, -0.1, 0.1, 0.5, 1.0):
    n = np.array(Nf(tt), float)
    tan = np.array([1.0, 3 * tt ** 2, 0.0]) / np.hypot(1, 3 * tt ** 2)
    assert abs(np.linalg.norm(n) - 1) < 1e-12 and abs(n @ tan) < 1e-12
    assert np.sign(n[1]) == np.sign(tt)            # t > 0이면 위, t < 0이면 아래
assert np.allclose(Nf(1e-6)[:2], [0, 1], atol=1e-6) and np.allclose(Nf(-1e-6)[:2], [0, -1], atol=1e-6)
assert abs(kf(1e-9)) < 1e-6                         # κ(0) = 0

fig, ax = plt.subplots(figsize=(4.8, 4.4))
ts = np.linspace(-1.07, 1.07, 500)
ax.plot(ts, ts ** 3, color=C["main"], lw=dgfig.LW["main"], zorder=3)
for tt in (-0.95, -0.62, -0.32, 0.32, 0.62, 0.95):
    p = np.array([tt, tt ** 3])
    n = np.array(Nf(tt), float)[:2]
    ax.annotate("", xy=p + SC * n, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=dgfig.LW["vector"], mutation_scale=11,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.plot(*p, "o", color=C["main"], ms=3.2, zorder=6)
# t = 0: 두 극한 방향(점선)
for sgn in (1, -1):
    ax.annotate("", xy=(0, sgn * SC), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.1, ls=(0, (3, 2)), mutation_scale=10,
                                shrinkA=0, shrinkB=0, alpha=0.8), zorder=4)
ax.plot(0, 0, "o", ms=5.5, mfc="white", mec=C["main"], zorder=7)
ax.text(0.05, SC + 0.07, r"$t\to 0^+$", fontsize=11, ha="left", va="bottom", color=C["normal"])
ax.text(0.05, -SC - 0.07, r"$t\to 0^-$", fontsize=11, ha="left", va="top", color=C["normal"])
ax.text(0.56, 1.0, r"$\mathbf{n}$", fontsize=12, color=C["normal"], ha="right", va="center")
ax.text(-1.12, -0.22, r"$\kappa>0$", fontsize=11, ha="left")
ax.text(1.12, 0.2, r"$\kappa>0$", fontsize=11, ha="right")
ax.text(-0.1, 0.1, r"$\kappa=0$", fontsize=11, ha="right", va="bottom")
ax.axhline(0, color=C["aux"], lw=0.6, ls=(0, (4, 3)), zorder=1)
dgfig.schematic_axes(ax, xlim=(-1.15, 1.15), ylim=(-1.25, 1.25))
dgfig.save(fig, __file__)

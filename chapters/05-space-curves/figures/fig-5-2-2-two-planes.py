"""그림 5.2.2: κ > 0 가정이 없으면 τ = 0이어도 평면곡선이 아닐 수 있다 (5.2절, 비예 5.2.10).

f(t) = e^{-1/t} (t > 0), f(t) = 0 (t ≤ 0) (2장 예 2.2.20). 곡선 γ(t) = (t, f(-t), f(t)), -2 ≤ t ≤ 2.
t ≤ 0인 부분은 xy평면(연한 하늘색)에, t ≥ 0인 부분은 xz평면(연한 회색)에 들어 있다.
t = 0, ±1/2에서 κ = 0이다(주황 점). 그 밖의 점에서는 τ = 0이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/05-space-curves/figures/fig-5-2-2-two-planes.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS
t = sp.symbols("t", real=True)
fpos = sp.exp(-1 / t)
right = sp.Matrix([t, 0, fpos])              # t > 0
left = sp.Matrix([t, fpos.subs(t, -t), 0])   # t < 0
kR, tR = dgsym.curvature_torsion(right, t)
kL, tL = dgsym.curvature_torsion(left, t)


def f(x):
    x = np.asarray(x, float)
    return np.where(x > 0, np.exp(-1 / np.where(x > 0, x, 1.0)), 0.0)


def G(tt):
    tt = np.asarray(tt, float)
    return np.stack([tt, f(-tt), f(tt)], axis=-1)


# 자기검사: 두 조각의 κ = 0인 점은 t = ±1/2, 나머지에서 τ = 0 (각 조각이 평면에 있다)
check_k = [float(kR.subs(t, sp.Rational(1, 2))), float(kL.subs(t, -sp.Rational(1, 2)))]
assert all(abs(k) < 1e-12 for k in check_k)
assert sp.simplify(tR) == 0 or all(abs(float(tR.subs(t, v))) < 1e-12 for v in (0.3, 0.9, 1.7))
assert all(abs(float(tL.subs(t, v))) < 1e-12 for v in (-0.3, -0.9, -1.7))
ts = np.linspace(-2, 2, 801)
H = G(ts)
assert np.all(H[ts <= 0, 2] == 0) and np.all(H[ts >= 0, 1] == 0)
# 세 점 (0,0,0), γ(-1), γ(-2)는 한 직선 위에 있지 않고, γ(1)은 그 평면 밖에 있다
P = G(np.array([-1.0, -2.0, 1.0]))
assert abs(np.linalg.det(P)) > 1e-2

fig = plt.figure(figsize=(5.6, 4.2))
ax = dgfig.axes3d(fig, elev=22, azim=-58)
# 두 평면의 일부
xy = [(-2.1, -0.15, 0), (0.15, -0.15, 0), (0.15, 0.75, 0), (-2.1, 0.75, 0)]
xz = [(-0.15, 0, -0.1), (2.1, 0, -0.1), (2.1, 0, 0.75), (-0.15, 0, 0.75)]
ax.add_collection3d(Poly3DCollection([xy], facecolor=C["region"], alpha=0.22, edgecolor=C["aux"], linewidth=0.6, zorder=1))
ax.add_collection3d(Poly3DCollection([xz], facecolor=C["surface"], alpha=0.45, edgecolor=C["aux"], linewidth=0.6, zorder=1))
ax.plot(*H.T, color=C["main"], lw=dgfig.LW["main"], zorder=5)
for tt in (-0.5, 0.0, 0.5):
    dgfig.point3d(ax, G(tt), role="accent", size=22, zorder=8)
for axis, lab in ((np.array([2.35, 0, 0]), r"$x$"), (np.array([0, 0.95, 0]), r"$y$"), (np.array([0, 0, 0.95]), r"$z$")):
    ax.plot(*np.stack([np.zeros(3), axis]).T, color=C["aux"], lw=0.7, zorder=2)
    ax.text(*(axis * 1.06), lab, fontsize=11, color=C["aux"])
ax.text(-1.8, 0.62, 0.02, r"$z=0$", fontsize=11, color=C["tangent"], zorder=9)
ax.text(1.55, 0.0, 0.62, r"$y=0$", fontsize=11, color=C["aux"], zorder=9)
ax.text(*(G(0.0) + np.array([0.05, 0.0, -0.13])), r"$\gamma(0)$", fontsize=11, zorder=9)
dgfig.equal_aspect(ax, np.concatenate([H, np.array(xy), np.array(xz)]), zoom=1.0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)

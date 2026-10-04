"""그림 8.4.1: 원환면의 가우스 곡률의 부호 분포 (8.4절, 예 8.4.5).

원환면 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8. K = cos u / (r(R + r cos u)) (dgsym.K_H로 계산, §7 값과 대조).
발산형 컬러맵 RdBu_r, 0이 중앙(TwoSlopeNorm)인 색으로 칠한다 (§12.3). 포물점의 원 u = ±π/2 (검정).
자기검사: K_H의 K가 §7의 식과 같다, u = 0에서 K > 0, u = π에서 K < 0, u = ±π/2에서 K = 0.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-4-1-torus-curvature.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.colors import TwoSlopeNorm

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
vals = {Rs: RV, rs: rV}
Ksym, _ = dgsym.K_H(ex["expr"], u, v, ex["positive"])
assert sp.simplify(Ksym - ex["expected"]["K"]) == 0
Kf = sp.lambdify(u, Ksym.subs(vals), "numpy")
X = sp.lambdify((u, v), list(ex["expr"].subs(vals)), "numpy")
assert Kf(0.0) > 0 and Kf(np.pi) < 0 and abs(Kf(np.pi / 2)) < 1e-15 and abs(Kf(-np.pi / 2)) < 1e-15


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


Ug, Vg = np.meshgrid(np.linspace(-np.pi, np.pi, 121), np.linspace(0, 2 * np.pi, 181), indexing="ij")
S = xmap(Ug, Vg)
Kc = Kf(0.5 * (Ug[:-1, :-1] + Ug[1:, 1:]))          # 면 중앙의 K
norm = TwoSlopeNorm(vmin=Kf(np.pi), vcenter=0.0, vmax=Kf(0.0))
cmap = plt.get_cmap("RdBu_r")

fig = plt.figure(figsize=(6.2, 3.9))
ax = dgfig.axes3d(fig, elev=38, azim=-60)
ax.set_position([0.0, 0.0, 0.78, 1.0])
ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], facecolors=cmap(norm(Kc)), rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False, rasterized=True, zorder=1)
view = dgfig.view_vector(ax)
tt = np.linspace(0, 2 * np.pi, 400)
for u0 in (np.pi / 2, -np.pi / 2):
    c = xmap(u0 * np.ones_like(tt), tt)
    nrm = np.stack([np.cos(u0) * np.cos(tt), np.cos(u0) * np.sin(tt), np.sin(u0) * np.ones_like(tt)], axis=1)
    dgfig.curve3d(ax, c, role="main", lw=1.4, visible=nrm @ view > 0, hidden=None, zorder=5)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
cax = fig.add_axes([0.72, 0.22, 0.022, 0.56])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cax)
cb.set_ticks([Kf(np.pi), 0, Kf(0.0)])
cb.set_ticklabels([r"$-\frac{1}{r(R-r)}$", r"$0$", r"$\frac{1}{r(R+r)}$"])
cb.set_label(r"$K$", rotation=0, labelpad=2)

dgfig.save(fig, __file__)

"""그림 8.5.1: 타원점과 쌍곡점 근처의 모양 — 원환면과 접평면 (8.5절, 명제 8.5.12).

원환면 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8.
(a) 타원점 p = x(0, 0) = (R + r, 0, 0): 접평면 x = R + r (하늘색). 근처의 곡면은 한쪽(x < R + r)에만 있다.
(b) 쌍곡점 p = x(π, 0) = (R − r, 0, 0): 접평면 x = R − r. 곡면과 접평면의 교선
    (R + r cos u) cos v = R − r (주황)이 p에서 두 점근방향으로 교차하고, 곡면은 접평면의 양쪽에 걸친다.
자기검사: 두 점에서 접평면의 법선 = N_x(p), (a)의 조각의 높이 ⟨q − p, N⟩의 부호가 일정, (b)는 두 부호.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-5-1-local-shape.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
vals = {Rs: RV, rs: rV}
X = sp.lambdify((u, v), list(ex["expr"].subs(vals)), "numpy")
Nf = sp.lambdify((u, v), list(dgsym.unit_normal(ex["expr"], u, v, ex["positive"]).subs(vals)), "numpy")


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


fig = plt.figure(figsize=(6.6, 3.3))
for k, (u0, lab) in enumerate(((0.0, "(a)"), (np.pi, "(b)"))):
    ax = dgfig.axes3d(fig, pos=121 + k, elev=22 if k == 0 else 18, azim=-72 if k == 0 else -142)
    p = np.array(X(u0, 0.0), float)
    n = np.array(Nf(u0, 0.0), float)
    assert np.allclose(np.abs(n), [1, 0, 0])
    U, V = np.meshgrid(np.linspace(u0 - 1.3, u0 + 1.3, 53), np.linspace(-1.0, 1.0, 41), indexing="ij")
    S = xmap(U, V)
    height = (S - p) @ n
    if k == 0:
        assert np.all(height[np.hypot(U - u0, V) > 1e-9] > 0)       # 한쪽 (N_x 쪽 = 안쪽)
    else:
        assert height.min() < 0 < height.max()
    dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.45, grid_every=4, zorder=1)
    e2, e3 = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    dgfig.tangent_plane(ax, p, e2, e3, size=0.95, zorder=4)
    if k == 1:
        uu = np.linspace(u0 - 1.3, u0 + 1.3, 400)
        cv = (RV - rV) / (RV + rV * np.cos(uu))
        vv = np.arccos(np.clip(cv, -1, 1))
        for sgn in (1, -1):
            c = xmap(uu, sgn * vv)
            m = np.abs(sgn * vv) <= 1.0
            ax.plot(*c[m].T, color=C["accent"], lw=1.8, zorder=6)
    dgfig.arrow3d(ax, p, 0.6 * n, role="normal", label=r"$\mathbf{N}$", zorder=10, label_offset=(0.0, 0.0, 0.08))
    dgfig.point3d(ax, p, size=14, zorder=11)
    ax.text(*(p + np.array([0, 0.08, -0.18])), r"$p$", fontsize=12, zorder=12)
    ax.text2D(0.02, 0.92, lab, transform=ax.transAxes, fontsize=11)
    dgfig.equal_aspect(ax, S.reshape(-1, 3), np.array([p + 0.95 * (e2 + e3), p - 0.95 * (e2 + e3)]), zoom=1.2)
fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=1.0, bottom=0.0)

dgfig.save(fig, __file__)

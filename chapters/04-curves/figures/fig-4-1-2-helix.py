"""그림 4.1.2: 나선과 그 속도벡터 (4.1절, 예 4.1.4, 예 4.1.8).

기준 예제의 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]), a = 1, b = 0.3, t ∈ [0, 4π].
나선은 원기둥 x² + y² = a² 위에 있다. t = kπ/2 (k = 1, …, 7)에서 속도벡터 γ'(t)를 실제 길이로 그린다.
|γ'(t)| = √(a² + b²) ≈ 1.044로 일정하다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-1-2-helix.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["helix"]
(t,) = ex["coords"]
a, b = ex["params"]
A, B = 1.0, 0.3
gam = sp.lambdify(t, list(ex["expr"].subs({a: A, b: B})), "numpy")
dgam = sp.lambdify(t, list(ex["expr"].diff(t).subs({a: A, b: B})), "numpy")

ts = np.linspace(0, 4 * np.pi, 800)
P = np.array([np.broadcast_to(c, ts.shape) for c in gam(ts)]).T
V = np.array([np.broadcast_to(c, ts.shape) for c in dgam(ts)]).T

# 자기검사: 원기둥 위에 있고, 속력은 √(a² + b²)로 일정
assert np.allclose(P[:, 0] ** 2 + P[:, 1] ** 2, A ** 2)
assert np.allclose(np.linalg.norm(V, axis=1), np.hypot(A, B))
assert np.allclose(P[-1] - P[0], [0, 0, 4 * np.pi * B])       # 두 바퀴에 4πb만큼 올라감

fig = plt.figure(figsize=(3.6, 4.4))
ax = dgfig.axes3d(fig, elev=17, azim=-58)

# 원기둥 (연회색 면 + 격자)
uu, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(-0.25, 4 * np.pi * B + 0.25, 25))
dgfig.surface(ax, A * np.cos(uu), A * np.sin(uu), zz, alpha=0.16, grid_every=6, zorder=1, shade=False)

# z축 (보조선)
ax.plot([0, 0], [0, 0], [-0.4, 4 * np.pi * B + 0.6], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(0.05, 0, 4 * np.pi * B + 0.72, r"$z$", color=C["aux"], fontsize=11)

# 나선: 보는 쪽의 반은 실선, 뒤쪽 반은 옅은 점선
normals = np.stack([np.cos(ts), np.sin(ts), 0 * ts], axis=1)
vis = dgfig.visible_mask(ax, P, normals)
dgfig.curve3d(ax, P, visible=vis, zorder=6)

# 속도벡터 (실제 길이): 보는 쪽 반원기둥 위의 점에서
T_VEC = [2 * np.pi * k + d for k in (0, 1, 2) for d in (-2.1, -0.95, 0.2)]
T_VEC = [tk for tk in T_VEC if 0 <= tk <= 4 * np.pi]
for tk in T_VEC:
    p = np.array(gam(tk), float)
    v = np.array(dgam(tk), float)
    n = np.array([np.cos(tk), np.sin(tk), 0.0])
    assert n @ dgfig.view_vector(ax) > 0          # 보이는 쪽인지 확인
    dgfig.arrow3d(ax, p, v, role="tangent", zorder=9)
    dgfig.point3d(ax, p, size=10, zorder=10)
tk = 2 * np.pi - 0.95
p = np.array(gam(tk), float)
v = np.array(dgam(tk), float)
ax.text(*(p + v + np.array([0.1, 0.0, -0.32])), r"$\gamma'(t)$", color=C["tangent"], fontsize=12)

# 한 바퀴에 올라가는 높이 2πb: γ(2π)와 γ(4π) 사이
p2, p4 = np.array(gam(2 * np.pi), float), np.array(gam(4 * np.pi), float)
off = np.array([0.0, 0.0, 0.0])
ax.plot(*np.stack([p2, p4]).T, color=C["aux"], lw=0.8, zorder=7)
for q in (p2, p4):
    dgfig.point3d(ax, q, size=8, role="aux", zorder=8)
ax.text(*((p2 + p4) / 2 + np.array([0.12, -0.12, 0.0])), r"$2\pi b$", color=C["aux"], fontsize=11)

dgfig.equal_aspect(ax, P, np.array([[1, 1, -0.3], [-1, -1, 4 * np.pi * B + 0.4]]), zoom=1.25)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)

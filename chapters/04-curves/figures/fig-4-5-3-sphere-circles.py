"""그림 4.5.3: 구면 위 한 점을 지나는 위도원과 대원의 곡률중심 (4.5절, 예 4.5.11, 예 4.5.12).

구면 S²(r) (dgsym.EXAMPLES["sphere"]), r = 1. 점 p = x(θ0, φp), θ0 = π/3, φp = 0.05.
- 위도원 θ = θ0 (검정): 평면 z = r cos θ0 위의 원, 반지름 r sin θ0 ≈ 0.866, 곡률중심 c = (0, 0, r cos θ0) = (0, 0, 0.5).
- p를 지나는 경선을 포함하는 대원 (주황): 곡률중심은 구의 중심 O, 반지름 r = 1.
p에서 두 곡률중심으로 가는 선분(주홍)은 곡률벡터의 방향이고, 두 선분 사이의 각은 π/2 − θ0 = 30°이다.
파란 화살표는 p에서 두 곡선의 단위접벡터(실제 길이의 0.45배).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-5-3-sphere-circles.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
Xf = sp.lambdify((th, ph), list(ex["expr"].subs(r, 1)), "numpy")


def X(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.stack([np.broadcast_to(c, a.shape) for c in Xf(a, b)], axis=-1)


TH0, PHP = np.pi / 3, 0.05
p = X(TH0, PHP)
c_lat = np.array([0.0, 0.0, np.cos(TH0)])
O = np.zeros(3)

# 위도원과 대원 (대원: φ = φp인 경선과 φ = φp + π인 경선을 이은 원)
phs = np.linspace(0, 2 * np.pi, 600)
LAT = X(TH0, phs)
ang = np.linspace(0, 2 * np.pi, 600)
e_rho = np.array([np.cos(PHP), np.sin(PHP), 0.0])
GC = np.cos(ang)[:, None] * np.array([0, 0, 1.0]) + np.sin(ang)[:, None] * e_rho   # (r cos a) e3 + (r sin a) e_ρ

# 자기검사 ----------------------------------------------------------------------
assert np.allclose(np.linalg.norm(LAT, axis=1), 1) and np.allclose(np.linalg.norm(GC, axis=1), 1)
assert np.allclose(np.linalg.norm(LAT - c_lat, axis=1), np.sin(TH0))          # 반지름 r sin θ0
assert np.allclose(LAT[:, 2], np.cos(TH0))                                     # 평면 z = r cos θ0
i_p = np.argmin(np.linalg.norm(GC - p, axis=1))
assert np.linalg.norm(GC[i_p] - p) < 1e-2                                      # 대원이 p를 지난다
u1, u2 = (c_lat - p) / np.linalg.norm(c_lat - p), (O - p) / np.linalg.norm(O - p)
assert abs(np.degrees(np.arccos(u1 @ u2)) - 30.0) < 1e-9                        # 각 π/2 − θ0
assert abs(np.linalg.norm(c_lat - p) - np.sin(TH0)) < 1e-12

fig = plt.figure(figsize=(4.6, 4.6))
ax = dgfig.axes3d(fig, elev=20, azim=-35)
V = dgfig.view_vector(ax)
assert p @ V > 0                                                              # p는 보이는 쪽

uu, vv = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
S = X(uu, vv)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.14, grid=False, zorder=1, shade=False)
# 윤곽선 (보는 방향에 수직인 대원)
w1 = np.cross(V, [0, 0, 1.0]); w1 /= np.linalg.norm(w1)
w2 = np.cross(V, w1)
RIM = np.cos(ang)[:, None] * w1 + np.sin(ang)[:, None] * w2
ax.plot(*RIM.T, color=C["aux"], lw=0.7, zorder=1.5)

dgfig.curve3d(ax, LAT, visible=LAT @ V > 0, zorder=6)
dgfig.curve3d(ax, GC, role="accent", visible=GC @ V > 0, zorder=5)
ax.plot([0, 0], [0, 0], [-1.25, 1.3], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(0.03, 0, 1.36, r"$z$", color=C["aux"], fontsize=11)

for q in (c_lat, O):
    ax.plot(*np.stack([p, q]).T, color=C["normal"], lw=1.3, zorder=8)
    dgfig.point3d(ax, q, role="third", size=16, zorder=9)
dgfig.point3d(ax, p, size=18, zorder=10)
t_lat = np.array([-np.sin(PHP), np.cos(PHP), 0.0])                              # ∂x/∂φ 방향 (단위)
t_mer = np.array([np.cos(TH0) * np.cos(PHP), np.cos(TH0) * np.sin(PHP), -np.sin(TH0)])   # ∂x/∂θ 방향 (단위)
dgfig.arrow3d(ax, p, t_lat, role="tangent", scale=0.45, zorder=11)
dgfig.arrow3d(ax, p, t_mer, role="tangent", scale=0.45, zorder=11)
ax.text(*(p + np.array([0.05, -0.12, 0.08])), r"$p$", fontsize=12, zorder=12)
ax.text(*(c_lat + np.array([-0.08, 0.0, 0.08])), r"$c$", fontsize=12, color=C["third"], zorder=12, ha="right")
ax.text(*(O + np.array([-0.08, 0.0, -0.12])), r"$O$", fontsize=12, color=C["third"], zorder=12, ha="right")

dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1.3]]), zoom=1.35)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)

"""그림 4.2.2: 나선 한 바퀴의 길이 — 원기둥을 펼치면 나선은 직각삼각형의 빗변이 된다 (4.2절, 예 4.2.8).

(a) 나선 γ(t) = (a cos t, a sin t, bt) (dgsym.EXAMPLES["helix"]), a = 1, b = 0.3, 0 ≤ t ≤ 2π와
    원기둥 x² + y² = a². 원기둥을 자르는 세로선 {(a, 0, z)}를 회색으로 그린다.
(b) 원기둥을 그 세로선을 따라 잘라 펼친 직사각형 [0, 2πa] × [0, 2πb]. 펼치는 사상
    (a cos t, a sin t, z) ↦ (at, z)가 나선을 선분 (at, bt)로 보낸다. 길이 = √((2πa)² + (2πb)²) = 2π√(a² + b²).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/04-curves/figures/fig-4-2-2-helix-unrolled.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["helix"]
(t,) = ex["coords"]
a, b = ex["params"]
A, B = 1.0, 0.3
gam = sp.lambdify(t, list(ex["expr"].subs({a: A, b: B})), "numpy")
speed_sym = sp.sqrt(ex["expr"].diff(t).dot(ex["expr"].diff(t)))

ts = np.linspace(0, 2 * np.pi, 600)
P = np.array([np.broadcast_to(c, ts.shape) for c in gam(ts)]).T
L_formula = 2 * np.pi * np.hypot(A, B)

# 자기검사: 수치로 잰 길이 = 공식, 펼친 곡선의 길이 = 3D 곡선의 길이
L_num = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
assert abs(L_num - L_formula) < 1e-4
U = np.stack([A * ts, P[:, 2]], axis=1)                        # 펼친 좌표 (at, z)
assert np.allclose(U[:, 1], B * ts)                             # 직선 z = (b/a)·(at)
L_flat = float(np.sum(np.linalg.norm(np.diff(U, axis=0), axis=1)))
assert abs(L_flat - L_formula) < 1e-9          # 펼친 선분은 정확히 공식의 길이
assert sp.simplify(speed_sym - sp.sqrt(a ** 2 + b ** 2)) == 0

fig = plt.figure(figsize=(7.0, 3.0))

# (a) 3D -------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.33, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=16, azim=-60)
ax.set_axis_off()
H = 2 * np.pi * B
uu, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(-0.12, H + 0.12, 12))
dgfig.surface(ax, A * np.cos(uu), A * np.sin(uu), zz, alpha=0.16, grid_every=6, zorder=1, shade=False)
normals = np.stack([np.cos(ts), np.sin(ts), 0 * ts], axis=1)
vis = dgfig.visible_mask(ax, P, normals)
dgfig.curve3d(ax, P, visible=vis, zorder=6)
ax.plot([A, A], [0, 0], [-0.12, H + 0.12], color=C["aux"], lw=1.0, zorder=5)
dgfig.point3d(ax, P[0], size=12, zorder=10)
dgfig.point3d(ax, P[-1], size=12, zorder=10)
ax.text(*(P[0] + np.array([0.0, -0.12, 0.08])), r"$\gamma(0)$", fontsize=11, ha="right")
ax.text(*(P[-1] + np.array([0.1, -0.15, 0.12])), r"$\gamma(2\pi)$", fontsize=11)
dgfig.equal_aspect(ax, P, np.array([[1, 1, -0.2], [-1, -1, H + 0.2]]), zoom=1.3)
ax.text2D(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) 펼친 직사각형 -------------------------------------------------------------------
ax = fig.add_axes([0.37, 0.12, 0.62, 0.8])
W = 2 * np.pi * A
ax.fill([0, W, W, 0], [0, 0, H, H], color=C["surface"], alpha=0.35, lw=0, zorder=0)
ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color=C["aux"], lw=0.8, zorder=1)
ax.plot(U[:, 0], U[:, 1], color=C["main"], lw=LW["main"], zorder=3)
ax.plot([0, W], [0, H], "o", color=C["main"], ms=4, zorder=4)
ax.text(W / 2, -0.12, r"$2\pi a$", ha="center", va="top", fontsize=12)
ax.text(W + 0.12, H / 2, r"$2\pi b$", ha="left", va="center", fontsize=12)
ax.text(W * 0.45, H * 0.45 + 0.2, r"$2\pi\sqrt{a^2+b^2}$", ha="right", va="bottom", fontsize=12,
        rotation=np.degrees(np.arctan2(H, W)), rotation_mode="anchor")
ax.text(-0.1, 0.0, r"$\gamma(0)$", ha="right", va="center", fontsize=11)
ax.text(W - 0.1, H + 0.1, r"$\gamma(2\pi)$", ha="right", va="bottom", fontsize=11)
dgfig.schematic_axes(ax, (-1.1, W + 1.0), (-0.55, H + 0.5))
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

dgfig.save(fig, __file__)

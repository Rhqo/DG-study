"""그림 3.4.1: 정사각형의 몫공간으로서의 원환면 (3.4절, 예 3.4.10).

왼쪽: [0,1]^2에서 (0, t) ~ (1, t) (세로 변, 파랑 화살표 하나)와 (s, 0) ~ (s, 1) (가로 변, 주황 화살표 둘)을 붙인다.
오른쪽: 기준 매개화 x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u) (R = 2, r = 0.8)에
u = 2πs, v = 2πt를 넣은 사상 (s, t) ↦ x(2πs, 2πt)의 상. 세로 변 s = 0, 1은 바깥 적도(u = 0, 파랑)로,
가로 변 t = 0, 1은 경선(v = 0, 주황)으로 간다. 점 (s, t) = (0.3, 0.2)와 그 상을 표시했다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-4-1-torus-square.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from dgsym import EXAMPLES
import sympy as sp

C = dgfig.COLORS
LW = dgfig.LW

# 기준 매개화 (§7)를 EXAMPLES에서 가져와 수치 함수로 만든다
ex = EXAMPLES["torus"]
u_, v_ = ex["coords"]
R_, r_ = ex["params"]
Rv, rv = 2.0, 0.8
xfun = sp.lambdify((u_, v_), list(ex["expr"].subs({R_: Rv, r_: rv})), "numpy")


def X(u, v):
    out = xfun(u, v)
    return np.stack([np.broadcast_to(np.asarray(c, float), np.broadcast(u, v).shape) for c in out], axis=-1)


def h(s, t):
    """(s, t) ∈ [0,1]^2 ↦ x(2πs, 2πt)"""
    return X(2 * np.pi * np.asarray(s, float), 2 * np.pi * np.asarray(t, float))


# 자기검사: 붙이는 변의 상이 같다, 점이 원환면 방정식 (sqrt(x^2+y^2) - R)^2 + z^2 = r^2를 만족
ts = np.linspace(0, 1, 101)
assert np.allclose(h(0 * ts, ts), h(0 * ts + 1, ts))
assert np.allclose(h(ts, 0 * ts), h(ts, 0 * ts + 1))
P = h(*np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30)))
assert np.allclose((np.hypot(P[..., 0], P[..., 1]) - Rv) ** 2 + P[..., 2] ** 2, rv ** 2)
# 세로 변 s = 0의 상은 바깥 적도(반지름 R + r, z = 0), 가로 변 t = 0의 상은 xz평면의 경선
E0 = h(0 * ts, ts)
assert np.allclose(np.hypot(E0[:, 0], E0[:, 1]), Rv + rv) and np.allclose(E0[:, 2], 0)
M0 = h(ts, 0 * ts)
assert np.allclose(M0[:, 1], 0)

fig = plt.figure(figsize=(6.6, 3.0))
ax0 = fig.add_axes([0.02, 0.1, 0.3, 0.8])
ax1 = fig.add_axes([0.36, -0.08, 0.64, 1.16], projection="3d", computed_zorder=False)
ax1.view_init(elev=32, azim=-62)
ax1.set_axis_off()

# 왼쪽: 정사각형 ---------------------------------------------------------------------
ax = ax0
dgfig.schematic_axes(ax, (-0.25, 1.25), (-0.25, 1.25))
ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=C["region"], alpha=dgfig.ALPHA["region"], lw=0))
ax.plot([0, 0], [0, 1], color=C["tangent"], lw=2.0)
ax.plot([1, 1], [0, 1], color=C["tangent"], lw=2.0)
ax.plot([0, 1], [0, 0], color=C["accent"], lw=2.0)
ax.plot([0, 1], [1, 1], color=C["accent"], lw=2.0)


def arrowhead(ax, xy, d, color, off=0.0):
    xy = np.asarray(xy, float); d = np.asarray(d, float)
    ax.annotate("", xy=xy + 0.06 * d + off, xytext=xy - 0.06 * d + off,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=12))


for x0 in (0, 1):
    arrowhead(ax, (x0, 0.5), (0, 1), C["tangent"])
for y0 in (0, 1):
    arrowhead(ax, (0.44, y0), (1, 0), C["accent"])
    arrowhead(ax, (0.56, y0), (1, 0), C["accent"])
sp0 = (0.3, 0.2)
ax.plot(*sp0, "o", ms=4, color=C["main"], zorder=5)
ax.text(sp0[0] + 0.04, sp0[1] + 0.04, r"$(s,t)$", fontsize=10)
ax.text(-0.08, -0.12, r"$0$", fontsize=10)
ax.text(1.0, -0.14, r"$1$", fontsize=10, ha="center")
ax.text(-0.14, 1.0, r"$1$", fontsize=10, va="center")
ax.text(0.5, -0.2, r"$s$", fontsize=11, ha="center")
ax.text(-0.2, 0.5, r"$t$", fontsize=11, va="center")

# 오른쪽: 원환면 ---------------------------------------------------------------------
ax = ax1
uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 73), np.linspace(0, 2 * np.pi, 97))
S = X(uu, vv)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.35, grid_every=6, zorder=1)
# 바깥 적도 (u = 0)와 경선 (v = 0); 보는 쪽 판정은 바깥 법벡터로
vw = dgfig.view_vector(ax)


def outward(u, v):
    return np.stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], -1)


tt = np.linspace(0, 2 * np.pi, 400)
eq = X(0 * tt, tt)
vis_eq = outward(0 * tt, tt) @ vw > 0
dgfig.curve3d(ax, eq, role="tangent", lw=2.0, visible=vis_eq, zorder=6)
mer = X(tt, 0 * tt)
vis_m = outward(tt, 0 * tt) @ vw > 0
# 경선은 안쪽 반이 몸통에 가려지는지를 대략 바깥 법벡터로 판정한다(개념 표시용)
dgfig.curve3d(ax, mer, role="accent", lw=2.0, visible=vis_m, zorder=6)
p3 = h(*sp0)
dgfig.point3d(ax, p3, role="main", size=16, zorder=12)
ax.text(*(p3 + np.array([0.1, 0.0, 0.35])), r"$\mathbf{x}(2\pi s, 2\pi t)$", fontsize=10, zorder=13)
dgfig.equal_aspect(ax, S[..., 0], S[..., 1], S[..., 2], zoom=1.35)

dgfig.map_arrow(fig, ax0, ax1, "", xy_from=(1.0, 0.75), xy_to=(0.12, 0.72), rad=-0.25)
dgfig.save(fig, __file__)

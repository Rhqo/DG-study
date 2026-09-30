"""그림 7.2.1: 평면을 원기둥에 감는 국소 등거리사상 (7.2절, 예 7.2.9).

φ(x, y, 0) = (r cos(x/r), r sin(x/r), y) (06장 연습 6.5.4), r = 1.
원기둥의 기준 매개화 x(u, v) = (r cos u, r sin u, v)는 dgsym.EXAMPLES["cylinder"]이고 φ(x, y, 0) = x(x/r, y)이다.
왼쪽: 평면의 띠 −0.4 ≤ x ≤ 2πr + 2.3, 0 ≤ y ≤ 2πb (b = 0.3)와 격자.
      선분 σ(t) = (rt, bt, 0), 0 ≤ t ≤ 2π (주황), 삼각형 T(파랑)와 그것을 2πr만큼 옮긴 삼각형 T'(파랑 점선).
      세로 점선 x = 0, x = 2πr는 띠 P₀ = {0 < x < 2πr}의 경계다.
오른쪽: 원기둥과 격자, σ의 상인 나선(주황), T와 T'의 공통의 상 φ(T) = φ(T')(파랑).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-2-1-cylinder-unrolling.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["cylinder"]
cu, cv = ex["coords"]
(rs,) = ex["params"]
RV = 1.0
BV = 0.3
Xe = ex["expr"].subs(rs, RV)
X = sp.lambdify((cu, cv), list(Xe), "numpy")


def phi(P):
    """P: (..., 2) 평면 좌표 (x, y) → 원기둥 위의 점 φ(x, y, 0) = x(x/r, y)."""
    P = np.asarray(P, float)
    return np.stack([np.broadcast_to(c, P.shape[:-1]) for c in X(P[..., 0] / RV, P[..., 1])], axis=-1)


# 자기검사: φ는 선분의 길이를 보존한다 (국소 등거리)
t = np.linspace(0, 2 * np.pi, 4001)
seg = np.stack([RV * t, BV * t], axis=1)
hel = phi(seg)
Lseg = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
Lhel = np.sum(np.linalg.norm(np.diff(hel, axis=0), axis=1))
assert np.isclose(Lseg, 2 * np.pi * np.hypot(RV, BV), rtol=1e-6) and np.isclose(Lhel, Lseg, rtol=1e-5)
assert np.allclose(np.hypot(hel[:, 0], hel[:, 1]), RV)          # 상은 원기둥 위에 있다
tri = np.array([[0.9, 0.3], [2.0, 0.45], [1.3, 1.25]])
tri2 = tri + np.array([2 * np.pi * RV, 0.0])
assert np.allclose(phi(tri), phi(tri2))                          # φ는 단사가 아니다


def edges(T, n=60):
    out = []
    for i in range(3):
        a, b = T[i], T[(i + 1) % 3]
        s = np.linspace(0, 1, n)[:, None]
        out.append(a + s * (b - a))
    return np.concatenate(out)


# 삼각형의 변의 길이와 꼭짓점의 각이 보존되는지 확인
def side_lengths(pts3):
    return [np.sum(np.linalg.norm(np.diff(pts3[i * 60:(i + 1) * 60], axis=0), axis=1)) for i in range(3)]


E3 = phi(edges(tri))
flat = [np.linalg.norm(tri[(i + 1) % 3] - tri[i]) for i in range(3)]
assert np.allclose(side_lengths(E3), flat, rtol=1e-3)

fig = plt.figure(figsize=(6.8, 3.0))

# 왼쪽: 평면 --------------------------------------------------------------------------------
axP = fig.add_axes([0.01, 0.05, 0.56, 0.9])
x0, x1, y1 = -0.4, 2 * np.pi * RV + 2.3, 2 * np.pi * BV
axP.fill([x0, x1, x1, x0], [0, 0, y1, y1], color=C["region"], alpha=0.18, lw=0)
for k in np.arange(-0.5, x1, 0.5):
    if k >= x0:
        axP.plot([k, k], [0, y1], color="#A0A0A0", lw=0.35)
for k in np.arange(0.0, y1 + 1e-9, 0.5):
    axP.plot([x0, x1], [k, k], color="#A0A0A0", lw=0.35)
for xv in (0.0, 2 * np.pi * RV):
    axP.plot([xv, xv], [-0.15, y1 + 0.15], color=C["aux"], lw=0.9, ls=(0, (4, 3)))
axP.plot(seg[:, 0], seg[:, 1], color=C["accent"], lw=1.8)
axP.fill(*tri.T, color=C["tangent"], alpha=0.25, lw=0)
axP.plot(*np.vstack([tri, tri[:1]]).T, color=C["tangent"], lw=1.4)
axP.plot(*np.vstack([tri2, tri2[:1]]).T, color=C["tangent"], lw=1.1, ls=(0, (3, 2)))
axP.text(*(tri.mean(0) + np.array([0.0, -0.03])), r"$T$", fontsize=11, ha="center", va="center")
axP.text(*(tri2.mean(0) + np.array([0.0, -0.03])), r"$T'$", fontsize=11, ha="center", va="center")
axP.text(2 * np.pi * RV * 0.52, BV * np.pi * 1.04 + 0.12, r"$\sigma$", color=C["accent"], fontsize=12)
axP.text(0.0, -0.3, r"$0$", fontsize=10, ha="center", va="top")
axP.text(2 * np.pi * RV, -0.3, r"$2\pi r$", fontsize=10, ha="center", va="top")
axP.text((x0 + x1) / 2, y1 + 0.35, r"$P = \{z = 0\}$", fontsize=12, ha="center")
dgfig.schematic_axes(axP, (x0 - 0.2, x1 + 0.2), (-0.8, y1 + 0.8))

# 오른쪽: 원기둥 -------------------------------------------------------------------------------
ax = fig.add_axes([0.6, 0.0, 0.4, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=22, azim=95)
ax.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 97), np.linspace(-0.05, y1 + 0.05, 13))
S = np.stack(X(Ug, Vg), axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.22, grid=False, zorder=1)


def vis(P):
    return dgfig.visible_mask(ax, P, np.stack([P[:, 0], P[:, 1], 0 * P[:, 2]], axis=1))


s = np.linspace(0, 2 * np.pi, 300)
for k in range(12):
    L = np.stack(X(k * np.pi / 6 * np.ones_like(s[:2]), np.array([0.0, y1])), axis=-1)
    dgfig.curve3d(ax, L, role="#9A9A9A", lw=0.4, visible=vis(L), hidden=None, zorder=3)
for k in np.arange(0.0, y1 + 1e-9, 0.5):
    L = np.stack(X(s, k * np.ones_like(s)), axis=-1)
    dgfig.curve3d(ax, L, role="#9A9A9A", lw=0.4, visible=vis(L), hidden=None, zorder=3)
dgfig.curve3d(ax, hel, role="accent", lw=1.8, visible=vis(hel), hidden="dashed", zorder=6)
view = dgfig.view_vector(ax)
assert np.all(np.stack([E3[:, 0], E3[:, 1], 0 * E3[:, 2]], 1) @ view > 0)    # φ(T)는 보이는 쪽에 있다
poly = Poly3DCollection([E3 * np.array([1.003, 1.003, 1.0])], facecolor=C["tangent"], alpha=0.25,
                        edgecolor=C["tangent"], linewidth=1.4, zorder=7)
ax.add_collection3d(poly)
c3 = phi(tri.mean(0)) * np.array([1.05, 1.05, 1.0])
ax.text(*c3, r"$\varphi(T)$", fontsize=11, ha="center", va="center", zorder=15)
ax.text(*(phi(np.array([2 * np.pi * RV * 0.75, BV * 1.5 * np.pi])) * np.array([1.12, 1.12, 1.0])),
        r"$\varphi\circ\sigma$", color=C["accent"], fontsize=11, zorder=15)
dgfig.equal_aspect(ax, np.array([[-1, -1, 0], [1, 1, y1]]), zoom=1.05)

dgfig.map_arrow(fig, axP, ax, r"$\varphi$", xy_from=(0.93, 0.78), xy_to=(0.2, 0.78), rad=-0.35,
                label_offset=(0.0, -0.06))

dgfig.save(fig, __file__)

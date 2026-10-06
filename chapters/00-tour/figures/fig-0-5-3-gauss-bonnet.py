"""Figure 0.5.3: Gauss–Bonnet theorem의 두 얼굴 (0.5절, Theorem 0.5.5, Example 0.5.6, Exercise 0.5.3).

(a) unit sphere의 octant triangle: 꼭짓점 (1,0,0), (0,1,0), (0,0,1), 세 변은 great circle의 호(주황), 세 내각은 모두 π/2.
    내각의 합 3π/2 = π + ∫∫K dA = π + 1 · (4π/8).
(b) angle defect: 정이십면체(icosahedron)의 한 꼭짓점에는 정삼각형 5개가 모여 각의 합이 5 · π/3 = 5π/3이다.
    다섯 삼각형을 평면에 펼치면 각 δ = 2π − 5π/3 = π/3의 틈(주홍)이 남는다. 12개 꼭짓점의 δ의 합은 4π = 2π χ(S²).
자기검사: octant의 넓이(수치 적분) = π/2, 세 내각 = π/2, Σδ(icosahedron) = 4π, cube의 Σδ = 4π.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-5-3-gauss-bonnet.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Wedge

C = dgfig.COLORS
e = np.eye(3)


def arc(a, b, n=120):
    d = np.arccos(np.clip(a @ b, -1, 1))
    w = np.linspace(0, 1, n)
    return np.stack([(np.sin((1 - t) * d) * a + np.sin(t * d) * b) / np.sin(d) for t in w])


# 자기검사: octant ----------------------------------------------------------------------------
ang = []
for i in range(3):
    a, b, c = e[i], e[(i + 1) % 3], e[(i + 2) % 3]
    tb = b - (b @ a) * a
    tc = c - (c @ a) * a
    ang.append(np.arccos(tb @ tc / np.linalg.norm(tb) / np.linalg.norm(tc)))
assert np.allclose(ang, np.pi / 2)
thg = np.linspace(0, np.pi / 2, 801)
area = np.sum(0.5 * (np.sin(thg[1:]) + np.sin(thg[:-1])) * np.diff(thg)) * (np.pi / 2)
assert np.isclose(area, np.pi / 2, atol=1e-6)
assert np.isclose(sum(ang), np.pi + area)


# 자기검사: icosahedron과 cube의 angle defect 합 ------------------------------------------------
def defect_sum(V, F):
    s = np.zeros(len(V))
    for f in F:
        for k in range(len(f)):
            i, j, l = f[k], f[(k + 1) % len(f)], f[(k - 1) % len(f)]
            u, v = V[j] - V[i], V[l] - V[i]
            s[i] += np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
    return np.sum(2 * np.pi - s), 2 * np.pi - s


t_ = (1 + 5 ** 0.5) / 2
Vi = np.array([(-1, t_, 0), (1, t_, 0), (-1, -t_, 0), (1, -t_, 0), (0, -1, t_), (0, 1, t_), (0, -1, -t_), (0, 1, -t_),
               (t_, 0, -1), (t_, 0, 1), (-t_, 0, -1), (-t_, 0, 1)], float)
Fi = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
      (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7),
      (9, 8, 1)]
tot, each = defect_sum(Vi, Fi)
assert np.isclose(tot, 4 * np.pi) and np.allclose(each, np.pi / 3)
Vc = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
Fc = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
assert np.isclose(defect_sum(Vc, Fc)[0], 4 * np.pi)
assert len(Vi) - 30 + len(Fi) == 2                          # χ = n0 − n1 + n2 = 12 − 30 + 20

fig = plt.figure(figsize=(6.6, 3.2))
ax = fig.add_axes([0.0, 0.0, 0.52, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=24, azim=40)
ax.set_proj_type("ortho")
ax.set_axis_off()
T, P = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing="ij")
S = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)
To, Po = np.meshgrid(np.linspace(0, np.pi / 2, 30), np.linspace(0, np.pi / 2, 30), indexing="ij")
So = 1.003 * np.stack([np.sin(To) * np.cos(Po), np.sin(To) * np.sin(Po), np.cos(To)], axis=-1)
ax.plot_surface(So[..., 0], So[..., 1], So[..., 2], color=C["region"], alpha=0.45, linewidth=0, shade=False,
                rasterized=True, zorder=2)
for i in range(3):
    dgfig.curve3d(ax, 1.004 * arc(e[i], e[(i + 1) % 3]), role="accent", lw=2.0, zorder=6)
    a, b, c = e[i], e[(i + 1) % 3], e[(i + 2) % 3]
    m = 0.14
    corner = np.stack([a + m * b, a + m * b + m * c, a + m * c])
    corner = corner / np.linalg.norm(corner, axis=1)[:, None]
    ax.plot(*corner.T, color=C["main"], lw=0.9, zorder=7)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)
ax.text2D(0.03, 0.93, r"$(\mathrm{a})$", transform=ax.transAxes, fontsize=11)

ax2 = fig.add_axes([0.56, 0.08, 0.42, 0.84])
for k in range(5):
    a0, a1 = k * np.pi / 3, (k + 1) * np.pi / 3
    tri = np.array([[0, 0], [np.cos(a0), np.sin(a0)], [np.cos(a1), np.sin(a1)]])
    ax2.add_patch(Polygon(tri, closed=True, fc=C["surface"], ec=C["main"], lw=1.0, alpha=0.8, zorder=2))
    ax2.add_patch(Wedge((0, 0), 0.22, np.degrees(a0), np.degrees(a1), fc="none", ec=C["aux"], lw=0.7, zorder=3))
ax2.add_patch(Wedge((0, 0), 1.0, 300, 360, fc=C["normal"], alpha=0.18, ec="none", zorder=1))
ax2.add_patch(Wedge((0, 0), 0.3, 300, 360, fc="none", ec=C["normal"], lw=1.2, zorder=3))
ax2.text(0.42 * np.cos(np.deg2rad(330)), 0.42 * np.sin(np.deg2rad(330)), r"$\delta$", color=C["normal"], fontsize=13,
         ha="center", va="center")
ax2.text(0.33 * np.cos(np.deg2rad(150)), 0.33 * np.sin(np.deg2rad(150)), r"$\theta_j$", fontsize=11,
         ha="center", va="center")
ax2.plot(0, 0, "o", color=C["main"], ms=4, zorder=5)
dgfig.schematic_axes(ax2, (-1.1, 1.1), (-1.1, 1.1))
ax2.text(-1.08, 1.0, r"$(\mathrm{b})$", fontsize=11)

dgfig.save(fig, __file__)

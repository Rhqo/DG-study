"""Figure 42.5.2: the dual cell of a boundary vertex is clipped by the boundary (Section 42.5, Proposition 42.5.9).

Planar mesh piece: boundary vertex x_0 = (0, 0) on the boundary line y = 0, boundary neighbours x_1 = (1, 0),
x_4 = (-1, 0), interior neighbours x_2 = (0.45, 0.85), x_3 = (-0.5, 0.8); faces (0, 1, 2), (0, 2, 3), (0, 3, 4)
(counterclockwise; the mesh lies above y = 0). The circumcentric dual cell C_0 is the union of the right triangles
(x_0, m_0j, c_f) (light orange). Its boundary consists of the dual-edge pieces [m, c_f] (orange: the flux through them is
(L u)_0) and the two half boundary edges [m_04, x_0], [x_0, m_01] on y = 0 (vermilion, with the outward conormal n:
there the flux is not part of L u; it is the Neumann datum).
Self-check: for u = 2x + 3y (affine), the flux through the orange pieces equals (L u)_0, and adding the flux through
the vermilion pieces (= int <grad u, n> = -3 * 1) gives 0 = int_{C_0} Delta u (divergence theorem).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-5-2-clipped-cell.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

col = dgfig.COLORS
V = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.45, 0.85, 0], [-0.5, 0.8, 0], [-1.0, 0, 0]])
F = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
assert np.all(dm.face_cross(V, F)[:, 2] > 0) and dm.corner_angles(V, F).max() < np.pi / 2
C = dd.circumcenters(V, F)
L = dm.cotan_laplacian(V, F)
u = 2 * V[:, 0] + 3 * V[:, 1]
flux_dual = 0.0
for f, (i, j, k) in enumerate(F):
    g = (u[[i, j, k]] @ dd.barycentric_gradients(V[[i, j, k]]))[:2]
    for a, b in ((0.5 * (V[i] + V[j]), C[f]), (C[f], 0.5 * (V[i] + V[k]))):
        s = (b - a)[:2]
        flux_dual += g @ np.array([s[1], -s[0]])
assert abs(flux_dual - (L @ u)[0]) < 1e-14
flux_bdry = np.array([2.0, 3.0]) @ np.array([0.0, -1.0]) * 1.0       # the two half edges have total length 1
assert abs(flux_dual + flux_bdry) < 1e-14
print(f"(L u)_0 = {(L @ u)[0]:.4f}, boundary flux = {flux_bdry:.4f}, cell area = {dd.dual_areas(V, F, 'circumcentric')[0]:.4f}")

fig, ax = plt.subplots(figsize=(4.8, 2.9))
for f in F:
    ax.fill(*V[f][:, :2].T, color="#EEEEEE", zorder=0)
for a, b in dm.edges(F):
    ax.plot(*V[[a, b]][:, :2].T, color="#888888", lw=0.8, zorder=1)
ax.plot([-1.25, 1.25], [0, 0], color=col["main"], lw=1.6, zorder=1)
for f, (i, j, k) in enumerate(F):
    mij, mik = 0.5 * (V[i] + V[j]), 0.5 * (V[i] + V[k])
    for tri in ((V[i], mij, C[f]), (V[i], C[f], mik)):
        ax.add_patch(Polygon(np.array(tri)[:, :2], closed=True, facecolor=col["accent"], alpha=0.3, edgecolor="none",
                             zorder=2))
    ax.plot([mij[0], C[f][0], mik[0]], [mij[1], C[f][1], mik[1]], color=col["accent"], lw=1.8, zorder=3)
ax.plot([-0.5, 0.5], [0, 0], color=col["normal"], lw=3.2, zorder=4)
for x0 in (-0.25, 0.25):
    ax.annotate("", xy=(x0, -0.28), xytext=(x0, 0.0), arrowprops=dict(arrowstyle="->", color=col["normal"], lw=1.4),
                zorder=5)
ax.text(0.3, -0.27, r"$\mathbf{n}$", fontsize=11, color=col["normal"])
ax.plot(*C[:, :2].T, "o", color=col["accent"], ms=3.5, zorder=6)
ax.plot(V[:, 0], V[:, 1], "o", color=col["main"], ms=3.5, zorder=7)
for idx, off in enumerate([(-0.07, -0.16), (0.03, 0.06), (0.04, 0.03), (-0.12, 0.04), (-0.08, 0.06)]):
    ax.text(V[idx, 0] + off[0], V[idx, 1] + off[1], rf"$\mathbf{{x}}_{idx}$", fontsize=10, zorder=8)
ax.text(-0.15, 0.25, r"$C_0$", fontsize=11, color="#9A6A00")
ax.text(0.85, -0.2, r"$\partial S$", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-0.35, 1.0)
ax.set_axis_off()
dgfig.save(fig, __file__)

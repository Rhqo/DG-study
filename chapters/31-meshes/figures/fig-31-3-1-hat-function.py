"""Figure 31.3.1: hat functions and their gradients (Section 31.3).

(a) The hat function phi_i of an interior vertex i of a planar mesh (value 1 at i, 0 at every other vertex,
    linear on each triangle), drawn as a tent over the mesh (orthographic view).
(b) One triangle ijk with corners x_i = (0, 0), x_j = (2.4, 0), x_k = (0.7, 1.5). On this triangle phi_i and phi_j
    have constant gradients: grad phi_i is perpendicular to the opposite edge jk, points toward i, and has length
    1 / h_i (h_i = distance from x_i to the line jk); the same for phi_j. theta_k is the angle at k.
Self-check: <grad phi_i, grad phi_j> * Area = -cot(theta_k) / 2 and |grad phi_i|^2 * Area = (cot theta_j + cot theta_k) / 2.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-3-1-hat-function.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1.0], wspace=0.05))

# ---------------------------------------------------------------- (a) tent over a small planar mesh
ang = np.radians([10, 70, 125, 180, 235, 300])
rad = np.array([1.0, 0.9, 1.1, 0.95, 1.05, 0.9])
ring = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.zeros(6)], 1)
outer = np.stack([1.9 * np.cos(ang + 0.5), 1.9 * np.sin(ang + 0.5), np.zeros(6)], 1)
V = np.vstack([[0, 0, 0], ring, outer])
F = [[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)]
for k in range(6):
    a, b, o = 1 + k, 1 + (k + 1) % 6, 7 + k
    F += [[a, o, b]]
    F += [[o, 7 + (k + 1) % 6, b]]
F = np.array(F)
phi = np.zeros(len(V))
phi[0] = 1.0
cam = dm.Ortho(elev=28, azim=-62)
flat = dm.shade(V, F, dm.hex_to_rgb(C["surface"]), light=(0, 0, 1), ambient=1.0)
dm.draw_mesh_2d(ax, V, F, cam, flat, edgecolor="#666666", lw=0.7)
T = V.copy()
T[:, 2] = 1.6 * phi
star = F[:6]
cols = dm.shade(T, star, dm.hex_to_rgb(C["region"]), light=(0.3, -0.6, 0.8), ambient=0.55)
dm.draw_mesh_2d(ax, T, star, cam, cols, edgecolor=C["tangent"], lw=0.9, zorder=3)
q0, q1 = cam(V[0]), cam(T[0])
ax.plot(*q1, "o", color=C["main"], ms=4, zorder=5)
ax.text(q1[0] + 0.08, q1[1] + 0.05, r"$\varphi_i(x_i) = 1$", fontsize=11)
P = cam(V)
ax.text(P[9, 0] - 0.1, P[9, 1] - 0.25, r"$\varphi_i = 0$", fontsize=11, color=C["aux"])
ax.set_xlim(P[:, 0].min() - 0.1, P[:, 0].max() + 0.1)
ax.set_ylim(P[:, 1].min() - 0.1, cam(T)[:, 1].max() + 0.2)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(r"(a) hat function $\varphi_i$", fontsize=11.5)

# ---------------------------------------------------------------- (b) gradients on one triangle
xi, xj, xk = np.array([0.0, 0.0]), np.array([2.4, 0.0]), np.array([0.7, 1.5])
axb.add_patch(Polygon([xi, xj, xk], closed=True, fc=C["surface"], ec=C["main"], lw=1.2, alpha=0.6))


def grad_hat(p, a, b):
    """Gradient of the hat function of p on triangle (p, a, b): perpendicular to ab, toward p, length 1/h."""
    e = b - a
    n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
    h = np.dot(p - a, n)
    if h < 0:
        n, h = -n, -h
    return n / h, h


gi, hi = grad_hat(xi, xj, xk)
gj, hj = grad_hat(xj, xk, xi)
d1, d2 = xj - xi, xk - xi
area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
tk = np.arccos(np.dot(xi - xk, xj - xk) / np.linalg.norm(xi - xk) / np.linalg.norm(xj - xk))
tj = np.arccos(np.dot(xi - xj, xk - xj) / np.linalg.norm(xi - xj) / np.linalg.norm(xk - xj))
assert abs(np.dot(gi, gj) * area + 0.5 / np.tan(tk)) < 1e-12
assert abs(np.dot(gi, gi) * area - 0.5 * (1 / np.tan(tj) + 1 / np.tan(tk))) < 1e-12
c = (xi + xj + xk) / 3
S = 0.9
for g, col, lab, off in ((gi, C["tangent"], r"$\nabla\varphi_i$", (-0.33, -0.05)),
                         (gj, C["third"], r"$\nabla\varphi_j$", (0.06, -0.05))):
    axb.add_patch(FancyArrowPatch(c, c + S * g, arrowstyle="-|>", mutation_scale=12, color=col, lw=2.0, zorder=4))
    axb.text(*(c + S * g + np.array(off)), lab, fontsize=12, color=col)
# height h_i: foot of perpendicular from x_i to jk
e = xk - xj
foot = xj + np.dot(xi - xj, e) / np.dot(e, e) * e
axb.plot(*np.array([xi, foot]).T, color=C["aux"], lw=1.0, ls=(0, (4, 3)))
axb.text(*(0.55 * (xi + foot) + np.array([0.02, 0.08])), r"$h_i$", fontsize=12, color=C["aux"])
ak = np.degrees(np.arctan2(*(xi - xk)[::-1]))
aj = np.degrees(np.arctan2(*(xj - xk)[::-1]))
axb.add_patch(Arc(xk, 0.5, 0.5, theta1=min(ak, aj), theta2=max(ak, aj), color=C["accent"], lw=1.8))
axb.text(xk[0] - 0.05, xk[1] - 0.48, r"$\theta_k$", fontsize=12, color=C["accent"])
for p, lab, off in ((xi, r"$x_i$", (-0.25, -0.15)), (xj, r"$x_j$", (0.05, -0.15)), (xk, r"$x_k$", (-0.05, 0.08))):
    axb.plot(*p, "o", color=C["main"], ms=4)
    axb.text(*(p + np.array(off)), lab, fontsize=12)
axb.text(1.2, -0.55, r"$|\nabla\varphi_i| = 1/h_i$,  $\nabla\varphi_i \perp x_jx_k$", ha="center", fontsize=10.5)
axb.set_xlim(-0.4, 2.7)
axb.set_ylim(-0.75, 1.75)
axb.set_aspect("equal")
axb.set_axis_off()
axb.set_title("(b) gradients on one triangle", fontsize=11.5)

fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
dgfig.save(fig, __file__)

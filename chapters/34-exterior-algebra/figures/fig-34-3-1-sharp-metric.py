"""Figure 34.3.1: one covector, two inner products, two sharps (Section 34.3, Proposition 34.3.1,
Example 34.3.4).

The covector alpha = e^1 + e^2 on R^2 (alpha(x) = x^1 + x^2) is drawn by its level lines {alpha = c},
c = -3, ..., 3 (purple).
(a) Euclidean inner product: alpha^sharp = (1, 1) is perpendicular to the level lines, |alpha^sharp| = sqrt 2 and
    the circle of radius sqrt 2 touches the level line {alpha = 2} exactly at alpha^sharp. The orange test
    vector u = (2.2, 0.4) has alpha(u) = 2.6 = |alpha^sharp| * (signed length 2.6/sqrt 2 of its projection
    onto the line of alpha^sharp, dashed drop).
(b) Inner product g(x, y) = x^T G y, G = [[2, 1], [1, 1]] (Section 34.2): alpha^sharp_g = G^{-1}(1, 1) = (0, 1).
    It is g-perpendicular to the level lines; the g-circle {x^T G x = 1} (radius |alpha^sharp_g|_g = 1)
    touches the level line {alpha = 1} exactly at alpha^sharp_g.
Self-checks: alpha(alpha^sharp) = |alpha^sharp|^2 in both metrics, tangency of the circles to the level lines,
g(alpha^sharp_g, d) = 0 for the level-line direction d = (1, -1), and the projection identity for u.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
a = np.array([1.0, 1.0])                          # components of alpha
G = np.array([[2.0, 1.0], [1.0, 1.0]])
sh_e = a.copy()
sh_g = np.linalg.solve(G, a)
assert np.allclose(sh_g, [0, 1])
d = np.array([1.0, -1.0])                         # direction of the level lines
assert abs(sh_e @ d) < 1e-12 and abs(sh_g @ G @ d) < 1e-12
assert abs(a @ sh_e - sh_e @ sh_e) < 1e-12 and abs(a @ sh_g - sh_g @ G @ sh_g) < 1e-12
u = np.array([2.2, 0.4])
proj = (u @ sh_e) / np.linalg.norm(sh_e)
assert abs(a @ u - np.linalg.norm(sh_e) * proj) < 1e-12 and abs(a @ u - 2.6) < 1e-12


def arrow(ax, base, vec, color, lw=1.7, zorder=6):
    ax.add_patch(FancyArrowPatch(tuple(base), tuple(np.asarray(base) + vec), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))


def level_lines(ax, lim):
    s = np.linspace(-lim, lim, 2)
    for c in range(-3, 4):
        # x + y = c
        ax.plot(s, c - s, color=C["covector"], lw=0.9 if c else 1.3, alpha=0.85, zorder=1)
    for c, (x, y) in zip((1, 2), ((2.45, -1.45), (2.95, -0.95))):
        ax.text(x, y, rf"$\alpha={c}$", color=C["covector"], fontsize=10, ha="left")


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 3.4))
o = np.zeros(2)
t = np.linspace(0, 2 * np.pi, 300)

# (a)
level_lines(axA, 4)
r = np.linalg.norm(sh_e)
axA.plot(r * np.cos(t), r * np.sin(t), color=C["aux"], lw=0.9, ls=(0, (2, 2)), zorder=2)
arrow(axA, o, sh_e, C["tangent"])
axA.text(0.55, 1.35, r"$\alpha^\sharp$", color=C["tangent"], fontsize=13)
arrow(axA, o, u, C["accent"], lw=1.5)
axA.text(2.25, 0.55, r"$u$", color=C["accent"], fontsize=13)
foot = proj * sh_e / np.linalg.norm(sh_e)
axA.plot([u[0], foot[0]], [u[1], foot[1]], color=C["accent"], lw=0.9, ls="--", zorder=3)
axA.plot([0, 2.3], [0, 2.3], color=C["tangent"], lw=0.6, ls=":", zorder=2)
axA.scatter(*sh_e, s=14, color=C["main"], zorder=7)
dgfig.schematic_axes(axA, xlim=(-2.0, 3.4), ylim=(-1.75, 2.7))
axA.set_title("(a) Euclidean", fontsize=12)

# (b)
level_lines(axB, 4)
w_, V_ = np.linalg.eigh(G)
Gm12 = V_ @ np.diag(1 / np.sqrt(w_)) @ V_.T
ell = (Gm12 @ np.stack([np.cos(t), np.sin(t)])).T
assert np.allclose(np.einsum("ij,jk,ik->i", ell, G, ell), 1.0)
assert np.all(ell @ a <= 1 + 1e-9)                # the g-circle stays in {alpha <= 1}: tangency at sh_g
axB.plot(*ell.T, color=C["aux"], lw=0.9, ls=(0, (2, 2)), zorder=2)
arrow(axB, o, sh_g, C["tangent"])
axB.text(-1.95, 1.55, r"$\alpha^{\sharp_g} = (0,1)$", color=C["tangent"], fontsize=12)
arrow(axB, o, sh_e, "#9A9A9A", lw=1.1, zorder=5)
axB.text(1.05, 0.95, r"$\alpha^\sharp$", color="#777777", fontsize=11)
axB.scatter(*sh_g, s=14, color=C["main"], zorder=7)
axB.text(0.75, -1.55, r"$x^{\mathsf{T}}Gx = 1$", color="#555555", fontsize=10)
dgfig.schematic_axes(axB, xlim=(-2.0, 3.4), ylim=(-1.75, 2.7))
axB.set_title(r"(b) inner product $G$", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.06)
dgfig.save(fig, __file__)

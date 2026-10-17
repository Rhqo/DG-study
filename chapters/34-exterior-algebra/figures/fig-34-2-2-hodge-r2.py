"""Figure 34.2.2: the Hodge star on vectors in R^2 (Section 34.2, Example 34.2.7 and Example 34.2.10).

(a) Euclidean inner product, standard orientation: star is the counterclockwise quarter turn. Starting from
    u = b1 = (2, 1): star u = (-1, 2), star star u = -u, star star star u = (1, -2). The dotted circle has
    radius |u| = sqrt 5; each star vector is tangent to it at the previous one.
(b) Inner product g(x, y) = x^T G y with G = [[2, 1], [1, 1]] (det G = 1), same orientation. star_g e1 = (-1, 2):
    g-orthogonal to e1 and of the same g-length sqrt 2. The ellipse is the g-circle {x^T G x = 2} through e1;
    star_g e1 is parallel to its tangent at e1 (dashed). The Euclidean star e1 = e2 is drawn in gray for comparison.
Self-checks: the four Euclidean iterates, g(e1, star_g e1) = 0, |star_g e1|_g = |e1|_g, det[e1, star_g e1] > 0,
and the tangent of the g-circle at e1 is parallel to star_g e1.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
J = np.array([[0.0, -1.0], [1.0, 0.0]])          # Euclidean star on R^2 (counterclockwise quarter turn)
u = np.array([2.0, 1.0])
its = [u, J @ u, J @ J @ u, J @ J @ J @ u]
assert np.allclose(its[1], [-1, 2]) and np.allclose(its[2], -u) and np.allclose(its[3], [1, -2])

G = np.array([[2.0, 1.0], [1.0, 1.0]])
e1, e2 = np.eye(2)
sg = np.array([-1.0, 2.0])                         # star_g e1
assert abs(e1 @ G @ sg) < 1e-12 and abs(sg @ G @ sg - e1 @ G @ e1) < 1e-12 and e1[0] * sg[1] - e1[1] * sg[0] > 0
# tangent of {x^T G x = 2} at e1 is the direction d with e1^T G d = 0
tangent = np.array([-(G[0, 1]), G[0, 0]])
assert abs(tangent[0] * sg[1] - tangent[1] * sg[0]) < 1e-12


def arrow(ax, base, vec, color, ls="-", lw=1.6, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(base), tuple(np.asarray(base) + vec), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=zorder))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 3.3))
o = np.zeros(2)

t = np.linspace(0, 2 * np.pi, 300)
r = np.linalg.norm(u)
axA.plot(r * np.cos(t), r * np.sin(t), color=C["aux"], lw=0.8, ls=(0, (2, 2)))
styles = [(C["tangent"], "-"), (C["third"], "-"), (C["tangent"], "--"), (C["third"], "--")]
labels = [r"$u$", r"$\star u$", r"$\star\star u = -u$", r"$\star\star\star u$"]
offs = [(0.25, 0.15), (-0.15, 0.3), (0.1, -0.4), (0.55, -0.2)]
for vec, (col, ls), lab, off in zip(its, styles, labels, offs):
    arrow(axA, o, vec, col, ls=ls)
    axA.text(vec[0] + off[0], vec[1] + off[1], lab, color=col, fontsize=12, ha="center", va="center")
# quarter-turn arc from u to star u
a0 = np.arctan2(u[1], u[0])
ts = np.linspace(a0 + 0.12, a0 + np.pi / 2 - 0.12, 40)
arc = 0.8 * np.stack([np.cos(ts), np.sin(ts)], 1)
axA.plot(*arc.T, color=C["main"], lw=1.0)
axA.add_patch(FancyArrowPatch(tuple(arc[-4]), tuple(arc[-1]), arrowstyle="-|>", mutation_scale=9,
                              color=C["main"], lw=1.0, shrinkA=0, shrinkB=0))
axA.text(0.95, 0.95, r"$90^\circ$", fontsize=11)
dgfig.schematic_axes(axA, xlim=(-3.3, 3.3), ylim=(-3.0, 3.0))
axA.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
axA.axvline(0, color="#BBBBBB", lw=0.5, zorder=0)
axA.set_title("(a) Euclidean: quarter turn", fontsize=12)

# (b) g-circle x^T G x = 2: parametrize with G^{-1/2}
w_, V_ = np.linalg.eigh(G)
Ginvhalf = V_ @ np.diag(1 / np.sqrt(w_)) @ V_.T
ell = (np.sqrt(2) * Ginvhalf @ np.stack([np.cos(t), np.sin(t)])).T
assert np.allclose(np.einsum("ij,jk,ik->i", ell, G, ell), 2.0)
axB.plot(*ell.T, color=C["aux"], lw=0.8, ls=(0, (2, 2)))
axB.plot([1 - 0.9 * tangent[0] / 2, 1 + 0.9 * tangent[0] / 2], [-0.9 * tangent[1] / 2, 0.9 * tangent[1] / 2],
         color=C["aux"], lw=0.9, ls="--")
arrow(axB, o, e1, C["tangent"])
arrow(axB, o, sg, C["third"])
arrow(axB, o, e2, "#999999", lw=1.2, zorder=4)
axB.text(1.2, -0.22, r"$e_1$", color=C["tangent"], fontsize=12)
axB.text(-1.35, 2.15, r"$\star_g e_1 = (-1,2)$", color=C["third"], fontsize=12)
axB.text(0.06, 1.18, r"$\star e_1 = e_2$", color="#777777", fontsize=11)
axB.text(1.05, -1.6, r"$x^{\mathsf{T}}Gx = 2$", color="#555555", fontsize=11)
dgfig.schematic_axes(axB, xlim=(-2.4, 2.8), ylim=(-2.3, 2.6))
axB.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
axB.axvline(0, color="#BBBBBB", lw=0.5, zorder=0)
axB.set_title(r"(b) inner product $G$", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.08)
dgfig.save(fig, __file__)

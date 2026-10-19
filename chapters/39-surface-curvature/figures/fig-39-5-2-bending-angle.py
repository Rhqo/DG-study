"""Figure 39.5.2: the sign of the bending angle.

Cross-section perpendicular to an interior edge ij, seen with e_ij = (x_j - x_i)/l_ij pointing toward the viewer
(the dot at the origin). The face f = (i, j, k), which traverses i -> j, lies to the right with unit normal N_f
(up); the other face g lies in the direction -(N_g x e_ij). N_g = R(e_ij, theta) N_f. (a) theta = +0.6: the
surface bends away from N_f (convex edge). (b) theta = -0.6: it bends toward N_f (concave edge). Both pictures are
computed from actual triangles in R^3 with dgmesh.edge_bending_angles.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-5-2-bending-angle.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402

C = dgfig.COLORS


def hinge(theta):
    """x_i = (0, 0, -0.5), x_j = (0, 0, 0.5): edge along +z (toward the viewer); f = (i, j, k) with k at (1.2, 0, 0)
    so N_f = e_y; g = (j, i, l) with l = 1.2 (-cos theta, -sin theta, 0)."""
    V = np.array([[0, 0, -0.5], [0, 0, 0.5], [1.2, 0, 0.1], [-1.2 * np.cos(theta), -1.2 * np.sin(theta), -0.1]])
    F = np.array([[0, 1, 2], [1, 0, 3]])
    return V, F


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
for ax, th, title in zip(axes, (0.6, -0.6), ("(a) convex edge", "(b) concave edge")):
    V, F = hinge(th)
    b = dm.edge_bending_angles(V, F)
    assert abs(b[(0, 1)][0] - th) < 1e-12                          # the drawn angle is the bending angle
    Nf, Ng = dm.face_normals(V, F)
    assert np.allclose(Nf, [0, 1, 0]) and abs(Ng[2]) < 1e-12
    k2, l2 = V[2, :2], V[3, :2]
    ax.plot([0, k2[0]], [0, 0], color=C["main"], lw=dgfig.LW["main"])
    ax.plot([0, l2[0]], [0, l2[1]], color=C["main"], lw=dgfig.LW["main"])
    mf, mg = 0.5 * k2, 0.5 * l2
    for p, n, lab, off in ((mf, Nf[:2], r"$\mathbf{N}_f$", (0.05, 0.05)), (mg, Ng[:2], r"$\mathbf{N}_g$", (-0.05, 0.08))):
        ax.annotate("", xy=p + 0.7 * n, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.5,
                                                                mutation_scale=11, shrinkA=0, shrinkB=0))
        ax.text(*(p + 0.7 * n + np.array(off)), lab, color=C["normal"], fontsize=12, ha="center")
    # copies at the edge and the arc from N_f to N_g
    for n in (Nf[:2], Ng[:2]):
        ax.plot([0, 0.55 * n[0]], [0, 0.55 * n[1]], color=C["normal"], lw=0.8, ls=(0, (3, 2)))
    a0, a1 = np.pi / 2, np.pi / 2 + th
    aa = np.linspace(a0, a1, 50)
    ax.plot(0.42 * np.cos(aa), 0.42 * np.sin(aa), color=C["accent"], lw=1.8)
    ax.annotate("", xy=(0.42 * np.cos(a1), 0.42 * np.sin(a1)),
                xytext=(0.42 * np.cos(a1 - 0.08 * np.sign(th)), 0.42 * np.sin(a1 - 0.08 * np.sign(th))),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.4, mutation_scale=10))
    ax.text(-0.3 if th > 0 else -0.08, 0.7, rf"$\theta_{{ij}} = {th:+.1f}$", color=C["accent"], fontsize=11, ha="right")
    ax.plot(0, 0, "o", ms=9, mfc="white", mec=C["main"], zorder=6)
    ax.plot(0, 0, "o", ms=2.5, color=C["main"], zorder=7)
    ax.text(0.08, -0.22, r"$\hat{e}_{ij}$", fontsize=12)
    ax.text(*(k2 + np.array([-0.15, -0.2])), r"$f$", fontsize=12)
    ax.text(*(l2 + np.array([0.0, -0.22 if th > 0 else 0.1])), r"$g$", fontsize=12)
    dgfig.schematic_axes(ax, (-1.4, 1.4), (-0.9, 1.25))
    ax.set_title(title, loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.05)
dgfig.save(fig, __file__)

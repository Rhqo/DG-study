"""Figure 31.2.4: mean curvature lives on edges (Section 31.2).

(a) Cross-section perpendicular to a convex edge e (schematic, exact geometry): two faces (black) meet at the edge
    point O at a bending angle theta_e = 50 degrees between their outward unit normals (vermillion). Offsetting each
    face by epsilon along its normal (blue dashed) leaves a gap that the offset surface fills with a circular arc of
    radius epsilon and angle theta_e (orange): it adds the area epsilon * theta_e * l_e along an edge of length l_e.
(b) The torus T_{R,r}, R = 2, r = 0.8, outward normals: vertex mean curvature from bending angles
    H_i = -(1 / (4 A_i)) sum_j theta_ij l_ij on the 12 x 24 (open circles) and 24 x 48 (dots) grids, and the exact
    H(u) = -(R + 2 r cos u) / (2 r (R + r cos u)) (black).
Self-check: the arc length equals epsilon * theta_e; the maximal error of H_i is below 0.02 and 0.005.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-2-4-dihedral.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8

fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.05], wspace=0.25))

# ---------------------------------------------------------------- (a)
beta = np.radians(25)            # each face slopes down by beta, so theta_e = 2 beta
theta = 2 * beta
eps = 0.45
O = np.array([0.0, 0.0])
d1, d2 = np.array([-np.cos(beta), -np.sin(beta)]), np.array([np.cos(beta), -np.sin(beta)])
n1, n2 = np.array([-np.sin(beta), np.cos(beta)]), np.array([np.sin(beta), np.cos(beta)])
assert abs(np.degrees(np.arccos(n1 @ n2)) - 50) < 1e-9
L = 1.6
for d in (d1, d2):
    ax.plot(*np.array([O, O + L * d]).T, color=C["main"], lw=2.2, zorder=3)
for d, n in ((d1, n1), (d2, n2)):
    ax.plot(*np.array([O + eps * n, O + eps * n + L * d]).T, color=C["tangent"], lw=1.6, ls=(0, (5, 3)), zorder=2)
    mid = O + 0.75 * L * d
    ax.add_patch(FancyArrowPatch(mid, mid + 0.42 * n, arrowstyle="-|>", mutation_scale=11, color=C["normal"], lw=1.6,
                                 zorder=4))
a1, a2 = np.degrees(np.arctan2(n1[1], n1[0])), np.degrees(np.arctan2(n2[1], n2[0]))
ax.add_patch(Arc(O, 2 * eps, 2 * eps, theta1=a2, theta2=a1, color=C["accent"], lw=2.6, zorder=4))
arc_len = eps * np.radians(a1 - a2)
assert abs(arc_len - eps * theta) < 1e-12
for n in (n1, n2):
    ax.plot(*np.array([O, O + eps * n]).T, color=C["aux"], lw=0.8, ls=":", zorder=1)
ax.add_patch(Arc(O, 0.36, 0.36, theta1=a2, theta2=a1, color=C["main"], lw=1.0, zorder=4))
ax.text(0.0, 0.25, r"$\theta_e$", ha="center", fontsize=12)
ax.text(0.0, eps + 0.06, r"arc: $\varepsilon\,\theta_e$", ha="center", fontsize=11, color=C["accent"])
# the offset distance epsilon, shown once on face g
foot = O + 0.6 * d2
ax.add_patch(FancyArrowPatch(foot, foot + eps * n2, arrowstyle="<|-|>", mutation_scale=8, color=C["tangent"], lw=1.0,
                             zorder=4))
ax.text(*(foot + 0.5 * eps * n2 + np.array([-0.07, 0.0])), r"$\varepsilon$", fontsize=12, color=C["tangent"],
        ha="right", va="center")
ax.text(-1.25, -0.82, r"face $f$", fontsize=11)
ax.text(0.85, -0.82, r"face $g$", fontsize=11)
tip1 = O + 0.75 * L * d1 + 0.42 * n1
tip2 = O + 0.75 * L * d2 + 0.42 * n2
ax.text(tip1[0] - 0.08, tip1[1] + 0.02, r"$\mathbf{N}_f$", fontsize=12, color=C["normal"], ha="right")
ax.text(tip2[0] + 0.06, tip2[1] - 0.05, r"$\mathbf{N}_g$", fontsize=12, color=C["normal"], ha="left")
ax.plot(*O, "o", color=C["main"], ms=4, zorder=5)
ax.text(0.04, -0.2, r"$O$", fontsize=11)
ax.set_xlim(-1.6, 1.75)
ax.set_ylim(-0.95, 0.75)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a) offsetting a convex edge", fontsize=11.5)

# ---------------------------------------------------------------- (b)
uu = np.linspace(0, 2 * np.pi, 400)
axb.plot(uu, -(R + 2 * r * np.cos(uu)) / (2 * r * (R + r * np.cos(uu))), color=C["main"], lw=1.6,
         label=r"exact $H(u)$", zorder=1)
for (nu, nv), sty, tol in (((12, 24), dict(mfc="none", mec=C["tangent"], ms=7, marker="o", ls="none"), 0.02),
                           ((24, 48), dict(color=C["accent"], ms=4, marker="o", ls="none"), 0.005)):
    V, F = dm.torus_mesh(R, r, nu, nv)
    u = np.repeat(2 * np.pi * np.arange(nu) / nu, nv)
    Hi = dm.mean_curvature_dihedral(V, F)
    Hex = -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))
    assert np.abs(Hi - Hex).max() < tol
    sel = np.arange(0, len(u), nv)
    axb.plot(u[sel], Hi[sel], label=rf"$H_i$ from $\theta_e$, ${nu}\times{nv}$", zorder=2, **sty)
axb.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
axb.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axb.set_xlabel(r"$u$ (angle around the tube; $u = 0$ outside)")
axb.set_ylabel(r"mean curvature (outward $\mathbf{N}$)")
axb.legend(fontsize=10.5, frameon=False, loc="lower center")
axb.set_title(r"(b) $H_i$ against $u$ on the torus", fontsize=11.5)
axb.set_ylim(-1.12, 0.0)

fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)

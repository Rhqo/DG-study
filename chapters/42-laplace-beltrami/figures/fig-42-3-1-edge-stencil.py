"""Figure 42.3.1: the stiffness entry of an edge collects one term from each adjacent triangle (Section 42.3,
Lemma 42.3.2, Theorem 42.3.4).

Shared edge x_i = (0, 0), x_j = (2, 0); upper triangle f = (i, j, k) with x_k = (0.9, 1.3) in both panels; lower
triangle g = (j, i, l) with (a) x_l = (1.1, -1.1) and (b) x_l = (1.0, -0.35) (an obtuse angle at x_l).
The gradients of the hat functions phi_i (blue) and phi_j (green) on f and on g are drawn to the right of the mesh, from a
common point for each face, scaled so that the longer arrow of the face has length 0.42 (directions are exact). Their area-weighted inner product A <grad phi_i, grad phi_j> = -(1/2) cot(opposite angle) is printed in the
triangle; the stiffness entry a_ij is the sum and equals -L_ij = -(cot alpha + cot beta)/2.
Self-check: the per-face values from dgdec.barycentric_gradients match -(1/2) cot theta, and a_ij = -L_ij with
L = dgmesh.cotan_laplacian of the two-triangle mesh.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-3-1-edge-stencil.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
xi, xj, xk = np.array([0.0, 0, 0]), np.array([2.0, 0, 0]), np.array([0.9, 1.3, 0])
cases = {"a": np.array([1.1, -1.1, 0]), "b": np.array([1.0, -0.35, 0])}


def angle_at(p, q, r):
    a, b = q - p, r - p
    return np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
for ax, (key, xl) in zip(axes, cases.items()):
    V = np.array([xi, xj, xk, xl])
    F = np.array([[0, 1, 2], [1, 0, 3]])
    L = dm.cotan_laplacian(V, F)
    vals = []
    for tri, opp in ((F[0], 2), (F[1], 3)):
        P = V[tri]
        G = dd.barycentric_gradients(P)
        A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        gi, gj = G[list(tri).index(0)], G[list(tri).index(1)]
        val = A * gi @ gj
        theta = angle_at(V[opp], xi, xj)
        assert abs(val + 0.5 / np.tan(theta)) < 1e-12
        vals.append((val, theta, P.mean(0), gi, gj))
    aij = vals[0][0] + vals[1][0]
    assert abs(aij + L[0, 1]) < 1e-12
    print(key, "angles", [round(float(np.degrees(v[1])), 1) for v in vals], "contributions",
          [round(float(v[0]), 4) for v in vals], "a_ij", round(float(aij), 4), "L_ij", round(float(L[0, 1]), 4))
    for tri, fc in ((F[0], "#E8E8E8"), (F[1], "#D6E9F5")):
        ax.fill(*V[tri][:, :2].T, color=fc, zorder=0)
    for a_, b_ in ((0, 2), (1, 2), (0, 3), (1, 3)):
        ax.plot(*V[[a_, b_]][:, :2].T, color="#888888", lw=0.9)
    ax.plot(*V[[0, 1]][:, :2].T, color=col["main"], lw=2.2)
    for (val, theta, c, gi, gj), opp, face in zip(vals, (2, 3), ("f", "g")):
        sc = 0.42 / max(np.linalg.norm(gi), np.linalg.norm(gj))       # same scale for the two arrows of a face
        anchor = np.array([2.75, 0.55 if opp == 2 else -0.75])
        ax.plot(*anchor, "o", color=col["main"], ms=2.5)
        for g, cc in ((gi, col["tangent"]), (gj, col["third"])):
            ax.annotate("", xy=anchor + sc * g[:2], xytext=anchor, arrowprops=dict(arrowstyle="->", color=cc, lw=1.4))
        ax.text(anchor[0] - 0.55, anchor[1] + 0.05, rf"${face}$:", fontsize=10)
        ax.text(anchor[0] - 0.3, anchor[1] - (0.42 if opp == 2 else -0.5), rf"${val:+.3f}$", fontsize=10)
        ty = V[opp, 1] - 0.3 if opp == 2 else (V[opp, 1] + 0.18 if key == "a" else V[opp, 1] - 0.24)
        ax.text(V[opp, 0], ty, rf"${np.degrees(theta):.0f}^\circ$", ha="center", fontsize=9.5, color=col["accent"])
    ax.text(0.85, 0.42, r"$f$", fontsize=11)
    ax.text(0.85, -0.32 if key == "a" else -0.2, r"$g$", fontsize=11)
    for p, lab, off in ((xi, r"$\mathbf{x}_i$", (-0.25, -0.05)), (xj, r"$\mathbf{x}_j$", (0.06, -0.05)),
                        (xk, r"$\mathbf{x}_k$", (0.08, 0.04)), (xl, r"$\mathbf{x}_l$", (0.1, -0.12))):
        ax.text(p[0] + off[0], p[1] + off[1], lab, fontsize=10)
    ax.set_title(rf"({key}) $a_{{ij}} = {aij:+.3f}$", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 3.3)
    ax.set_ylim(-1.35, 1.6)
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

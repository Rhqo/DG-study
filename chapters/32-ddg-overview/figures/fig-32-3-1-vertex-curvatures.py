"""Figure 32.3.1: one vertex gamma_i of a discrete curve and what the four discrete curvatures measure there.
(a) turning angle theta_i, vertex normal n_i, and the length gradient grad L = t_{i-1} - t_i = -kappa^B_i n_i
(Proposition 32.3.6); (b) the circle through gamma_{i-1}, gamma_i, gamma_{i+1} (Proposition 32.3.8).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-3-1-vertex-curvatures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402

C = dgfig.COLORS
J = ddg32.J
G = np.array([[-1.1, -0.45], [0.0, 0.0], [0.05, 1.0]])   # gamma_{i-1}, gamma_i, gamma_{i+1} (a left turn)
th = dd.turning_angles(G, False)[1]
kB = dd.curvature_length_variation(G, False)[1]
kD = dd.curvature_osculating_circle(G, False)[1]
t0 = (G[1] - G[0]) / np.linalg.norm(G[1] - G[0])
t1 = (G[2] - G[1]) / np.linalg.norm(G[2] - G[1])
n_i = dd.curve_vertex_normals(G, False)[1]
grad = dd.length_gradient(G, False)[1]
assert th > 0 and np.allclose(grad, t0 - t1) and np.allclose(grad, -kB * n_i)
c = ddg32.circumcenters(G)[1]          # treat the three points as a closed triangle; center at vertex 1
rad = np.linalg.norm(c - G[1])
assert abs(1 / rad - kD) < 1e-12 and np.allclose([np.linalg.norm(c - p) for p in G], rad)
lbar = 0.5 * (np.linalg.norm(G[1] - G[0]) + np.linalg.norm(G[2] - G[1]))
assert abs(np.dot(c - G[1], n_i) - lbar / kB) < 1e-12        # Proposition 32.3.8

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.2))


def arrow(a, p, v, col, lw=1.5):
    a.annotate("", xy=p + v, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=10))


for a, off_i in ((ax, (-0.12, -0.3)), (bx, (0.1, -0.12))):     # (a): gamma_i below, clear of the gradient arrow
    a.plot(G[:, 0], G[:, 1], color=C["main"], lw=dgfig.LW["main"])
    a.plot(G[:, 0], G[:, 1], "o", color=C["main"], ms=4, zorder=5)
    for k, lab, off in ((0, r"$\gamma_{i-1}$", (-0.05, -0.2)), (1, r"$\gamma_i$", off_i),
                        (2, r"$\gamma_{i+1}$", (0.08, 0.05))):
        a.text(G[k, 0] + off[0], G[k, 1] + off[1], lab, fontsize=12)
    a.set_aspect("equal")
    a.set_axis_off()

# (a) extension of edge i-1, turning angle, vertex normal, length gradient
ext = G[1] + 0.85 * t0
ax.plot([G[1, 0], ext[0]], [G[1, 1], ext[1]], color=C["aux"], lw=dgfig.LW["aux"], ls="--")
a0 = np.degrees(np.arctan2(t0[1], t0[0]))
a1 = np.degrees(np.arctan2(t1[1], t1[0]))
ax.add_patch(Arc(G[1], 1.0, 1.0, theta1=a0, theta2=a1, color=C["accent"], lw=1.6))
mid = np.radians(0.5 * (a0 + a1))
ax.text(G[1, 0] + 0.6 * np.cos(mid), G[1, 1] + 0.6 * np.sin(mid), r"$\theta_i$", color=C["accent"], fontsize=12)
arrow(ax, G[1], 0.7 * n_i, C["normal"])
ax.text(*(G[1] + 0.82 * n_i + np.array([-0.18, 0.0])), r"$\mathbf{n}_i$", color=C["normal"], fontsize=12)
arrow(ax, G[1], grad, C["third"])
ax.text(*(G[1] + grad + np.array([0.02, -0.18])), r"$\nabla_{\gamma_i}L$", color=C["third"], fontsize=12)
ax.set_xlim(-1.55, 1.25)
ax.set_ylim(-0.95, 1.3)
ax.set_title("(a)", fontsize=12)

# (b) circumcircle, its center, the chord w_i
bx.add_patch(Circle(c, rad, fc="none", ec=C["tangent"], lw=1.2))
bx.plot(*c, "o", color=C["tangent"], ms=4)
bx.text(c[0] - 0.22, c[1] + 0.02, r"$c_i$", color=C["tangent"], fontsize=12)
bx.plot([c[0], G[1, 0]], [c[1], G[1, 1]], color=C["tangent"], lw=dgfig.LW["aux"], ls=":")
mr = c + 0.12 * (G[1] - c)                       # near c, where the n_i arrow does not reach
bx.text(mr[0] - 0.02, mr[1] - 0.2, r"$r_i$", color=C["tangent"], fontsize=12)
bx.plot([G[0, 0], G[2, 0]], [G[0, 1], G[2, 1]], color=C["aux"], lw=dgfig.LW["aux"], ls="--")
mw = G[0] + 0.3 * (G[2] - G[0])
bx.text(mw[0] - 0.24, mw[1] + 0.04, r"$w_i$", color=C["aux"], fontsize=12)
arrow(bx, G[1], 0.7 * n_i, C["normal"])
bx.text(*(G[1] + 0.7 * n_i + np.array([-0.05, 0.08])), r"$\mathbf{n}_i$", color=C["normal"], fontsize=12)
bx.set_xlim(c[0] - rad - 0.15, c[0] + rad + 0.15)
bx.set_ylim(c[1] - rad - 0.15, c[1] + rad + 0.15)
bx.set_title("(b)", fontsize=12)

fig.tight_layout()
dgfig.save(fig, __file__)
print(f"theta_i = {th:.4f}, kappa^B = {kB:.4f}, kappa^D = {kD:.4f}, r_i = {rad:.4f}, lbar = {lbar:.4f}")

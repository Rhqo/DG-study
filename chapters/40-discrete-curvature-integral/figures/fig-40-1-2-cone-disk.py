"""Figure 40.1.2: at a cone vertex all the curvature of every small geodesic disk sits at the vertex.

(a) the star of Figure 40.1.1 (ddg40.example_star, Theta_i = 5.447, delta_i = 0.836) with the geodesic disk of radius
eps = 0.6 around x_i shaded: the union of the six circular sectors of radius eps in the faces (Proposition 40.1.10,
valid for eps <= h_i = 0.872, the distance from x_i to the link).
(b) the total Gaussian curvature inside the polar cap of radius eps (arc length along the meridian) of a circular cone
with the same cone angle Theta = Theta_i whose tip is rounded by a tangent sphere cap of radius a = 1.2, 0.6, 0.2
(Example 40.1.11, from Proposition 39.4.6: 2 pi (1 - rho'(eps))). The polyhedral vertex (black) has the value
2 pi - A''(eps) = delta_i for every eps in (0, h_i].

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-1-2-cone-disk.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402

C = dgfig.COLORS
V, F = ddg40.example_star()
i = 0
theta = dm.corner_angles(V, F)[:, 0]
Theta = theta.sum()
delta = 2 * np.pi - Theta
seq = ddg40.link_cycle(F, i)
h = min(ddg40.point_triangle_distance(V[[i]], V[a], V[b], V[b])[0] for a, b in zip(seq, seq[1:] + seq[:1]))
eps = 0.6
assert eps <= h and abs(h - 0.8720044985237847) < 1e-12

# sectors: in each face (i, j, k) the points x_i + eps (cos s u + sin s w), s in [0, theta], u = unit(x_j - x_i)
sectors, arcs = [], []
for f, (a, j, k) in enumerate(F):
    u = (V[j] - V[a]) / np.linalg.norm(V[j] - V[a])
    w = V[k] - V[a] - ((V[k] - V[a]) @ u) * u
    w /= np.linalg.norm(w)
    s = np.linspace(0, theta[f], 30)
    pts = V[a] + eps * (np.cos(s)[:, None] * u + np.sin(s)[:, None] * w)
    assert np.allclose(np.linalg.norm(pts - V[a], axis=1), eps)
    sectors.append(np.vstack([V[a], pts]))
    arcs.append(pts)
# area of the disk = (1/2) Theta eps^2 (sum of the sector areas computed as polygons)
area = sum(0.5 * np.linalg.norm(np.cross(P[1:-1] - P[0], P[2:] - P[0]), axis=1).sum() for P in sectors)
assert abs(area - 0.5 * Theta * eps ** 2) < 2e-3

fig = plt.figure(figsize=(7.0, 3.3))
ax = dgfig.axes3d(fig, 121, elev=34, azim=-62)
ax.set_position([0.0, 0.02, 0.5, 0.98])
col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.7, alpha=0.6, zorder=1))
ax.add_collection3d(Poly3DCollection([P + np.array([0, 0, 0.004]) for P in sectors], facecolors=C["region"], alpha=0.55,
                                     edgecolors="none", zorder=3))
for P in arcs:
    ax.plot(*(P + np.array([0, 0, 0.005])).T, color=C["tangent"], lw=1.5, zorder=5)
dgfig.point3d(ax, V[i], None, size=14)
ax.text(*(V[i] + np.array([-0.05, 0.08, 0.1])), r"$\mathbf{x}_i$", fontsize=12)
ax.text(*(arcs[2][20] + np.array([-0.2, 0.05, 0.02])), r"$D_\varepsilon$", color=C["tangent"], fontsize=12)
dgfig.equal_aspect(ax, V, zoom=1.4)
ax.set_title("(a)", loc="left", fontsize=12, y=0.93)

ax2 = fig.add_subplot(122)
e = np.linspace(0, 1.0, 801)
styles = [(1.2, C["third"], "-"), (0.6, C["accent"], "-"), (0.2, C["normal"], "-")]
for a, c, ls in styles:
    tot = ddg40.rounded_cone_curvature_in_disk(e, Theta, a)
    assert abs(tot[-1] - delta) < 1e-12 and np.all(np.diff(tot) >= -1e-12)
    ax2.plot(e, tot, color=c, lw=1.6, ls=ls, label=r"rounded, $a = %.1f$" % a)
ev = e[(e > 0) & (e <= h)]
ax2.plot(ev, np.full(len(ev), delta), color=C["main"], lw=1.8, label=r"vertex: $2\pi - A''(\varepsilon)$")
ax2.plot([0], [0], "o", color=C["main"], ms=3.5, mfc="white")
ax2.axhline(delta, color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax2.axvline(h, color=C["aux"], lw=0.7, ls=(0, (1, 2)))
ax2.text(h + 0.015, 0.45, r"$h_i$", color=C["aux"], fontsize=11)
ax2.set_xlabel(r"geodesic radius $\varepsilon$")
ax2.set_ylabel(r"$\int_{D_\varepsilon} K\,dA$")
ax2.set_yticks([0, 0.4, delta])
ax2.set_yticklabels([r"$0$", r"$0.4$", r"$\delta_i$"])
ax2.set_xlim(0, 1.0)
ax2.set_ylim(-0.03, 1.0)
ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
ax2.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.0, right=0.98, bottom=0.17, top=0.9, wspace=0.28)
dgfig.save(fig, __file__)
print(f"Theta {Theta:.4f}, delta {delta:.4f}, h {h:.4f}, disk area {area:.4f} vs {0.5 * Theta * eps ** 2:.4f}")

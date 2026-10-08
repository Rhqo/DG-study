"""Figure 31.3.3: the cotan Laplacian of the position is the mean curvature normal (Section 31.3).

(a) Unit icosphere, level 3 (642 vertices), mixed Voronoi mass. The vertices with |z| < 0.09 (a band around the
    equator) seen from above, with the vectors (M^{-1} L x)_i projected to the xy-plane and scaled by 0.15.
    Every vector points to the centre and has length 2 = |2 H N| (H = -1 for the outward normal).
(b) Histogram of the normal component <(M^{-1} L x)_i, N_i> (N_i = x_i, exact outward normal) on the level-4
    icosphere (2562 vertices) with the barycentric mass (blue) and the mixed Voronoi mass (orange: all values
    equal -2 to 1e-12), and on the jittered level-4 icosphere (vertices moved by 0.1 h and projected back, seed 1)
    with the mixed Voronoi mass (green; standard deviation 0.020, range [-2.52, -1.79]). The spread is not the
    pointwise noise of Figure 31.2.5: with the pure circumcentric Voronoi area the normal component is exactly -2 on
    the jittered mesh too, and the mixed area deviates from it only at the 439 vertices next to one of the 178
    obtuse triangles.
Self-check: Voronoi mass on the inscribed regular icosphere gives exactly -2; the tangential part is below 0.01;
on the jittered mesh the deviation is confined to vertices next to obtuse triangles.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-3-3-mean-curvature-normal.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw=dict(width_ratios=[0.9, 1.1], wspace=0.25))

# ---------------------------------------------------------------- (a)
V, F = dm.icosphere(3)
L = dm.cotan_laplacian(V, F)
m = dm.mass_voronoi(V, F)
Hn = (L @ V) / m[:, None]
assert np.allclose((Hn * V).sum(1), -2.0, atol=1e-10)
band = np.abs(V[:, 2]) < 0.09
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=C["main"], lw=1.2))
for p, h in zip(V[band], Hn[band]):
    ax.add_patch(FancyArrowPatch(p[:2], p[:2] + 0.15 * h[:2], arrowstyle="-|>", mutation_scale=7, color=C["normal"],
                                 lw=1.1, zorder=3))
ax.plot(V[band, 0], V[band, 1], "o", color=C["main"], ms=2.5, zorder=4)
ax.plot(0, 0, "+", color=C["aux"], ms=8)
ax.text(0.0, -1.32, r"arrows: $0.15\,(M^{-1}Lx)_i$, length $0.3 = 0.15 \cdot 2$", ha="center", fontsize=10.5)
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.45, 1.2)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a) equatorial band, top view", fontsize=11.5)

# ---------------------------------------------------------------- (b)
V4, F4 = dm.icosphere(4)
L4 = dm.cotan_laplacian(V4, F4)
nb = ((L4 @ V4) / dm.mass_barycentric(V4, F4)[:, None] * V4).sum(1)
Hv = (L4 @ V4) / dm.mass_voronoi(V4, F4)[:, None]
nv = (Hv * V4).sum(1)
tang = np.linalg.norm(Hv - nv[:, None] * V4, axis=1)
assert np.allclose(nv, -2.0, atol=1e-10) and tang.max() < 0.01
rng = np.random.default_rng(1)
for lv in range(1, 5):
    V0, F0 = dm.icosphere(lv)
    h = np.mean([np.linalg.norm(V0[a] - V0[b]) for a, b in dm.edges(F0)])
    J = V0 + 0.1 * h * rng.normal(size=V0.shape)
    J /= np.linalg.norm(J, axis=1, keepdims=True)
LJ = dm.cotan_laplacian(J, F0) @ J
nj = ((LJ / dm.mass_voronoi(J, F0)[:, None]) * J).sum(1)
# where does the spread come from?  Example 31.3.6 holds for any mesh inscribed in a sphere if A_i is the pure
# (circumcentric, possibly signed) Voronoi area; the mixed area differs from it only next to obtuse triangles.
angJ = dm.corner_angles(J, F0)
obt = np.zeros(len(J), bool)
for f in np.where(angJ.max(1) > np.pi / 2)[0]:
    obt[F0[f]] = True
AV = np.zeros(len(J))
for f, tri in enumerate(F0):
    for c in range(3):
        i, j, k = tri[c], tri[(c + 1) % 3], tri[(c + 2) % 3]
        AV[i] += (np.sum((J[i] - J[j]) ** 2) / np.tan(angJ[f, (c + 2) % 3])
                  + np.sum((J[i] - J[k]) ** 2) / np.tan(angJ[f, (c + 1) % 3])) / 8
assert np.allclose((LJ * J).sum(1) / AV, -2.0, atol=1e-9)        # pure Voronoi area: exactly -2 on the jittered mesh
assert np.allclose(nj[~obt], -2.0, atol=1e-9)                    # mixed area: exact away from obtuse triangles
assert (np.abs(nj[obt] + 2) > 1e-6).all() and obt.sum() == 439 and (angJ.max(1) > np.pi / 2).sum() == 178
bins = np.linspace(-2.6, -1.7, 91)
axb.hist(nb, bins=bins, color=C["tangent"], alpha=0.7, label="regular, barycentric $M$")
axb.hist(nj, bins=bins, color=C["third"], alpha=0.6, label="jittered, mixed $M$")
axb.hist(nv, bins=bins, color=C["accent"], alpha=0.9, label="regular, Voronoi $M$")
axb.axvline(-2.0, color=C["main"], lw=1.0, ls=(0, (4, 3)))
axb.text(-1.98, axb.get_ylim()[1] * 0.92, r"$2H = -2$", fontsize=10.5)
axb.set_yscale("log")
n5 = int((nb < -2.2).sum())
assert n5 == 12
axb.annotate("12 vertices of\nvalence 5", xy=(nb.min(), 12), xytext=(-2.58, 60), fontsize=10.5, color=C["tangent"],
             arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=0.9))
axb.set_xlabel(r"normal component $\langle (M^{-1}Lx)_i, \mathbf{N}_i \rangle$")
axb.set_ylabel("number of vertices")
axb.legend(fontsize=10.5, frameon=False, loc="upper left")
axb.set_title("(b) level-4 icospheres (2562 vertices)", fontsize=11.5)
assert round(nb.min(), 2) == -2.29 and round(nb.max(), 2) == -1.99
assert round(nj.std(), 3) == 0.020 and round(nj.min(), 2) == -2.52 and round(nj.max(), 2) == -1.79

fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)

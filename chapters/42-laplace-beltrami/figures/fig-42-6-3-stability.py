"""Figure 42.6.3: explicit and semi-implicit mean curvature flow of a noisy sphere (Section 42.6, Proposition 42.6.4,
Example 42.6.8).

Mesh: dgmesh.icosphere(3) with every vertex scaled by 1 + 0.05 N(0, 1) (seed 5). Steps with L and M (mixed Voronoi) of
the current mesh: explicit x' = x + h M^{-1} L x for h = 0.5, 0.9, 1.5 h_c, h_c = 2 / lambda_max(M^{-1}(-L)) of the
initial mesh (= 0.00656), and semi-implicit (M - h L) x' = M x for h = 0.002 and 0.02, all up to t = 0.1.
Plotted: roughness = std / mean of the distances of the vertices from their centroid (0 for a round sphere), log scale.
Self-check: the semi-implicit roughness decreases at every step and the area never increases; the explicit run with
1.5 h_c ends rougher than it started; h_c is about 0.24 times the squared mean edge length.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-6-3-stability.py``
"""

import warnings

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
col = dgfig.COLORS
rng = np.random.default_rng(5)
V0, F = dm.icosphere(3)
Vn = V0 * (1 + 0.05 * rng.normal(size=(len(V0), 1)))
L, m = dm.cotan_laplacian(Vn, F), dm.mass_voronoi(Vn, F)
lam = np.linalg.eigvalsh(-(L / np.sqrt(m)[:, None] / np.sqrt(m)[None, :]))
hc = 2 / lam[-1]
ell = dd.primal_edge_lengths(Vn, F).mean()
assert abs(hc / ell ** 2 - 0.24) < 0.01


def rough(V):
    r = np.linalg.norm(V - V.mean(0), axis=1)
    return r.std() / r.mean()


T = 0.1
runs = [("explicit", 0.5 * hc, col["third"], "--"), ("explicit", 0.9 * hc, col["accent"], "--"),
        ("explicit", 1.5 * hc, col["normal"], "--"), ("implicit", 0.002, col["tangent"], "-"),
        ("implicit", 0.02, col["main"], "-")]
res = []
for kind, h, c, ls in runs:
    V = Vn.copy()
    n = int(T / h)
    R, A = [rough(V)], [dm.face_areas(V, F).sum()]
    for k in range(n):
        L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
        V = V + h * (L @ V) / m[:, None] if kind == "explicit" else np.linalg.solve(np.diag(m) - h * L, m[:, None] * V)
        R.append(rough(V))
        A.append(dm.face_areas(V, F).sum())
    R, A = np.array(R), np.array(A)
    if kind == "implicit":
        assert np.all(np.diff(R) < 0) and np.all(np.diff(A) <= 1e-12)
    res.append((kind, h, c, ls, h * np.arange(len(R)), R))
    print(f"{kind:8s} h = {h:.5f}: roughness {R[0]:.4f} -> min {R.min():.4f} -> end {R[-1]:.4f} at t = {h * (len(R) - 1):.3f}")
assert res[2][5][-1] > res[2][5][0]
print(f"h_c = {hc:.5f}, mean edge length {ell:.4f}, h_c / l^2 = {hc / ell ** 2:.3f}")

fig, ax = plt.subplots(figsize=(6.6, 3.2))
for kind, h, c, ls, t, R in res:
    lab = (rf"explicit, $h = {h / hc:.1f}\,h_c$" if kind == "explicit" else rf"semi-implicit, $h = {h}$")
    ax.semilogy(t, R, ls, color=c, lw=1.4, marker="o" if kind == "implicit" and h > 0.01 else None, ms=4, label=lab)
ax.set_xlabel(r"$t$")
ax.set_ylabel("roughness")
ax.set_ylim(0.0045, 0.08)
ax.set_yticks([0.005, 0.01, 0.02, 0.05])
ax.set_yticklabels(["0.005", "0.01", "0.02", "0.05"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlim(0, 0.1)
ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.11, right=0.66, top=0.97, bottom=0.15)
dgfig.save(fig, __file__)

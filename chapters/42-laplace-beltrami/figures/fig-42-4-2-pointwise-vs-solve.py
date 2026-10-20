"""Figure 42.4.2: pointwise Laplacians may fail to converge while Poisson solutions converge (Section 42.4,
Example 42.4.7).

Meshes: ch42.jittered_sphere(level, s=0.15, seed=42), levels 1-4 (unit sphere, tangential noise, obtuse triangles).
Dual areas A_i: barycentric (orange), circumcentric (blue), mixed Voronoi (green), M = diag(A_i).
(a) pointwise error max_i |(M^{-1} L z)_i + 2 z_i| (Delta z = -2z).
(b) the Poisson solution of L u = M f with f = -2z (mean removed, M-weighted mean 0): area-weighted RMS error of u - z.
Horizontal axis: mean edge length (log-log). Self-check: in (b) all three errors decrease by a factor > 3 per level
from level 2 on; in (a) the barycentric and mixed errors do not decrease (level 4 >= level 1).
The interactive version shows the level-3 mesh colored by the pointwise error of the mixed area.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-4-2-pointwise-vs-solve.py``
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch42  # noqa: E402

col = dgfig.COLORS
kinds = {"barycentric": col["accent"], "circumcentric": col["tangent"], "mixed": col["third"]}
hs, pw, so = [], {k: [] for k in kinds}, {k: [] for k in kinds}
for lv in range(1, 5):
    V, F = ch42.jittered_sphere(lv, s=0.15, seed=42)
    L = dm.cotan_laplacian(V, F)
    z = V[:, 2]
    hs.append(dd.primal_edge_lengths(V, F).mean())
    for k in kinds:
        m = ch42.mass(V, F, k)
        pw[k].append(np.abs(L @ z / m + 2 * z).max())
        b = m * (-2 * z)
        b -= m * (b.sum() / m.sum())
        u = np.linalg.solve(-L + np.outer(m, m), -b)
        e = u - (z - (m @ z) / m.sum())
        so[k].append(np.sqrt((e ** 2 * m).sum() / m.sum()))
for k in kinds:
    r = np.array(so[k][1:-1]) / np.array(so[k][2:])
    assert np.all(r > 3), (k, r)
    print(f"{k:13s} pointwise {[f'{v:.3f}' for v in pw[k]]}  solve {[f'{v:.2e}' for v in so[k]]}")
assert pw["barycentric"][-1] >= pw["barycentric"][0] and pw["mixed"][-1] >= pw["mixed"][0]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0))
for k, c in kinds.items():
    axa.loglog(hs, pw[k], "o-", color=c, ms=3.5, lw=1.4, label=k)
    axb.loglog(hs, so[k], "o-", color=c, ms=3.5, lw=1.4, label=k)
axb.loglog(hs, 0.6 * so["circumcentric"][-1] * (np.array(hs) / hs[-1]) ** 2, ":", color=col["aux"], lw=1.0)
axb.text(hs[1], 0.6 * so["circumcentric"][-1] * (hs[1] / hs[-1]) ** 2 * 0.35, "slope 2", fontsize=9, color=col["aux"])
for ax, title, yl in ((axa, r"(a) pointwise $\max_i|(M^{-1}Lz)_i + 2z_i|$", "error"),
                      (axb, r"(b) Poisson solution, RMS error", "error")):
    ax.set_xlabel(r"mean edge length $\bar\ell$")
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([0.08, 0.15, 0.3, 0.6], ["0.08", "0.15", "0.3", "0.6"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
axa.set_ylim(0.03, 1.5)
axa.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.17, wspace=0.3)
dgfig.save(fig, __file__)

V, F = ch42.jittered_sphere(3, s=0.15, seed=42)
L = dm.cotan_laplacian(V, F)
err = np.abs(L @ V[:, 2] / ch42.mass(V, F, "mixed") + 2 * V[:, 2])
mesh = dm.plotly_mesh(V, F, intensity=err, colorscale="Viridis", cmin=0, cmax=float(err.max()), name="error",
                      showscale=True, colorbar_title="|Delta_h z + 2z|",
                      hovertemplate="pointwise error %{intensity:.3f}<extra></extra>")
dm.plotly_save([mesh, dm.plotly_edges(V, F, color="#444444", width=1.0)], __file__,
               "jittered sphere (level 3): pointwise error of the mixed-area Laplacian of z", "fig-42-4-2")

"""Figure 41.3.1: discrete principal curvatures kappa_{1,2} = H_i +- sqrt(H_i^2 - K_i) (Definition 41.3.3).

(a) torus_mesh(2, 0.8, 24, 48) (faces outward): K_i = delta_i / A_i, H_i = <(L x)_i, N_i> / (2 A_i) (dgdec.mean_curvature,
    mixed area, area-weighted N_i), plotted against u for every vertex; the curves are the exact kappa_1 = -cos u / (R + r
    cos u) and kappa_2 = -1/r of T_{2,0.8} with the outward normal.
(b) unit icosphere(level), level 1-4: every vertex has H_i^2 < K_i (umbilic sphere: H^2 - K = 0). Plotted against the
    level: max |K_i - 1| and the would-be split 2 sqrt(max |H_i^2 - K_i|) that the square root would produce without the
    clamp (log scale): the first falls by 1/4 per level, the second only by 1/2.
Also writes an interactive version: the torus colored by the discrete kappa_1.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-3-1-principal-curvatures.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8
V, F = dm.torus_mesh(R, r, 24, 48)
A = dm.mass_voronoi(V, F)
K, H = dm.angle_defect(V, F) / A, dd.mean_curvature(V, F)
k1, k2, cl = ddg41.principal_curvatures(H, K)
u, Ke, He, e1, e2 = ddg41.torus_exact(V, R, r)
assert cl.sum() == 0 and np.abs(k1 - e1).max() < 0.007 and np.abs(k2 - e2).max() < 0.006
lev_err, lev_split = [], []
for lev in (1, 2, 3, 4):
    Vs, Fs = dm.icosphere(lev)
    Ks = dm.angle_defect(Vs, Fs) / dm.mass_voronoi(Vs, Fs)
    Hs = dd.mean_curvature(Vs, Fs)
    disc = Hs ** 2 - Ks
    assert np.all(disc < 0)
    lev_err.append(np.abs(Ks - 1).max())
    lev_split.append(2 * np.sqrt(np.abs(disc).max()))
assert 3.5 < lev_err[2] / lev_err[3] < 4.5 and 1.8 < lev_split[2] / lev_split[3] < 2.2

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
ax = axes[0]
uu = np.linspace(-np.pi, np.pi, 400)
ax.plot(uu, -np.cos(uu) / (R + r * np.cos(uu)), color=C["main"], lw=1.2)
ax.plot(uu, np.full_like(uu, -1 / r), color=C["main"], lw=1.2)
ax.plot(u, k1, "o", color=C["tangent"], ms=2.5, label=r"$\kappa_{1,i}$")
ax.plot(u, k2, "s", color=C["normal"], ms=2.5, label=r"$\kappa_{2,i}$")
ax.set_xlabel(r"$u$")
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.legend(fontsize=10, frameon=False, loc="center right")
ax.set_title("(a)", loc="left", fontsize=12)
ax = axes[1]
levels = [1, 2, 3, 4]
ax.semilogy(levels, lev_err, "o-", color=C["tangent"], lw=1.4, ms=4, label=r"$\max|K_i - 1|$")
ax.semilogy(levels, lev_split, "s--", color=C["normal"], lw=1.4, ms=4, label=r"$2\sqrt{\max|H_i^2 - K_i|}$")
ax.set_xticks(levels)
ax.set_xlabel("icosphere level")
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.9, wspace=0.28)
dgfig.save(fig, __file__)

tr = [dm.plotly_mesh(V, F, intensity=k1, colorscale="RdBu_r", cmin=-0.9, cmax=0.9, showscale=True,
                     name="discrete kappa_1", colorbar_title="kappa_1"), dm.plotly_edges(V, F, width=1)]
dm.plotly_save(tr, __file__, "torus_mesh(2, 0.8, 24, 48) colored by the discrete principal curvature kappa_1", "fig-41-3-1")
print("torus errors", np.abs(k1 - e1).max(), np.abs(k2 - e2).max(), "ico", np.round(lev_err, 5), np.round(lev_split, 4))

"""Figure 31.4.1: the spectrum of the cotan Laplacian on sphere meshes (Section 31.4).

Unit icospheres of level 2 (162 vertices) and level 3 (642 vertices), mixed Voronoi mass. The generalized
eigenvalues of -L phi = lambda M phi, lambda_0 <= lambda_1 <= ..., for k = 0..63 (degrees l = 0..7).
(a) lambda_k against k (dots) and the exact eigenvalues l (l + 1) of the unit sphere with multiplicity 2l + 1
    (black staircase, Proposition 29.6.1).
(b) Relative error of the mean of each group of 2l + 1 eigenvalues against l (log scale).
Self-check: lambda_1..3 = 2 (relative 5e-6 on level 2, 4e-7 on level 3), groups of 2l + 1 values; level-3 error at l = 7 is 7.4%, level 2 26.8%.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-4-1-spectrum.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw=dict(width_ratios=[1.3, 1.0], wspace=0.3))
K = 64
ls = np.arange(8)
res = {}
for lv, col, mk, ms in ((2, C["tangent"], "s", 3.5), (3, C["accent"], "o", 3.2)):
    V, F = dm.icosphere(lv)
    lam, _ = dm.generalized_eigs(dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F), K)
    assert abs(lam[0]) < 1e-9 and np.allclose(lam[1:4], 2.0, rtol=0, atol=2e-5)
    grp = np.array([lam[l * l:(l + 1) ** 2].mean() for l in ls[1:]])
    res[lv] = (grp - ls[1:] * (ls[1:] + 1)) / (ls[1:] * (ls[1:] + 1))
    ax.plot(np.arange(K), lam, mk, color=col, ms=ms, label=f"level {lv} ({len(V)} vertices)", zorder=3)
# exact staircase
kk, vals = [], []
for l in ls:
    kk += [l * l - 0.5, (l + 1) ** 2 - 0.5]
    vals += [l * (l + 1)] * 2
ax.plot(kk, vals, color=C["main"], lw=1.4, label=r"sphere: $\ell(\ell+1)$, $2\ell+1$ times", zorder=2)
for l in ls[1:6]:
    ax.text((l + 1) ** 2 - 0.9, l * (l + 1) + 2.2, rf"$\ell={l}$", ha="right", fontsize=9.5)
ax.set_xlabel(r"index $k$")
ax.set_ylabel(r"eigenvalue $\lambda_k$")
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.set_xlim(-1, K)
ax.set_ylim(-2, 60)
ax.set_title("(a) eigenvalues come in groups", fontsize=11.5)

for lv, col, mk in ((2, C["tangent"], "s"), (3, C["accent"], "o")):
    axb.semilogy(ls[1:], np.abs(res[lv]), mk + "-", color=col, ms=5, lw=1.4, label=f"level {lv}")
assert round(abs(res[3][-1]), 3) == 0.074 and round(abs(res[2][-1]), 3) == 0.268
assert (res[3] <= 1e-12).all() and (res[2] <= 1e-12).all()   # the mesh underestimates
axb.set_xlabel(r"degree $\ell$")
axb.set_ylabel("relative error of the group mean")
axb.legend(fontsize=9.5, frameon=False, loc="lower right")
axb.grid(True, which="major", lw=0.3, alpha=0.5)
axb.set_title(r"(b) error grows with $\ell$", fontsize=11.5)

fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)

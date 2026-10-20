"""Figure 43.5.2: the smallest eigenvalues of E_C z = lambda M z on the hemisphere and their eigenvectors
(Example 43.5.11, Exercise 43.5.4).

Meshes ch43.hemisphere(level), level = 2, 3, 4 (89, 337, 1313 vertices). ch43.pencil_eigs gives the eigenpairs of the
pencil on the M-orthogonal complement of the constants (dense, through K = EC + m m^T).
(a) lambda_1..lambda_4 against the mean edge length (log-log); the grey dashed line has slope 2.
(b) level 3: the unwrapped argument of z_k (k = 1, 2, 3, minus its M-mean) along the boundary loop, against the
    parameter s = (index along the loop) / (number of boundary vertices); slopes 1, 2, 3 mean that z_k(boundary)
    winds k times around the origin, as conj(sigma)^k does.
Self-check: winding numbers 1, 2, 3 at every level; lambda_1..3 decrease by a factor in (2.9, 4.5) per level
(lambda_4 by more than 2: at level 2 it is not yet resolved);
relative distance of z_k to the best fit of conj(sigma)^k + c at level 4 below 0.04.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-5-2-modes.py``
"""

import pathlib
import sys

import dgfig  # noqa: E402
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch43  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
data = {}
for level in (2, 3, 4):
    V, F = ch43.hemisphere(level)
    m = dm.mass_barycentric(V, F)
    lam, Z = ch43.pencil_eigs(ch43.conformal_energy_matrix(V, F), m, 4)
    loop = dm.boundary_loop(F)
    E = dm.edges(F)
    hbar = np.mean(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1))
    wind, fit, args = [], [], []
    for k in range(3):
        zb = Z[loop, k] - (m @ Z[:, k]) / m.sum()
        ang = np.unwrap(np.angle(np.r_[zb, zb[:1]]))
        wind.append(int(round((ang[-1] - ang[0]) / (2 * np.pi))))
        fit.append(ch43.best_fit(Z[:, k], ch43.stereo_conj(V) ** (k + 1), m)[1])
        args.append(ang - ang[0])
    data[level] = dict(lam=lam, hbar=hbar, wind=wind, fit=fit, args=args, nb=len(loop), n0=len(V))
    assert wind == [1, 2, 3]
for lv in (2, 3):
    ratio = data[lv]["lam"] / data[lv + 1]["lam"]
    assert np.all((ratio[:3] > 2.9) & (ratio[:3] < 4.5)) and np.all(ratio > 2), ratio
    print("ratio level", lv, np.round(ratio, 2))
assert max(data[4]["fit"]) < 0.04
for lv, d in data.items():
    print(lv, d["n0"], "hbar %.4f" % d["hbar"], "lambda", np.round(d["lam"], 6), "fit", np.round(d["fit"], 4))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
ax = axes[0]
hs = np.array([data[lv]["hbar"] for lv in (2, 3, 4)])
for k, c in zip(range(4), (col["tangent"], col["third"], col["accent"], col["covector"])):
    ax.loglog(hs, [data[lv]["lam"][k] for lv in (2, 3, 4)], "o-", color=c, lw=1.4, ms=4, label=r"$\lambda_%d$" % (k + 1))
ax.loglog(hs, 2.2 * data[4]["lam"][0] * (hs / hs[-1]) ** 2, "--", color=col["aux"], lw=0.9, label="slope 2")
ax.set_xlabel(r"mean edge length $\bar\ell$")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=3, handlelength=1.4, columnspacing=0.8)
ax.set_ylim(3e-4, 20)
ax.set_title("(a) smallest eigenvalues", fontsize=11)
ax = axes[1]
d = data[3]
s = np.arange(d["nb"] + 1) / d["nb"]
for k, c in zip(range(3), (col["tangent"], col["third"], col["accent"])):
    ax.plot(s, d["args"][k] / (2 * np.pi), color=c, lw=1.5, label=r"$z_%d$" % (k + 1))
ax.set_xlabel(r"boundary parameter $s$")
ax.set_ylabel(r"$\arg z_k\ /\ 2\pi$")
ax.set_yticks([0, 1, 2, 3])
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(b) winding along the boundary", fontsize=11)
fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17, wspace=0.32)
dgfig.save(fig, __file__)

"""Figure 33.4.3: mean valence of closed simplicial surfaces (Section 33.4).

chi = 2: the tetrahedron, the octahedron, the triangulated cube (lib/ch33.py) and dgmesh.icosphere(level),
level = 0..5; chi = 0: dgmesh.torus_mesh(2, 0.8, nu, nv) for (nu, nv) = (3, 3), (4, 6), (8, 16), (16, 32),
(32, 64), (64, 128); chi = -2: dgmesh.genus2_mesh(h) for h = 0.2, 0.16, 0.12, 0.1.
(a) mean valence 2 n1 / n0 against n0 (log scale) with the curves 6 - 6 chi / n0;
(b) n0 (6 - mean valence) = sum_i (6 - deg i) against n0, which is exactly 6 chi (Proposition 33.4.23).
Self-check: every mesh is a closed simplicial surface (dgmesh.is_manifold, no boundary edge), 3 n2 = 2 n1, and
sum_i (6 - deg i) = 6 chi exactly (integers).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-4-3-mean-valence.py``
"""

import pathlib
import sys

import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
groups = {
    2: [ch33.tetrahedron()[1], ch33.octahedron()[1], ch33.cube_triangulated()[1]]
       + [dm.icosphere(lv)[1] for lv in range(6)],
    0: [dm.torus_mesh(2, 0.8, a, b)[1] for a, b in ((3, 3), (4, 6), (8, 16), (16, 32), (32, 64), (64, 128))],
    -2: [dm.genus2_mesh(h=h)[1] for h in (0.2, 0.16, 0.12, 0.1)],
}
style = {2: (C["tangent"], "o", r"$\chi = 2$"), 0: (C["third"], "s", r"$\chi = 0$"),
         -2: (C["normal"], "^", r"$\chi = -2$")}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.2))
xs = np.logspace(np.log10(3.5), np.log10(3e4), 200)
for chi, Fs in groups.items():
    col, mk, lab = style[chi]
    n0s, means, sums = [], [], []
    for F in Fs:
        n0, n1, n2 = dm.counts(F)
        assert n0 - n1 + n2 == chi and dm.is_manifold(F) and not dm.boundary_edges(F) and 3 * n2 == 2 * n1
        deg = np.bincount(dm.edges(F).ravel())
        assert deg.sum() == 2 * n1 and int(np.sum(6 - deg)) == 6 * chi
        n0s.append(n0)
        means.append(2 * n1 / n0)
        sums.append(int(np.sum(6 - deg)))
    axa.plot(xs, 6 - 6 * chi / xs, color=col, lw=1.0, alpha=0.7, zorder=1)
    axa.plot(n0s, means, mk, color=col, ms=5.5, label=lab, zorder=3)
    axb.axhline(6 * chi, color=col, lw=1.0, alpha=0.7, zorder=1)
    axb.plot(n0s, sums, mk, color=col, ms=5.5, label=lab, zorder=3)
axa.axhline(6, color=C["aux"], lw=0.8, ls=(0, (3, 2.5)), zorder=0)
axa.set_xscale("log")
axa.set_xlabel(r"$n_0$")
axa.set_ylabel("mean valence")
axa.set_ylim(2.7, 6.3)
axa.text(60, 4.6, r"$6 - 6\chi/n_0$", fontsize=11, color=C["tangent"])
axa.set_title("(a) mean valence", fontsize=12)
axa.legend(loc="lower right", frameon=False, fontsize=10)
axb.set_xscale("log")
axb.set_xlabel(r"$n_0$")
axb.set_ylabel(r"$\Sigma_i\,(6 - \deg i)$")
axb.set_ylim(-17, 17)
axb.set_yticks([-12, 0, 12])
axb.set_title(r"(b) $\Sigma_i\,(6 - \deg i) = 6\chi$", fontsize=12)
for ax in (axa, axb):
    ax.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.16, wspace=0.32)
dgfig.save(fig, __file__)

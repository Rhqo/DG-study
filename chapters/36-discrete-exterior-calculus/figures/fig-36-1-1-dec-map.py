"""Figure 36.1.1: the map of discrete exterior calculus on a triangle mesh (schematic, Section 36.1).

Three rows: smooth forms on an open set U containing the mesh, primal discrete forms (one number per vertex, edge,
face) and dual discrete forms (one number per dual cell). Horizontal arrows are exterior derivatives (d on smooth
forms, d0 and d1 on primal forms, the dual d~0 = d1^T and d~1 = -d0^T of Section 36.5), vertical arrows the de Rham
map R (down) and the Whitney map W (up), and the diagonal Hodge stars star0, star1, star2. The small gray labels say
in which section each arrow is defined.
Self-check: on icosphere(1) the matrix shapes are (n1 x n0) and (n2 x n1) and both rows are complexes
(d1 d0 = 0 and d~1 d~0 = 0).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-1-1-dec-map.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

V, F = dm.icosphere(1)
n0, n1, n2 = dm.counts(F)
D0, D1 = dd.d0(F, n0), dd.d1(F)
assert D0.shape == (n1, n0) and D1.shape == (n2, n1)
assert not np.any(D1 @ D0) and not np.any(ch36.dual_d1(F, n0) @ ch36.dual_d0(F))

C = dgfig.COLORS
GRAY, BLUE, ORANGE = "#555555", C["tangent"], C["accent"]
fig, ax = plt.subplots(figsize=(6.6, 4.2))
xs = [0.0, 2.25, 4.5]
ys = {"smooth": 2.3, "primal": 1.15, "dual": 0.0}
W_, H_ = 1.34, 0.56
styles = {"smooth": ("#EFEFEF", "#8C8C8C"), "primal": ("#DCEBF6", BLUE), "dual": ("#FCEBCB", ORANGE)}
labels = {
    "smooth": [r"$\Omega^0(U)$", r"$\Omega^1(U)$", r"$\Omega^2(U)$"],
    "primal": [r"$\mathbb{R}^{n_0}$", r"$\mathbb{R}^{n_1}$", r"$\mathbb{R}^{n_2}$"],
    "dual": [r"$\mathbb{R}^{n_0}$", r"$\mathbb{R}^{n_1}$", r"$\mathbb{R}^{n_2}$"],
}
subs = {
    "smooth": ["functions", "1-forms", "2-forms"],
    "primal": ["vertices", "edges", "faces"],
    "dual": ["dual faces", "dual edges", "dual vertices"],
}


def box(x, y, text, sub, face, edge):
    ax.add_patch(FancyBboxPatch((x - W_ / 2, y - H_ / 2), W_, H_, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, lw=1.2, zorder=2))
    ax.text(x, y + 0.08, text, ha="center", va="center", fontsize=13, zorder=3)
    ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=9, color="#333333", zorder=3)


def arrow(p, q, color, text=None, toff=(0, 0.12), fs=12, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        m = 0.5 * (np.array(p) + np.array(q)) + np.array(toff)
        ax.text(*m, text, ha="center", va="center", fontsize=fs, color=color, zorder=4)


for row in ("smooth", "primal", "dual"):
    for k, x in enumerate(xs):
        box(x, ys[row], labels[row][k], subs[row][k], *styles[row])

# exterior derivatives
for k in range(2):
    a, b = xs[k] + W_ / 2 + 0.05, xs[k + 1] - W_ / 2 - 0.05
    arrow((a, ys["smooth"]), (b, ys["smooth"]), "black", r"$d$")
    arrow((a, ys["primal"]), (b, ys["primal"]), "black", [r"$d_0$", r"$d_1$"][k])
    arrow((b, ys["dual"]), (a, ys["dual"]), ORANGE, [r"$\tilde d_1$", r"$\tilde d_0$"][k])
# de Rham map down, Whitney map up
for k, x in enumerate(xs):
    top, bot = ys["smooth"] - H_ / 2 - 0.04, ys["primal"] + H_ / 2 + 0.04
    arrow((x - 0.18, top), (x - 0.18, bot), BLUE, r"$R$", toff=(-0.17, 0))
    arrow((x + 0.18, bot), (x + 0.18, top), BLUE, r"$W$", toff=(0.19, 0), ls=(0, (4, 2)))
    top, bot = ys["primal"] - H_ / 2 - 0.04, ys["dual"] + H_ / 2 + 0.04
    arrow((x, top), (x, bot), ORANGE, [r"$\star_0$", r"$\star_1$", r"$\star_2$"][k], toff=(0.2, 0))
# section labels
ax.text(5.42, 0.5 * (ys["smooth"] + ys["primal"]), r"36.1  $R$" + "\n" + r"36.3  $W$", fontsize=9, color=GRAY,
        va="center")
ax.text(5.42, ys["primal"], "36.2, 36.4\n" + r"$d_0, d_1$", fontsize=9, color=GRAY, va="center")
ax.text(5.42, 0.5 * (ys["primal"] + ys["dual"]), "36.5  " + r"$\star_k$", fontsize=9, color=GRAY, va="center")
ax.text(5.42, ys["dual"], "36.5  " + r"$\tilde d_k$", fontsize=9, color=GRAY, va="center")
for row, y in ys.items():
    ax.text(-0.85, y, row, ha="right", va="center", fontsize=11, color=GRAY)
ax.text(2.25, -0.58, r"36.6: compose, e.g. $\Delta \approx \star_0^{-1}\,\tilde d_1\,\star_1\,d_0$",
        ha="center", va="center", fontsize=11, color="black")
ax.set_xlim(-1.55, 6.3)
ax.set_ylim(-0.8, 2.7)
ax.set_aspect("equal")
ax.set_axis_off()
dgfig.save(fig, __file__)

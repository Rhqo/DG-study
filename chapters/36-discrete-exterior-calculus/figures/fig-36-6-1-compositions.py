"""Figure 36.6.1: vector calculus as paths in the DEC diagram (schematic, Section 36.6).

Top row: primal discrete forms R^{n0} --d0--> R^{n1} --d1--> R^{n2}; bottom row: dual forms (dual 2-forms under
primal 0-forms, and so on), with the dual derivatives d~0 = d1^T and d~1 = -d0^T (Proposition 36.5.5) running from
right to left; vertical arrows: the diagonal Hodge stars down and their inverses up.
(a) the Laplacian path d0, star1, d~1, star0^{-1} (blue loop): Delta ~ star0^{-1} d~1 star1 d0 = star0^{-1} L.
(b) the divergence path star1, d~1, star0^{-1} (orange) and the curl path d1, star2 (green).
Self-check: on a jittered equilateral patch, star0^{-1} d~1 star1 d0 equals star0^{-1} L with L = -d0^T star1 d0 =
dgmesh.cotan_laplacian, and star2 d1 d0 = 0 (curl grad = 0).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-6-1-compositions.py``
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

rng = np.random.default_rng(366)
V, F = ch36.equilateral_patch(2)
V = V + 0.05 * rng.uniform(-1, 1, size=V.shape)
D0, D1 = dd.d0(F, len(V)), dd.d1(F)
s0, s1, s2 = dd.dual_areas(V, F, "circumcentric"), dd.hodge1(V, F), dd.hodge2(V, F)
dt1 = ch36.dual_d1(F, len(V))
lap = (dt1 @ (s1[:, None] * D0)) / s0[:, None]
L = dm.cotan_laplacian(np.c_[V, np.zeros(len(V))], F)
assert np.allclose(lap, L / s0[:, None]) and np.allclose(-D0.T @ (s1[:, None] * D0), L)
assert not np.any(s2[:, None] * (D1 @ D0))

col = dgfig.COLORS
BLUE, ORANGE, GREEN, GRAY = col["tangent"], col["accent"], col["third"], "#666666"
W_, H_ = 1.15, 0.52
xs = [0.0, 1.9, 3.8]
ytop, ybot = 1.25, 0.0


def draw_base(ax):
    for k, x in enumerate(xs):
        for y, face, edge, lab, sub in ((ytop, "#DCEBF6", BLUE, rf"$\mathbb{{R}}^{{n_{k}}}$", ["vertices", "edges", "faces"][k]),
                                        (ybot, "#FCEBCB", ORANGE, rf"$\mathbb{{R}}^{{n_{k}}}$",
                                         ["dual cells", "dual edges", "dual vertices"][k])):
            ax.add_patch(FancyBboxPatch((x - W_ / 2, y - H_ / 2), W_, H_, boxstyle="round,pad=0.02,rounding_size=0.07",
                                        facecolor=face, edgecolor=edge, lw=1.1, zorder=2))
            ax.text(x, y + 0.07, lab, ha="center", va="center", fontsize=12, zorder=3)
            ax.text(x, y - 0.15, sub, ha="center", va="center", fontsize=8.5, color="#333333", zorder=3)
    ax.set_xlim(-0.75, 4.55)
    ax.set_ylim(-0.75, 1.85)
    ax.set_aspect("equal")
    ax.set_axis_off()


def arrow(ax, p, q, color, text=None, toff=(0, 0.12), lw=1.2, fs=11, ms=12, zorder=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=zorder))
    if text:
        m = 0.5 * (np.array(p) + np.array(q)) + np.array(toff)
        ax.text(*m, text, ha="center", va="center", fontsize=fs, color=color, zorder=5)


fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.6, 5.4))
for ax in (axa, axb):
    draw_base(ax)
    for k in range(2):
        a, b = xs[k] + W_ / 2 + 0.04, xs[k + 1] - W_ / 2 - 0.04
        arrow(ax, (a, ytop), (b, ytop), GRAY, [r"$d_0$", r"$d_1$"][k])
        arrow(ax, (b, ybot), (a, ybot), GRAY, [r"$\tilde d_1$", r"$\tilde d_0$"][k], toff=(0, -0.13))
    for k, x in enumerate(xs):
        arrow(ax, (x - 0.12, ytop - H_ / 2 - 0.03), (x - 0.12, ybot + H_ / 2 + 0.03), GRAY, rf"$\star_{k}$",
              toff=(-0.17, 0), fs=10)
        arrow(ax, (x + 0.12, ybot + H_ / 2 + 0.03), (x + 0.12, ytop - H_ / 2 - 0.03), GRAY,
              rf"$\star_{k}^{{-1}}$", toff=(0.24, 0), fs=10)
# (a) Laplacian loop
y1, y0 = ytop + 0.36, ybot - 0.36
loop = [(xs[0] + 0.2, ytop + H_ / 2 + 0.02), (xs[0] + 0.25, y1), (xs[1] - 0.25, y1), (xs[1] - 0.2, ytop + H_ / 2 + 0.02)]
axa.plot(*np.array(loop).T, color=BLUE, lw=2.6, zorder=4)
axa.add_patch(FancyArrowPatch((xs[1] + 0.32, ytop - H_ / 2 + 0.02), (xs[1] + 0.32, ybot + H_ / 2 - 0.02),
                              arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=2.6, zorder=4))
axa.plot(*np.array([(xs[1] - 0.2, ybot - H_ / 2 - 0.02), (xs[1] - 0.25, y0), (xs[0] + 0.25, y0),
                    (xs[0] + 0.2, ybot - H_ / 2 - 0.02)]).T, color=BLUE, lw=2.6, zorder=4)
axa.add_patch(FancyArrowPatch((xs[0] - 0.33, ybot + H_ / 2 - 0.02), (xs[0] - 0.33, ytop - H_ / 2 + 0.02),
                              arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=2.6, zorder=4))
axa.text(0.95, y1 + 0.13, r"$d_0$", color=BLUE, fontsize=11, ha="center")
axa.text(0.95, y0 - 0.17, r"$\tilde d_1 = -d_0^{T}$", color=BLUE, fontsize=11, ha="center")
axa.set_title(r"(a) Laplacian: $\Delta \approx \star_0^{-1}\,\tilde d_1\,\star_1\,d_0 = \star_0^{-1}L$", fontsize=11.5,
              color="black")
# (b) divergence and curl
axb.add_patch(FancyArrowPatch((xs[1] + 0.32, ytop - H_ / 2 + 0.02), (xs[1] + 0.32, ybot + H_ / 2 - 0.02),
                              arrowstyle="-|>", mutation_scale=15, color=ORANGE, lw=2.6, zorder=4))
axb.plot(*np.array([(xs[1] - 0.2, ybot - H_ / 2 - 0.02), (xs[1] - 0.25, y0), (xs[0] + 0.25, y0),
                    (xs[0] + 0.2, ybot - H_ / 2 - 0.02)]).T, color=ORANGE, lw=2.6, zorder=4)
axb.add_patch(FancyArrowPatch((xs[0] - 0.33, ybot + H_ / 2 - 0.02), (xs[0] - 0.33, ytop - H_ / 2 + 0.02),
                              arrowstyle="-|>", mutation_scale=15, color=ORANGE, lw=2.6, zorder=4))
axb.plot(*np.array([(xs[1] + 0.2, ytop + H_ / 2 + 0.02), (xs[1] + 0.25, y1), (xs[2] - 0.25, y1),
                    (xs[2] - 0.2, ytop + H_ / 2 + 0.02)]).T, color=GREEN, lw=2.6, zorder=4)
axb.add_patch(FancyArrowPatch((xs[2] + 0.32, ytop - H_ / 2 + 0.02), (xs[2] + 0.32, ybot + H_ / 2 - 0.02),
                              arrowstyle="-|>", mutation_scale=15, color=GREEN, lw=2.6, zorder=4))
axb.text(0.95, y0 - 0.17, r"div $\approx \star_0^{-1}\tilde d_1\star_1 = -d_1^{*}$", color=ORANGE, fontsize=11,
         ha="center")
axb.text(2.85, y1 + 0.15, r"curl $\approx \star_2 d_1$", color=GREEN, fontsize=11, ha="center")
axb.set_title("(b) divergence and curl of a primal 1-form", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.12)
dgfig.save(fig, __file__)

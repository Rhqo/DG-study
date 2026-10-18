"""Figure 36.1.2: the de Rham map on the planar grid of Example 36.1.2 (Section 36.1).

Grid of the unit square with 3 x 3 vertices (spacing 1/2, alternating diagonals, faces counterclockwise).
(a) the 0-form f = x^2 - y^2: the numbers at the vertices are f(x_i); the numbers on the edges are the line
    integrals of df over the edges oriented i -> j (i < j, gray arrows), which equal f(x_j) - f(x_i).
(b) the 1-form x dy: the numbers are its line integrals over the oriented edges; edges are colored by sign
    (blue positive, vermillion negative, gray zero).
Self-check: R_1(df) = differences of R_0(f) (fundamental theorem), R_1(x dy) on a segment equals
(x_i + x_j)/2 (y_j - y_i), horizontal edges carry 0.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-1-2-grid-values.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

V, F = ch36.grid(2)
E, index = dd.edge_list(F)
f = lambda p: p[0] ** 2 - p[1] ** 2  # noqa: E731
df = lambda p: np.array([2 * p[0], -2 * p[1]])  # noqa: E731
xdy = lambda p: np.array([0.0, p[0]])  # noqa: E731
f_hat = dd.derham0(f, V)
df_hat = dd.derham1(df, V, E)
xdy_hat = dd.derham1(xdy, V, E)
assert np.allclose(df_hat, f_hat[E[:, 1]] - f_hat[E[:, 0]], atol=1e-14)
assert np.allclose(xdy_hat, 0.5 * (V[E[:, 0], 0] + V[E[:, 1], 0]) * (V[E[:, 1], 1] - V[E[:, 0], 1]), atol=1e-14)
assert np.allclose(xdy_hat[V[E[:, 0], 1] == V[E[:, 1], 1]], 0.0)

C = dgfig.COLORS
BLUE, RED, GRAY = C["tangent"], C["normal"], "#9A9A9A"


def fmt(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    s = "0" if s in ("-0", "") else s
    return s.replace("-", "−")


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.6))
for ax in (axa, axb):
    ch36.draw_mesh2d(ax, V, F, face="#F6F6F6", edge="#D0D0D0", lw=0.6)

# (a) vertex values of f and edge values of df
LABEL_BOX = dict(fc="#F7F7F7", ec="none", pad=0.6, alpha=0.9)  # keeps edge labels readable over arrows
cmap = plt.get_cmap("RdBu_r")
for i, p in enumerate(V):
    axa.plot(*p, "o", ms=9, color=cmap(0.5 + 0.45 * f_hat[i]), mec="black", mew=0.6, zorder=5)
    off = np.array([-0.075, 0.06]) if p[0] < 0.75 else np.array([0.075, 0.06])
    axa.text(*(p + off), fmt(f_hat[i]), fontsize=10, ha="center", va="center", zorder=6, fontweight="bold")
for e, (i, j) in enumerate(E):
    ch36.edge_arrow(axa, V[i], V[j], GRAY, lw=1.0, head=8)
    t = ch36.normal_offset(V[i], V[j], -0.055)
    axa.text(*t, fmt(df_hat[e]), fontsize=8.5, ha="center", va="center", color="#333333", zorder=6, bbox=LABEL_BOX)
axa.set_title(r"(a) $\hat f_i = f(x_i)$ and $\widehat{df}_{ij} = \hat f_j - \hat f_i$", fontsize=11)

# (b) edge values of x dy
for e, (i, j) in enumerate(E):
    v = xdy_hat[e]
    col = BLUE if v > 1e-12 else (RED if v < -1e-12 else GRAY)
    ch36.edge_arrow(axb, V[i], V[j], col, lw=1.6 if col != GRAY else 1.0, head=9)
    t = ch36.normal_offset(V[i], V[j], -0.055)
    axb.text(*t, fmt(v), fontsize=8.5, ha="center", va="center", color=col if col != GRAY else "#333333", zorder=6,
             bbox=LABEL_BOX)
for i, p in enumerate(V):
    axb.plot(*p, "o", ms=3.5, color="black", zorder=5)
axb.set_title(r"(b) $\widehat{x\,dy}_{ij} = \int_{[x_i, x_j]} x\,dy$", fontsize=11)
for ax in (axa, axb):
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.12, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()
axa.text(-0.05, -0.1, r"$(0,0)$", fontsize=9, color="#555555", ha="left")
axa.text(1.0, -0.1, r"$(1,0)$", fontsize=9, color="#555555", ha="center")
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)

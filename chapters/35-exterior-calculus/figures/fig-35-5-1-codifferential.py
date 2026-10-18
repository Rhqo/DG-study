"""Figure 35.5.1: the codifferential d* = -star d star in the plane, on a 1-form and on a 2-form
(Section 35.5, Theorem 35.5.8 and Example 35.5.9).

(a) The 1-form beta = X^flat with X = p exp(-|p|^2) (blue arrows on a 9 x 9 grid of [-2, 2]^2, drawn at 0.9 times
    their length). Background: the function d* beta = -div X = (2|p|^2 - 2) exp(-|p|^2), with the diverging colormap
    RdBu_r centred at 0 (red positive, blue negative). It is negative inside the circle |p| = 1, where X spreads out,
    and positive outside, where the flow slows down and converges.
(b) The 2-form omega = f dx ^ dy with the bump f = exp(-|p|^2) (background, sequential grey) and the vector field
    (d* omega)^sharp = (f_y, -f_x) = 2f(-y, x) (blue arrows, 0.5 times their length): it circulates counterclockwise
    around the bump, tangent to the level circles of f.
Self-checks (sympy): d* = -star d star on 1- and 2-forms of R^2 (lib/ch35.codifferential); d* X^flat = -div X;
(d* omega)^sharp = (f_y, -f_x) is orthogonal to grad f; the sign change of -div X at |p| = 1.
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import ch35  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
x, y = sp.symbols("x y", real=True)
g = sp.exp(-(x ** 2 + y ** 2))
Xs = [x * g, y * g]
dstar1 = ch35.codifferential(ch35.flat(Xs), 2, [x, y], 1)
assert sp.simplify(dstar1.get((), 0) + ch35.div(Xs, [x, y])) == 0
assert sp.simplify(dstar1.get((), 0) - (2 * (x ** 2 + y ** 2) - 2) * g) == 0
dstar2 = ch35.codifferential({(0, 1): g}, 2, [x, y], 2)
vec = [dstar2.get((0,), 0), dstar2.get((1,), 0)]
assert sp.simplify(vec[0] - sp.diff(g, y)) == 0 and sp.simplify(vec[1] + sp.diff(g, x)) == 0
assert sp.simplify(vec[0] * sp.diff(g, x) + vec[1] * sp.diff(g, y)) == 0

fx = sp.lambdify((x, y), dstar1.get((), 0), "numpy")
lim = 2.1
s = np.linspace(-lim, lim, 301)
XX, YY = np.meshgrid(s, s)
D1 = fx(XX, YY)
assert D1[150, 150] < 0 and D1[150, 0] > 0
G = np.exp(-(XX ** 2 + YY ** 2))
grid = np.linspace(-2, 2, 9)


def arrows(ax, field, scale):
    for px in grid:
        for py in grid:
            v = scale * np.asarray(field(px, py))
            if np.hypot(*v) < 0.02:
                continue
            ax.add_patch(FancyArrowPatch((px, py), (px + v[0], py + v[1]), arrowstyle="-|>", mutation_scale=8,
                                         color=C["tangent"], lw=1.1, shrinkA=0, shrinkB=0, zorder=5))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.55))
vmax = np.abs(D1).max()
imA = axA.imshow(D1, extent=(-lim, lim, -lim, lim), origin="lower", cmap="RdBu_r",
                 norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), alpha=0.8, zorder=0)
ts = np.linspace(0, 2 * np.pi, 200)
axA.plot(np.cos(ts), np.sin(ts), color=C["aux"], lw=0.9, ls="--", zorder=2)
arrows(axA, lambda a, b: (a * np.exp(-(a * a + b * b)), b * np.exp(-(a * a + b * b))), 0.9)
dgfig.schematic_axes(axA, xlim=(-lim, lim), ylim=(-lim, lim))
axA.set_title(r"(a) $d^*X^\flat = -\mathrm{div}\,X$", fontsize=12)
cb = fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.02)
cb.ax.tick_params(labelsize=8)

axB.imshow(G, extent=(-lim, lim, -lim, lim), origin="lower", cmap="Greys", vmin=0, vmax=1.6, zorder=0)
arrows(axB, lambda a, b: (-2 * b * np.exp(-(a * a + b * b)), 2 * a * np.exp(-(a * a + b * b))), 0.5)
dgfig.schematic_axes(axB, xlim=(-lim, lim), ylim=(-lim, lim))
axB.set_title(r"(b) $(d^*(f\,dx\wedge dy))^\sharp = (f_y, -f_x)$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9, wspace=0.12)
dgfig.save(fig, __file__)

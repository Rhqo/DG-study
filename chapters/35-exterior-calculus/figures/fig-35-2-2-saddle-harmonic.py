"""Figure 35.2.2: the differential of the saddle height f = x^2 - y^2 and its Hodge star (Section 35.2,
Example 35.2.13).

Purple solid curves: level sets f = c, c = -2, -1.5, ..., 2 (the level lines of the 1-form df).
Purple dashed curves: level sets g = c of g = 2xy, the same values c. Since star df = 2x dy + 2y dx = dg, these are
the level lines of star df. Blue arrows: grad f = (df)^sharp = (2x, -2y) on a 7 x 7 grid, drawn at 0.12 times their
length.
Because star df = dg is exact, d star df = 0 and Delta f = star d star df = 0. At every point the two families cross at
right angles (grad f is orthogonal to grad g) with the same spacing (|grad f| = |grad g|).
Self-checks (sympy): star df = d(2xy), Delta f = 0, <grad f, grad g> = 0, |grad f| = |grad g|; every blue arrow is
orthogonal to the solid level curve through its base point (numerically).
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
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
x, y = sp.symbols("x y", real=True)
fs = x ** 2 - y ** 2
gs = 2 * x * y
df = ch35.d({(): fs}, [x, y])
assert ch35.equal(ch35.hodge(df, 2), ch35.d({(): gs}, [x, y]))
assert sp.simplify(ch35.laplacian(fs, [x, y])) == 0
gf, gg = ch35.grad(fs, [x, y]), ch35.grad(gs, [x, y])
assert sp.simplify(gf[0] * gg[0] + gf[1] * gg[1]) == 0
assert sp.simplify(gf[0] ** 2 + gf[1] ** 2 - gg[0] ** 2 - gg[1] ** 2) == 0

lim = 1.55
s = np.linspace(-lim, lim, 401)
XX, YY = np.meshgrid(s, s)
F, G = XX ** 2 - YY ** 2, 2 * XX * YY
levels = np.arange(-2.0, 2.01, 0.5)

fig, ax = plt.subplots(figsize=(4.6, 4.75))
ax.contour(XX, YY, F, levels=levels, colors=C["covector"], linewidths=1.0, linestyles="solid", zorder=2)
ax.contour(XX, YY, G, levels=levels, colors=C["covector"], linewidths=1.0, linestyles=[(0, (4, 2.5))], zorder=2)
g = np.linspace(-1.2, 1.2, 7)
for px in g:
    for py in g:
        v = 0.12 * np.array([2 * px, -2 * py])
        if np.hypot(*v) < 1e-9:
            continue
        # orthogonal to the level curve of f through (px, py): the tangent of that curve is (2py, 2px)
        assert abs(v @ np.array([2 * py, 2 * px])) < 1e-12
        ax.add_patch(FancyArrowPatch((px, py), (px + v[0], py + v[1]), arrowstyle="-|>", mutation_scale=9,
                                     color=C["tangent"], lw=1.2, shrinkA=0, shrinkB=0, zorder=5))
ax.plot([0], [0], "o", color=C["main"], ms=3, zorder=6)
ax.plot([-1.5, -1.2], [-1.78, -1.78], color=C["covector"], lw=1.2)
ax.text(-1.13, -1.78, r"$f = x^2 - y^2 = c$ (level lines of $df$)", va="center", fontsize=10)
ax.plot([-1.5, -1.2], [-2.0, -2.0], color=C["covector"], lw=1.2, ls=(0, (4, 2.5)))
ax.text(-1.13, -2.0, r"$2xy = c$ (level lines of $\star df$)", va="center", fontsize=10)
ax.add_patch(FancyArrowPatch((-1.5, -2.22), (-1.2, -2.22), arrowstyle="-|>", mutation_scale=9, color=C["tangent"],
                             lw=1.2))
ax.text(-1.13, -2.22, r"$\mathrm{grad}\, f = (df)^\sharp$", va="center", fontsize=10)
dgfig.schematic_axes(ax, xlim=(-lim, lim), ylim=(-2.35, lim))
dgfig.save(fig, __file__)

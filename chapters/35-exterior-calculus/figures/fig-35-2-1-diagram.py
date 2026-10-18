"""Figure 35.2.1: the diagram view of exterior calculus on R^3 (Section 35.2, Proposition 35.2.9 and
Proposition 35.2.12). Schematic.

Top row: Omega^0 -> Omega^1 -> Omega^2 -> Omega^3 joined by d (labels: grad, curl, div under the identification of
Example 35.1.11). Bottom row: Omega^3, Omega^2, Omega^1, Omega^0, placed so that each Omega^k of the top row sits
above Omega^(3-k); the bottom d arrows point to the left. Vertical double arrows are the Hodge star.
The orange loop Omega^0 -> Omega^1 -> (star) Omega^2 -> Omega^3 -> (star) Omega^0 is the Laplacian star d star d.
Self-checks (sympy, through lib/ch35.py): on R^3, star d star d f = f_xx + f_yy + f_zz, star star = 1 on every
degree, and d d = 0 on 0- and 1-forms with general coefficient functions.
"""
import sys
from pathlib import Path

import sympy as sp

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import ch35  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = dgfig.COLORS
x, y, z = sp.symbols("x y z", real=True)
X3 = [x, y, z]
f = sp.Function("f")(x, y, z)
lap = ch35.hodge(ch35.d(ch35.hodge(ch35.d({(): f}, X3), 3), X3), 3)
assert sp.simplify(lap.get((), 0) - ch35.laplacian(f, X3)) == 0
for k in range(4):
    for I in ch35.increasing(3, k):
        assert ch35.hodge(ch35.hodge({I: 1}, 3), 3) == {I: 1}
assert ch35.d(ch35.d({(): f}, X3), X3) == {}
A = [sp.Function(n)(x, y, z) for n in ("A1", "A2", "A3")]
assert ch35.d(ch35.d(ch35.one_form(A), X3), X3) == {}

fig, ax = plt.subplots(figsize=(6.6, 2.75))
xs = [0.0, 2.0, 4.0, 6.0]
top, bot = 1.6, 0.0


def node(xc, yc, label):
    ax.add_patch(FancyBboxPatch((xc - 0.42, yc - 0.27), 0.84, 0.54, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=C["main"], lw=1.0, zorder=3))
    ax.text(xc, yc, label, ha="center", va="center", fontsize=13, zorder=4)


def arrow(p, q, color=C["main"], lw=1.2, style="-|>", zorder=2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, shrinkA=0,
                                 shrinkB=0, zorder=zorder, connectionstyle=f"arc3,rad={rad}"))


for k, xc in enumerate(xs):
    node(xc, top, rf"$\Omega^{k}$")
    node(xc, bot, rf"$\Omega^{3 - k}$")
names = ["grad", "curl", "div"]
for k in range(3):
    arrow((xs[k] + 0.45, top), (xs[k + 1] - 0.45, top))
    ax.text(0.5 * (xs[k] + xs[k + 1]), top + 0.12, r"$d$", ha="center", va="bottom", fontsize=12)
    ax.text(0.5 * (xs[k] + xs[k + 1]), top - 0.12, names[k], ha="center", va="top", fontsize=10, color="#555555")
    arrow((xs[k + 1] - 0.45, bot), (xs[k] + 0.45, bot))
    ax.text(0.5 * (xs[k] + xs[k + 1]), bot + 0.12, r"$d$", ha="center", va="bottom", fontsize=12)
for xc in xs:
    arrow((xc, top - 0.29), (xc, bot + 0.29), style="<|-|>")
    ax.text(xc + 0.1, 0.5 * (top + bot), r"$\star$", ha="left", va="center", fontsize=14)
# Laplacian loop: Omega^0 -d-> Omega^1 -star-> Omega^2 (bottom, under Omega^1) -d-> Omega^3 (bottom, under
# Omega^0) -star-> Omega^0
loop = [(xs[0] + 0.3, top + 0.42), (xs[1] - 0.3, top + 0.42), (xs[1] + 0.62, top + 0.42), (xs[1] + 0.62, bot - 0.42),
        (xs[0] - 0.62, bot - 0.42), (xs[0] - 0.62, top + 0.42), (xs[0] + 0.3, top + 0.42)]
lx, ly = zip(*loop)
ax.plot(lx, ly, color=C["accent"], lw=2.0, zorder=1, alpha=0.9)
for (p, q) in [(loop[1], loop[2]), (loop[3], loop[4])]:
    m = (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))
    d_ = (q[0] - p[0], q[1] - p[1])
    nrm = (d_[0] ** 2 + d_[1] ** 2) ** 0.5
    arrow((m[0] - 0.08 * d_[0] / nrm, m[1] - 0.08 * d_[1] / nrm), (m[0] + 0.08 * d_[0] / nrm, m[1] + 0.08 * d_[1] / nrm),
          color=C["accent"], lw=2.0, zorder=1)
ax.text(xs[0] - 0.75, 0.5 * (top + bot), r"$\Delta = \star d \star d$", ha="right", va="center", fontsize=12,
        color="#9A6200")
ax.text(xs[3] + 0.55, top, r"$d\circ d = 0$", ha="left", va="center", fontsize=11, color="#555555")
dgfig.schematic_axes(ax, xlim=(-2.25, 7.55), ylim=(-0.6, 2.2))
dgfig.save(fig, __file__)

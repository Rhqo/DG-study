"""Figure 35.1.1: a vector field and a differential 1-form on the plane, and the function alpha(X)
(Section 35.1, Example 35.1.7).

(a) The rotation field X = -y d/dx + x d/dy on a 7 x 7 grid of [-1.5, 1.5]^2 (blue arrows, drawn at 0.22 times
    their length).
(b) The 1-form alpha = x dy. At each grid point p the covector alpha_p = x e^2 is drawn as a short stack of
    horizontal level lines (purple): the number of lines in a window of height 0.4 is round(2 |x|), i.e. the
    density is proportional to |alpha_p| = |x| (schematic stacks; the density is computed). A small purple
    arrowhead shows the direction in which alpha_p increases (sign(x) e_2). The background is the function
    alpha(X) = x^2 (Example 35.1.7).
Self-checks: alpha(X) computed pointwise equals x^2 on the grid; the stack counts are proportional to |x|;
the two marked points give alpha(X) = 1 at (1, 0) and 0 at (0, 1).
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
g = np.linspace(-1.5, 1.5, 7)
P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def X(p):
    return np.array([-p[1], p[0]])


def alpha(p):
    return np.array([0.0, p[0]])


vals = np.array([alpha(p) @ X(p) for p in P])
assert np.allclose(vals, P[:, 0] ** 2)
assert abs(alpha(np.array([1.0, 0.0])) @ X(np.array([1.0, 0.0])) - 1.0) < 1e-15
assert abs(alpha(np.array([0.0, 1.0])) @ X(np.array([0.0, 1.0]))) < 1e-15
counts = {round(x, 2): int(round(2 * abs(x))) for x in g}
assert counts[1.5] == 3 and counts[1.0] == 2 and counts[0.5] == 1 and counts[0.0] == 0


def arrow(ax, p, v, color, lw=1.3, ms=9, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + v), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.45))
lim = (-1.95, 1.95)

# (a) vector field
for p in P:
    arrow(axA, p, 0.22 * X(p), C["tangent"])
axA.plot([0], [0], "o", color=C["main"], ms=2.5)
axA.text(0.0, -1.92, r"$X = -y\,\partial/\partial x + x\,\partial/\partial y$", ha="center", fontsize=10.5,
         color=C["tangent"])
dgfig.schematic_axes(axA, xlim=lim, ylim=(-2.15, 1.95))
axA.set_title("(a) vector field $X$", fontsize=12)

# (b) 1-form stacks over alpha(X) = x^2
xs = np.linspace(lim[0], lim[1], 200)
XX, YY = np.meshgrid(xs, xs)
axB.imshow(XX ** 2, extent=(lim[0], lim[1], lim[0], lim[1]), origin="lower", cmap="Greys", vmin=0, vmax=9,
           alpha=0.9, zorder=0)
for p in P:
    m = int(round(2 * abs(p[0])))
    if m == 0:
        continue
    for k in range(m):
        yk = p[1] - 0.2 + 0.4 * (k + 0.5) / m
        axB.plot([p[0] - 0.16, p[0] + 0.16], [yk, yk], color=C["covector"], lw=1.1, zorder=3)
    s = np.sign(p[0])
    arrow(axB, np.array([p[0] + 0.21, p[1] - 0.09 * s]), np.array([0.0, 0.18 * s]), C["covector"], lw=0.9, ms=6,
          zorder=4)
for q, txt, off in [((1.0, 0.0), r"$\alpha(X) = 1$", (0.05, -0.42)), ((0.0, 1.0), r"$\alpha(X) = 0$", (0.0, 0.3))]:
    q = np.array(q)
    arrow(axB, q, 0.22 * X(q), C["tangent"], lw=1.4, zorder=6)
    axB.plot(*q, "o", color=C["main"], ms=3, zorder=7)
    axB.text(q[0] + off[0], q[1] + off[1], txt, ha="center", fontsize=10.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
axB.text(0.0, -2.12, r"$\alpha = x\,dy$, background $\alpha(X) = x^2$", ha="center", fontsize=10.5,
         color="#333333")
dgfig.schematic_axes(axB, xlim=lim, ylim=(-2.3, 1.95))
axB.set_title(r"(b) 1-form $\alpha$ and $\alpha(X)$", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92, wspace=0.06)
dgfig.save(fig, __file__)

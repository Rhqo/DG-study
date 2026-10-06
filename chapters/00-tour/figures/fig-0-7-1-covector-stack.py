"""Figure 0.7.1: a covector as a stack of level lines (Section 0.7).

f(x, y) = x² + y², p = (1, 1/2). Then df_p = 2 dx + dy.
Left: level curves f = c (c = 0.25, 0.5, …, 2.5, grey) and the point p; the small square is the zoom window of the right panel.
Right (zoom around p): the level lines of the linear function df_p, i.e. the lines 2(x − 1) + (y − 1/2) = k·h with h = 1/8
(pink, the "stack" of df_p), the true level curves of f (thin grey), and the vector v = (1/4, 1/4) at p (blue).
df_p(v) = 2·(1/4) + 1/4 = 3/4, so v crosses (3/4)/h = 6 lines of the stack.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-7-1-covector-stack.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C = dgfig.COLORS
LW = dgfig.LW


def f(x, y):
    return x ** 2 + y ** 2


p = np.array([1.0, 0.5])
grad = np.array([2 * p[0], 2 * p[1]])          # components of df_p
v = np.array([0.25, 0.25])
h = 0.125
val = grad @ v

# self-checks (§12.1)
eps = 1e-6
num = np.array([(f(p[0] + eps, p[1]) - f(p[0] - eps, p[1])) / (2 * eps),
                (f(p[0], p[1] + eps) - f(p[0], p[1] - eps)) / (2 * eps)])
assert np.allclose(num, [2.0, 1.0])
assert np.isclose(val, 0.75) and np.isclose(val / h, 6)
# the segment from p to p + v meets exactly 6 lines k = 1, …, 6 of the stack (k = 0 passes through p)
ks = [k for k in range(-20, 21) if 0 < k * h <= val + 1e-12]
assert ks == [1, 2, 3, 4, 5, 6]

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1]))

# left: level curves of f
ax = axs[0]
dgfig.schematic_axes(ax, (-0.2, 1.75), (-0.2, 1.75))
t = np.linspace(0, np.pi / 2, 200)
for c in np.arange(0.25, 2.76, 0.25):
    r = np.sqrt(c)
    ax.plot(r * np.cos(t), r * np.sin(t), color=C["aux"], lw=0.8)
for d in ((1, 0), (0, 1)):
    ax.annotate("", xy=(1.72 * d[0] - 0.15 * d[1], 1.72 * d[1] - 0.15 * d[0]),
                xytext=(-0.15 * d[0] - 0.15 * d[1], -0.15 * d[1] - 0.15 * d[0]),
                arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.7, mutation_scale=8))
ax.text(1.72, -0.12, r"$x$", fontsize=11, ha="right", va="top")
ax.text(-0.1, 1.72, r"$y$", fontsize=11, ha="right", va="top")
W = 0.32
ax.add_patch(Rectangle(p - W, 2 * W, 2 * W, fill=False, ec=C["main"], lw=0.8, ls=(0, (3, 2))))
ax.plot(*p, "o", color=C["main"], ms=4, zorder=5)
ax.text(p[0] + 0.05, p[1] - 0.12, r"$p$", fontsize=12)
ax.text(0.15, 1.48, r"$f = c$", fontsize=11, color=C["aux"])

# right: zoom with the stack of df_p
ax = axs[1]
dgfig.schematic_axes(ax, (p[0] - W, p[0] + W), (p[1] - W, p[1] + W))
xs = np.linspace(p[0] - W, p[0] + W, 3)
for k in range(-16, 17):
    # 2(x − 1) + (y − 1/2) = k h  →  y = 1/2 + k h − 2(x − 1)
    ys = p[1] + k * h - grad[0] * (xs - p[0])
    ax.plot(xs, ys, color=C["covector"], lw=1.05 if k else 1.9, zorder=2)
tt = np.linspace(0, 2 * np.pi, 600)
for c in np.arange(0.25, 2.76, 0.25):
    r = np.sqrt(c)
    ax.plot(r * np.cos(tt), r * np.sin(tt), color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.annotate("", xy=p + v, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=LW["vector"] + 0.3,
                                                     mutation_scale=13), zorder=6)
ax.plot(*p, "o", color=C["main"], ms=4, zorder=7)
box = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9)
ax.text(p[0] - 0.06, p[1] - 0.055, r"$p$", fontsize=12, bbox=box, zorder=8)
ax.text(p[0] + v[0] * 0.5 - 0.075, p[1] + v[1] * 0.5 + 0.035, r"$v$", fontsize=13, color=C["tangent"], bbox=box,
        zorder=8)
ax.text(p[0] - W + 0.02, p[1] + W - 0.06, r"$df_p = 2\,dx + dy$", fontsize=11, color=C["covector"], bbox=box,
        zorder=8)
ax.text(p[0] + 0.0, p[1] - W + 0.025, r"$df_p(v) = 3/4$", fontsize=11, bbox=box, zorder=8)
ax.add_patch(Rectangle(p - W, 2 * W, 2 * W, fill=False, ec=C["main"], lw=0.8, ls=(0, (3, 2))))

fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.12)
dgfig.save(fig, __file__)

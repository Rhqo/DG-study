"""Figure 41.1.1: the direction of a gradient is found from the directions that do not change the quantity.

Triangle a = (0, 0), b = (4, 0), c = (1, 1) (the obtuse triangle of Example 41.1.5, counterclockwise, N = e_3).
(a) Level sets of the area A as a function of the position of c (the other two fixed): horizontal lines (A = 2 y),
    parallel to the opposite edge ab. Blue arrows: the three area gradients (1/2) N x (opposite edge), drawn at half
    length; |grad_c A| = 2 = l_ab / 2.
(b) Level sets of the angle alpha at a as a function of the position of b: rays from a. The gray circle is the circle
    about a through b; moving b along it changes alpha at the rate 1/|b - a| = 1/4. Arrows: grad_b alpha, grad_c alpha,
    grad_a alpha (Proposition 41.1.6), drawn at twice their length.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-1-1-level-sets.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402

C = dgfig.COLORS
a, b, c = np.array([0.0, 0, 0]), np.array([4.0, 0, 0]), np.array([1.0, 1, 0])
P = np.array([a, b, c])
G = dd.triangle_area_gradients(P, np.array([[0, 1, 2]]))[0]
assert np.allclose(G, [[-0.5, -1.5, 0], [0.5, -0.5, 0], [0, 2, 0]])
assert np.allclose(G.sum(0), 0)
ga, gb, gc = dd.angle_gradient(a, b, c)
assert np.allclose(gb, [0, -0.25, 0]) and np.allclose(gc, [-0.5, 0.5, 0]) and np.allclose(ga, [0.5, -0.25, 0])
assert np.isclose(np.degrees(np.arctan2(1, 1)), 45.0)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
tri = np.vstack([P, P[:1]])
for ax in axes:
    ax.fill(tri[:, 0], tri[:, 1], color=C["surface"], alpha=0.6, lw=0)
    ax.plot(tri[:, 0], tri[:, 1], color=C["main"], lw=1.4)
    for p, name, off in [(a, r"$\mathbf{x}_a$", (-0.42, -0.05)), (b, r"$\mathbf{x}_b$", (0.12, 0.08)),
                         (c, r"$\mathbf{x}_c$", (-0.45, 0.05))]:
        ax.plot(*p[:2], "o", color=C["main"], ms=3.5)
        ax.text(p[0] + off[0], p[1] + off[1], name, fontsize=12)
    ax.set_aspect("equal")
    ax.set_axis_off()

ax = axes[0]
for y, lab in [(0.5, r"$A=1$"), (1.0, r"$A=2$"), (1.5, r"$A=3$"), (2.0, r"$A=4$")]:
    ax.plot([-0.6, 4.4], [y, y], color=C["aux"], lw=0.8, ls="--")
    ax.text(4.45, y, lab, fontsize=10, color=C["aux"], va="center")
for p, g in zip(P, G):
    ax.annotate("", p[:2] + 0.5 * g[:2], p[:2], arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.6,
                                                               mutation_scale=11))
ax.text(1.1, 1.75, r"$\frac{1}{2}\nabla_{\mathbf{x}_c}A$", color=C["tangent"], fontsize=12)
ax.set_xlim(-0.7, 5.3)
ax.set_ylim(-0.95, 2.25)
ax.set_title("(a)", loc="left", fontsize=12)

ax = axes[1]
for deg in (-24, -12, 0, 12, 24):
    t = np.radians(deg)
    ax.plot([0, 4.9 * np.cos(t)], [0, 4.9 * np.sin(t)], color=C["aux"], lw=0.8, ls="--")
ax.text(3.35, 1.98, r"$\alpha$" + " = const", color=C["aux"], fontsize=10)
s = np.linspace(-0.5, 0.5, 100)
ax.plot(4 * np.cos(s), 4 * np.sin(s), color=C["aux"], lw=0.9)
for p, g in [(a, ga), (b, gb), (c, gc)]:
    ax.annotate("", p[:2] + 2 * g[:2], p[:2], arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.6,
                                                             mutation_scale=11))
ax.text(4.12, -0.62, r"$2\nabla_{\mathbf{x}_b}\alpha$", color=C["accent"], fontsize=12)
ax.text(-0.75, 1.75, r"$2\nabla_{\mathbf{x}_c}\alpha$", color=C["accent"], fontsize=12)
ax.text(0.65, -0.85, r"$2\nabla_{\mathbf{x}_a}\alpha$", color=C["accent"], fontsize=12)
ax.text(0.55, 0.12, r"$\alpha$", fontsize=12)
ax.set_xlim(-0.9, 5.1)
ax.set_ylim(-1.9, 2.1)
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.06)
dgfig.save(fig, __file__)

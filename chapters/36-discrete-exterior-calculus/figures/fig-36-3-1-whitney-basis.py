"""Figure 36.3.1: the three Whitney 1-forms of a triangle as vector fields (Section 36.3).

Triangle T = [x_0, x_1, x_2] with x_0 = (0, 0), x_1 = (1, 0), x_2 = (0.35, 0.9) (the triangle of Figure 33.1.1(b)),
counterclockwise. Panel (a), (b), (c) show the vector proxy (Euclidean sharp) of phi_01, phi_12, phi_20 where
phi_ab = phi_a d(phi_b) - phi_b d(phi_a), sampled on the barycentric grid with step 1/6 (points on the edges included).
The edge [a, b] whose Whitney form is drawn is orange with its direction a -> b.
Self-check: the integral of phi_ab over its own edge is 1 and over the other two edges 0; along its own edge the
tangential component phi_ab(x_b - x_a) is identically 1; on the other two edges the arrows are perpendicular to
the edge (tangential component 0); div of the proxy is 0 and its curl (d phi_ab) is 1/A on the triangle.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-3-1-whitney-basis.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

T = np.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.9]])
A = 0.5 * 0.9
G = dd.barycentric_gradients(T)
assert np.allclose(G.sum(0), 0)
pairs = ((0, 1), (1, 2), (2, 0))
m = 6
B = np.array([[i / m, j / m, (m - i - j) / m] for i in range(m + 1) for j in range(m + 1 - i)])
B = B[:, [2, 0, 1]]                       # (lambda_0, lambda_1, lambda_2)
P = B @ T
tq, wq = dd.gauss_legendre_01(4)
for (a, b) in pairs:
    # integrals over the three edges
    for (c, d) in pairs:
        val = sum(w * dd.whitney_basis(np.eye(3)[c] * (1 - t) + np.eye(3)[d] * t, G, a, b) @ (T[d] - T[c])
                  for t, w in zip(tq, wq))
        assert abs(val - (1.0 if (c, d) == (a, b) else 0.0)) < 1e-14
    W = dd.whitney_basis(B, G, a, b)
    on_ab = np.isclose(B[:, a] + B[:, b], 1.0)
    assert np.allclose(W[on_ab] @ (T[b] - T[a]), 1.0)
    for (c, d) in pairs:
        if (c, d) != (a, b):
            on = np.isclose(B[:, c] + B[:, d], 1.0)
            assert np.allclose(W[on] @ (T[d] - T[c]), 0.0)
    # curl: d phi_ab = 2 d(phi_a) ^ d(phi_b), whose density is 2 det[G_a G_b] = 1 / A for a counterclockwise pair
    assert abs(2 * (G[a, 0] * G[b, 1] - G[a, 1] * G[b, 0]) - 1 / A) < 1e-12

C = dgfig.COLORS
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
scale = 0.16
names = [r"(a) $\varphi_{01}$", r"(b) $\varphi_{12}$", r"(c) $\varphi_{20}$"]
for ax, (a, b), name in zip(axes, pairs, names):
    ax.add_patch(Polygon(T, closed=True, facecolor="#F3F3F3", edgecolor="#9A9A9A", lw=1.0, zorder=0))
    ax.plot(*T[[a, b]].T, color=C["accent"], lw=3.0, zorder=1)
    ax.add_patch(FancyArrowPatch(T[a] + 0.45 * (T[b] - T[a]), T[a] + 0.7 * (T[b] - T[a]), arrowstyle="-|>",
                                 mutation_scale=18, color=C["accent"], lw=2.8, zorder=5))
    W = dd.whitney_basis(B, G, a, b)
    for p, v in zip(P, W):
        ax.add_patch(FancyArrowPatch(p, p + scale * v, arrowstyle="-|>", mutation_scale=6, color=C["tangent"],
                                     lw=0.9, shrinkA=0, shrinkB=0, zorder=3))
    for v, off in zip(range(3), ([-0.08, -0.07], [0.06, -0.07], [0.0, 0.07])):
        ax.text(*(T[v] + np.array(off)), rf"$x_{v}$", fontsize=11, ha="center", va="center")
    ax.set_title(name, fontsize=12)
    ax.set_xlim(-0.2, 1.25)
    ax.set_ylim(-0.17, 1.05)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.04)
dgfig.save(fig, __file__)

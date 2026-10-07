"""Figure 29.5.1: one covector, two metrics, two gradients (Section 29.5).

At a point p (origin) the differential dL_p = alpha = (2, 1) (a covector) is drawn as its level lines
alpha(v) = k/2, k = -4..4 (purple). (a) Euclidean metric g = I; (b) metric g = [[1, 1.2], [1.2, 3]].
Blue curve: the unit ball {|v|_g = 1}. Blue arrow: the unit vector of steepest ascent (direction of grad_g L = g^{-1} alpha^T, whose value is printed). Orange point: the point of the unit ball
where alpha is largest (steepest ascent direction); it lies on grad_g L and the level line through it is tangent to the
unit ball (thicker purple line).
Self-checks: the orange point maximizes alpha over 2000 points of the unit ball; it is parallel to g^{-1} alpha.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-5-1-covector-metric.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
alpha = np.array([2.0, 1.0])
metrics = [(np.eye(2), r"(a) $g = I$"), (np.array([[1.0, 1.2], [1.2, 3.0]]), r"(b) $g = [[1, 1.2], [1.2, 3]]$")]
t = np.linspace(0, 2 * np.pi, 2000)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True, gridspec_kw=dict(wspace=0.08))
L = 2.6
for ax, (g, title) in zip(axes, metrics):
    # level lines of alpha: 2x + y = c
    xs = np.array([-L, L])
    for k in range(-6, 7):
        c = k / 2
        ax.plot(xs, c - 2 * xs, color=C["covector"], lw=0.7, alpha=0.8, zorder=1)
    w, V = np.linalg.eigh(g)
    ball = (np.stack([np.cos(t), np.sin(t)], 1) / np.sqrt(w)) @ V.T      # |v|_g = 1
    assert np.allclose(np.einsum("ni,ij,nj->n", ball, g, ball), 1.0)
    vals = ball @ alpha
    vstar = ball[np.argmax(vals)]
    grad = np.linalg.solve(g, alpha)
    assert abs(abs(vstar @ grad) / (np.linalg.norm(vstar) * np.linalg.norm(grad)) - 1) < 1e-5
    cmax = vals.max()
    ax.plot(xs, cmax - 2 * xs, color=C["covector"], lw=2.2, zorder=2)
    ax.fill(*ball.T, color=C["region"], alpha=0.25, lw=0, zorder=2)
    ax.plot(*ball.T, color=C["tangent"], lw=1.6, zorder=3)
    ax.annotate("", xy=vstar, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.8,
                                                              mutation_scale=13, shrinkA=0, shrinkB=0), zorder=5)
    ax.plot(*vstar, "o", color=C["accent"], ms=6, zorder=6)
    ax.plot(0, 0, "o", color=C["main"], ms=4, zorder=6)
    ax.text(-2.5, 2.45, r"$\mathrm{grad}_g L = g^{-1}dL^{\mathsf{T}}$" + "\n" + r"$= (%.2f,\ %.2f)$" % tuple(grad),
            color=C["tangent"], fontsize=10, va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(-2.5, -2.45, r"unit ball $|v|_g = 1$", color=C["tangent"], fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$v^1$")
    ax.tick_params(labelsize=9)
axes[0].set_ylabel(r"$v^2$")
axes[0].text(1.0, -1.7, r"$dL = 2\,dv^1 + dv^2$" + "\n(level lines)", color="#9A4F7F", fontsize=10)

dgfig.save(fig, __file__)

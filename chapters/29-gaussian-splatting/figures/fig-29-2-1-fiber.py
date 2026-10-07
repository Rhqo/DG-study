"""Figure 29.2.1: four (R, s) that give the same 2D Gaussian (Section 29.2).

The ellipse Sigma = R(30) diag(2^2, 0.7^2) R(30)^T drawn four times. Panel k uses R = R(30 + 90 k) and
s = (2, 0.7) for even k, s = (0.7, 2) for odd k (R(theta) = rotation by theta degrees). Blue arrow: s_1 r_1,
green arrow: s_2 r_2 (r_i = i-th column of R). In 2D the signed permutation matrices of determinant +1 are the four
rotations by multiples of 90 degrees; in 3D there are 24 of them.
Self-check: all four (R, s) give the same Sigma; the four columns r_1 are pairwise different.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-2-1-fiber.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS


def R(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


Sigma = R(30) @ np.diag([4.0, 0.49]) @ R(30).T
params = [(30 + 90 * k, (2.0, 0.7) if k % 2 == 0 else (0.7, 2.0)) for k in range(4)]
for th, s in params:
    assert np.allclose(R(th) @ np.diag(np.square(s)) @ R(th).T, Sigma)
r1s = [R(th)[:, 0] for th, _ in params]
assert all(np.linalg.norm(r1s[i] - r1s[j]) > 1 for i in range(4) for j in range(i + 1, 4))

fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.4))
t = np.linspace(0, 2 * np.pi, 300)
E = np.stack([np.cos(t), np.sin(t)], 1) @ (R(30) @ np.diag([2.0, 0.7])).T
arrow_kw = dict(arrowstyle="-|>", lw=1.7, mutation_scale=13, shrinkA=0, shrinkB=0)
for k, (ax, (th, s)) in enumerate(zip(axes.ravel(), params)):
    ax.fill(*E.T, color=C["region"], alpha=0.3, lw=0)
    ax.plot(*E.T, color=C["main"], lw=1.6)
    Rk = R(th)
    v1, v2 = s[0] * Rk[:, 0], s[1] * Rk[:, 1]
    ax.annotate("", xy=v1, xytext=(0, 0), arrowprops=dict(color=C["tangent"], **arrow_kw))
    ax.annotate("", xy=v2, xytext=(0, 0), arrowprops=dict(color=C["third"], **arrow_kw))
    ax.text(*(v1 * (1 + 0.28 / np.linalg.norm(v1))), r"$s_1r_1$", color=C["tangent"], fontsize=12, ha="center",
            va="center")
    ax.text(*(v2 * (1 + 0.42 / np.linalg.norm(v2))), r"$s_2r_2$", color=C["third"], fontsize=12, ha="center",
            va="center")
    ax.plot(0, 0, "o", color=C["main"], ms=3)
    ax.set_title(f"({'abcd'[k]}) " + r"$\theta = %d^\circ$, $s = (%.1f, %.1f)$" % (th, s[0], s[1]), fontsize=11.5)
    dgfig.schematic_axes(ax, (-2.6, 2.6), (-2.1, 2.1))
fig.subplots_adjust(hspace=0.25, wspace=0.08)

dgfig.save(fig, __file__)

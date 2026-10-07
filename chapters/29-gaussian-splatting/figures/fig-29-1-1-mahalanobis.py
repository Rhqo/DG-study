"""Figure 29.1.1: Euclidean distance vs. Mahalanobis distance (Section 29.1).

Same Gaussian as Figure 29.1.2: mu = (1, 0.5), R = rotation by 30 degrees, s = (2, 0.7).
Shading: G(x) = exp(-|x - mu|_Sigma^2 / 2). Black/gray curves: level sets k = 1, 2, 3.
Dashed gray circle: Euclidean circle |x - mu| = 2. Points p = mu + 2 r_1 and q = mu + 2 r_2 lie on it,
but |p - mu|_Sigma = 1 and |q - mu|_Sigma = 2/0.7 = 2.86.
Self-checks: these two Mahalanobis distances and G(p) = 0.607, G(q) = 0.017.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-1-1-mahalanobis.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
mu = np.array([1.0, 0.5])
a = np.radians(30.0)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
s = np.array([2.0, 0.7])
Sigma = R @ np.diag(s ** 2) @ R.T
Sinv = np.linalg.inv(Sigma)


def mdist(z):
    d = np.asarray(z) - mu
    return np.sqrt(np.einsum("...i,ij,...j->...", d, Sinv, d))


p = mu + 2.0 * R[:, 0]
q = mu + 2.0 * R[:, 1]
assert abs(mdist(p) - 1.0) < 1e-12 and abs(mdist(q) - 2 / 0.7) < 1e-12
assert round(float(np.exp(-0.5 * mdist(p) ** 2)), 3) == 0.607
assert round(float(np.exp(-0.5 * mdist(q) ** 2)), 3) == 0.017

fig, ax = plt.subplots(figsize=(6.0, 4.3))
xs = np.linspace(-5.0, 7.0, 400)
ys = np.linspace(-4.6, 4.6, 340)
X, Y = np.meshgrid(xs, ys)
Gv = np.exp(-0.5 * mdist(np.stack([X, Y], -1)) ** 2)
# Blues truncated at 1/1.15 so that the peak G = 1 is not the darkest navy (labels stay readable);
# the colorbar then runs exactly over the range of G, [0, 1].
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("BluesT", plt.cm.Blues(np.linspace(0, 1 / 1.15, 256)))
im = ax.imshow(Gv, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", cmap=cmap, vmin=0, vmax=1,
               interpolation="bilinear", zorder=0)
t = np.linspace(0, 2 * np.pi, 400)
circ = np.stack([np.cos(t), np.sin(t)], 1)
E = circ @ (R @ np.diag(s)).T
for k, lw, col in ((1, 1.8, C["main"]), (2, 1.0, "#444444"), (3, 1.0, "#444444")):
    ax.plot(*(mu + k * E).T, color=col, lw=lw, zorder=2)
    pk = mu - k * s[0] * R[:, 0] - 0.2 * R[:, 1]
    ax.text(pk[0], pk[1], f"$k={k}$", fontsize=10.5, ha="left", va="top", color=col, zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
ax.text(-4.8, -4.45, r"on $|x-\mu|_\Sigma = k$:  $G = e^{-k^2/2}$ $=$ 0.607, 0.135, 0.011 ($k = 1, 2, 3$)",
        fontsize=9.5, ha="left", va="bottom", bbox=dict(fc="white", ec="#BBBBBB", lw=0.6, pad=3), zorder=7)
ax.plot(*(mu + 2.0 * circ).T, color=C["aux"], lw=1.1, ls=(0, (4, 3)), zorder=3)
ax.text(*(mu + 2.0 * np.array([np.cos(np.radians(140)), np.sin(np.radians(140))]) + np.array([-0.1, 0.15])),
        r"$|x-\mu| = 2$", color=C["aux"], fontsize=10.5, ha="right")
arrow_kw = dict(arrowstyle="-|>", lw=1.5, mutation_scale=12, shrinkA=0, shrinkB=0)
ax.annotate("", xy=p, xytext=mu, arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=5)
ax.annotate("", xy=q, xytext=mu, arrowprops=dict(color=C["normal"], **arrow_kw), zorder=5)
ax.plot(*mu, "o", color=C["main"], ms=4, zorder=6)
ax.text(mu[0] + 0.12, mu[1] - 0.45, r"$\mu$", fontsize=12)
ax.plot(*p, "o", color=C["tangent"], ms=5, zorder=6)
ax.plot(*q, "o", color=C["normal"], ms=5, zorder=6)
ax.annotate(r"$p$:  $|p-\mu|=2$" + "\n" + r"$|p-\mu|_\Sigma = 1$,  $G = 0.607$", xy=p, xytext=(2.3, -1.25), fontsize=10.5,
            color=C["tangent"], arrowprops=dict(arrowstyle="-", color=C["tangent"], lw=0.7), ha="left", va="top",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
ax.annotate(r"$q$:  $|q-\mu|=2$" + "\n" + r"$|q-\mu|_\Sigma = 2.86$,  $G = 0.017$", xy=q, xytext=(-4.8, 3.5), fontsize=10.5,
            color=C["normal"], arrowprops=dict(arrowstyle="-", color=C["normal"], lw=0.7), ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
cb.set_label(r"$G(x)$", fontsize=11)
cb.ax.tick_params(labelsize=9)
ax.set_aspect("equal")
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ys[0], ys[-1])
ax.set_xlabel(r"$x^1$")
ax.set_ylabel(r"$x^2$")
ax.tick_params(labelsize=9)

dgfig.save(fig, __file__)

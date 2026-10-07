"""Figure 29.1.2: a Gaussian is the affine image of the standard Gaussian (Section 29.1).

(a) y-plane: level sets |y| = 1, 2, 3 of the standard Gaussian, a coordinate grid, basis vectors e_1, e_2, a point y*.
(b) x-plane: the same objects under x = mu + R S y with mu = (1, 0.5), R = rotation by 30 degrees, s = (2, 0.7).
    Level sets become the ellipses |x - mu|_Sigma = 1, 2, 3; e_i becomes s_i r_i (r_i = i-th column of R).
Self-checks: the image of the unit circle satisfies |x - mu|_Sigma = 1; s_i r_i has Euclidean length s_i.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-1-2-affine-image.py``
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
A = R @ np.diag(s)
Sigma = A @ A.T
Sinv = np.linalg.inv(Sigma)
ystar = np.array([1.3, 0.9])
xstar = mu + A @ ystar

t = np.linspace(0, 2 * np.pi, 400)
circ = np.stack([np.cos(t), np.sin(t)], 1)
img1 = circ @ A.T
assert np.allclose(np.einsum("ni,ij,nj->n", img1, Sinv, img1), 1.0)
assert np.allclose(np.linalg.norm(A[:, 0]), s[0]) and np.allclose(np.linalg.norm(A[:, 1]), s[1])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.45))
arrow_kw = dict(arrowstyle="-|>", lw=1.6, mutation_scale=13, shrinkA=0, shrinkB=0)

# ---------------- (a) y-plane
L = 3.4
grid_vals = np.arange(-3, 4)
for g in grid_vals:
    axa.plot([g, g], [-3, 3], color=C["aux"], lw=0.35, zorder=0)
    axa.plot([-3, 3], [g, g], color=C["aux"], lw=0.35, zorder=0)
for k, lw, col in ((1, 1.8, C["main"]), (2, 1.0, C["aux"]), (3, 1.0, C["aux"])):
    axa.plot(k * circ[:, 0], k * circ[:, 1], color=col, lw=lw, zorder=2)
axa.fill(circ[:, 0], circ[:, 1], color=C["region"], alpha=0.25, lw=0, zorder=1)
axa.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=5)
axa.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(color=C["third"], **arrow_kw), zorder=5)
axa.text(1.05, -0.32, r"$e_1$", color=C["tangent"], fontsize=12)
axa.text(-0.1, 0.72, r"$e_2$", color=C["third"], fontsize=12, ha="right")
axa.plot(*ystar, "o", color=C["accent"], ms=5, zorder=6)
axa.text(ystar[0] + 0.17, ystar[1] + 0.02, r"$y^*$", color=C["accent"], fontsize=12, va="bottom")
for k in (1, 2, 3):
    axa.text(k * np.cos(np.radians(-50)) + 0.05, k * np.sin(np.radians(-50)) - 0.05, f"$k={k}$", fontsize=10,
             color=C["main"] if k == 1 else C["aux"], ha="left", va="top")
axa.plot(0, 0, "o", color=C["main"], ms=3.5, zorder=6)
axa.set_aspect("equal")
axa.set_xlim(-L, L)
axa.set_ylim(-L, L)
axa.set_xlabel(r"$y^1$")
axa.set_ylabel(r"$y^2$")
axa.set_title(r"(a) $y$: standard Gaussian, $\Sigma = I$", fontsize=11)

# ---------------- (b) x-plane
for g in grid_vals:
    p0, p1 = mu + A @ np.array([g, -3]), mu + A @ np.array([g, 3])
    axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C["aux"], lw=0.35, zorder=0)
    p0, p1 = mu + A @ np.array([-3, g]), mu + A @ np.array([3, g])
    axb.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C["aux"], lw=0.35, zorder=0)
for k, lw, col in ((1, 1.8, C["main"]), (2, 1.0, C["aux"]), (3, 1.0, C["aux"])):
    e = mu + k * img1
    axb.plot(e[:, 0], e[:, 1], color=col, lw=lw, zorder=2)
axb.fill(*(mu + img1).T, color=C["region"], alpha=0.25, lw=0, zorder=1)
axb.annotate("", xy=mu + A[:, 0], xytext=mu, arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=5)
axb.annotate("", xy=mu + A[:, 1], xytext=mu, arrowprops=dict(color=C["third"], **arrow_kw), zorder=5)
axb.text(*(mu + A[:, 0] + np.array([0.12, -0.38])), r"$s_1 r_1$", color=C["tangent"], fontsize=12)
axb.text(*(mu + A[:, 1] + np.array([-0.12, 0.1])), r"$s_2 r_2$", color=C["third"], fontsize=12, ha="right",
          va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5), zorder=7)
axb.plot(*mu, "o", color=C["main"], ms=3.5, zorder=6)
axb.text(mu[0] + 0.12, mu[1] - 0.45, r"$\mu$", fontsize=12, zorder=7)
axb.plot(*xstar, "o", color=C["accent"], ms=5, zorder=6)
axb.text(xstar[0] - 0.25, xstar[1] + 0.3, r"$x^*$", color=C["accent"], fontsize=12, ha="right")
for k in (1, 2, 3):
    pk = mu - k * A[:, 0] - 0.18 * R[:, 1]
    axb.text(pk[0], pk[1], f"$k={k}$", fontsize=10, color=C["main"] if k == 1 else C["aux"],
             ha="left", va="top")
axb.set_aspect("equal")
axb.set_xlim(-5.6, 7.6)
axb.set_ylim(-4.6, 5.6)
axb.set_xlabel(r"$x^1$")
axb.set_ylabel(r"$x^2$")
axb.set_title(r"(b) $x = \mu + RSy$: $\Sigma = RSS^{\mathsf{T}}R^{\mathsf{T}}$", fontsize=11)

dgfig.map_arrow(fig, axa, axb, r"$\mu + RS$", xy_from=(1.03, 0.62), xy_to=(-0.16, 0.62), rad=-0.35,
                fontsize=12, label_offset=(0.0, 0.03))
for ax in (axa, axb):
    ax.tick_params(labelsize=9)
    for sp_ in ax.spines.values():
        sp_.set_linewidth(0.6)

dgfig.save(fig, __file__)

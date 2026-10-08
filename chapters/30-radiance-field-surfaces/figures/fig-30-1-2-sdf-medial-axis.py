"""Figure 30.1.2: the signed distance function of an ellipse, its gradient field, and the medial axis (Section 30.1).

Ellipse S: x^2/a^2 + y^2/b^2 = 1 with a = 2, b = 1. f = signed distance (negative inside, positive outside), computed
exactly by finding the closest point on S (dense search + Newton on the angle parameter).
(a) color = f (diverging, 0 = white), thin gray = level sets f = +-0.25, +-0.5, ..., black = S (f = 0),
    small vermillion arrows = grad f = the unit vector from the closest point (the normal field extended off S),
    orange segment = medial axis {|x| <= (a^2 - b^2)/a = 1.5, y = 0}; for q = (0.8, 0) the two closest points
    (1.067, +-0.846) are joined to q by dashed gray segments.
(b) |grad_eps f| estimated by central differences with step eps = 0.1 in x and y: exactly 1 away from the medial axis,
    smaller than 1 within eps of it (0 at the center, where the kink is symmetric).
Self-checks: |grad f| = 1 off the medial axis; two equidistant closest points for q; endpoints of the axis = centers of
curvature of the vertices (a - b^2/a = 1.5).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-1-2-sdf-medial-axis.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
A, B = 2.0, 1.0


def closest_param(P, n_init=256, iters=40):
    s0 = np.linspace(0, 2 * np.pi, n_init, endpoint=False)
    E = np.stack([A * np.cos(s0), B * np.sin(s0)], 1)
    s = np.empty(len(P))
    for i in range(0, len(P), 20000):                       # chunks keep memory small
        d2 = ((P[i:i + 20000, None, :] - E[None]) ** 2).sum(-1)
        s[i:i + 20000] = s0[np.argmin(d2, 1)]
    for _ in range(iters):
        c, si = np.cos(s), np.sin(s)
        dx, dy = A * c - P[:, 0], B * si - P[:, 1]
        g1 = -A * si * dx + B * c * dy
        g2 = (A * si) ** 2 + (B * c) ** 2 - A * c * dx - B * si * dy
        s = s - g1 / np.where(np.abs(g2) > 1e-12, g2, 1e-12)
    return s


def sdf_and_grad(P):
    s = closest_param(P)
    Q = np.stack([A * np.cos(s), B * np.sin(s)], 1)
    v = P - Q
    dist = np.linalg.norm(v, axis=1)
    inside = P[:, 0] ** 2 / A ** 2 + P[:, 1] ** 2 / B ** 2 < 1
    sgn = np.where(inside, -1.0, 1.0)
    g = sgn[:, None] * v / np.maximum(dist, 1e-15)[:, None]
    return sgn * dist, g, Q


h = 0.02
xs = np.arange(-3.0, 3.0 + h / 2, h)
ys = np.arange(-2.0, 2.0 + h / 2, h)
X, Y = np.meshgrid(xs, ys)
F, _, _ = sdf_and_grad(np.stack([X.ravel(), Y.ravel()], 1))
F = F.reshape(X.shape)
gy, gx = np.gradient(F, h)
G = np.hypot(gx, gy)
# panel (b): central differences with a finite step eps (as in numerical-gradient SDF training)
EPS = 0.1
P0 = np.stack([X.ravel(), Y.ravel()], 1)
fxp, _, _ = sdf_and_grad(P0 + [EPS, 0])
fxm, _, _ = sdf_and_grad(P0 - [EPS, 0])
fyp, _, _ = sdf_and_grad(P0 + [0, EPS])
fym, _, _ = sdf_and_grad(P0 - [0, EPS])
GE = np.hypot((fxp - fxm) / (2 * EPS), (fyp - fym) / (2 * EPS)).reshape(X.shape)

# ---------------------------------------------------------------- self-checks
c_med = A - B ** 2 / A
assert abs(c_med - (A ** 2 - B ** 2) / A) < 1e-15 and abs(c_med - 1.5) < 1e-15
off = (np.abs(Y) > 0.1) | (np.abs(X) > 1.7)
interior = (slice(1, -1), slice(1, -1))
assert np.max(np.abs(G[interior][off[interior]] - 1)) < 2e-2          # eikonal off the medial axis
on_axis = (np.abs(Y) < h / 2) & (np.abs(X) < 1.4)
# on the axis the y-difference cancels; what is left is |df/dx| along the axis = cos of the half kink angle
for xa in (0.0, 0.8, 1.2):
    k = np.argmin(np.abs(xs - xa))
    print(f"|grad f|_FD on the medial axis at x = {xa}: {G[np.argmin(np.abs(ys)), k]:.3f}")
assert G[np.argmin(np.abs(ys)), np.argmin(np.abs(xs))] < 1e-9 and np.max(G[on_axis]) < 0.9
far = (np.abs(Y) > 2 * EPS) | (np.abs(X) > 1.5 + 2 * EPS)
print(f"max | |grad_eps f| - 1 | away from the band: {np.max(np.abs(GE[far] - 1)):.4f}")
assert np.max(np.abs(GE[far] - 1)) < 0.03                             # O(eps^2) error away from the kink band
print(f"|grad f|_eps at the center (0, 0): {GE[np.argmin(np.abs(ys)), np.argmin(np.abs(xs))]:.3f}")
q = np.array([0.8, 0.0])
_, _, Qu = sdf_and_grad((q + [0, 1e-9])[None])
_, _, Qd = sdf_and_grad((q - [0, 1e-9])[None])
Qu, Qd = Qu[0], Qd[0]
assert abs(np.linalg.norm(Qu - q) - np.linalg.norm(Qd - q)) < 1e-9 and Qu[1] > 0.8 and Qd[1] < -0.8

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.6, 7.3), sharex=True, gridspec_kw=dict(hspace=0.16))
tt = np.linspace(0, 2 * np.pi, 400)

# (a) SDF, gradient field, medial axis
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.6)
pm = axa.pcolormesh(X, Y, F, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
axa.contour(X, Y, F, levels=[-0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.25], colors=C["aux"], linewidths=0.7)
axa.plot(A * np.cos(tt), B * np.sin(tt), color=C["main"], lw=2.2, zorder=4)
qx, qy = np.meshgrid(np.arange(-2.75, 2.8, 0.5), np.arange(-1.75, 1.8, 0.5))
Pq = np.stack([qx.ravel(), qy.ravel()], 1)
Pq = Pq[(np.abs(Pq[:, 1]) > 0.2) | (np.abs(Pq[:, 0]) > 1.7)]
_, gq, _ = sdf_and_grad(Pq)
axa.quiver(Pq[:, 0], Pq[:, 1], gq[:, 0], gq[:, 1], color=C["normal"], scale=14, width=0.006, headwidth=4,
           zorder=5)
axa.plot([-c_med, c_med], [0, 0], color=C["accent"], lw=4.0, solid_capstyle="butt", zorder=6)
axa.plot([q[0], Qu[0]], [q[1], Qu[1]], ls="--", color=C["main"], lw=1.1, zorder=7)
axa.plot([q[0], Qd[0]], [q[1], Qd[1]], ls="--", color=C["main"], lw=1.1, zorder=7)
axa.plot(*q, "o", ms=6, color=C["main"], zorder=8)
axa.plot([Qu[0], Qd[0]], [Qu[1], Qd[1]], "o", ms=5, color=C["main"], mfc="white", zorder=8)
axa.text(q[0] - 0.08, q[1] - 0.12, r"$q$", fontsize=13, ha="right", va="top", zorder=9)
axa.annotate("medial axis", xy=(-0.9, 0.0), xytext=(-1.55, 0.62), fontsize=12, color=C["main"],
             arrowprops=dict(arrowstyle="->", lw=1.0, color=C["main"]), ha="center",
             bbox=dict(fc="white", ec="none", pad=1.0), zorder=9)
axa.text(2.15, 1.65, r"$S$: $f = 0$", fontsize=12.5, ha="center", bbox=dict(fc="white", ec="none", pad=1.0))
axa.text(-2.55, -1.8, r"$\nabla f$", color=C["normal"], fontsize=13, ha="center", va="center",
         bbox=dict(fc="white", ec="none", pad=1.0))
axa.set_aspect("equal")
axa.set_ylim(-2, 2)
axa.set_title(r"(a) signed distance $f$ and its gradient", fontsize=13.5)
cb = fig.colorbar(pm, ax=axa, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1, 1.5])
cb.set_label(r"$f$", fontsize=13)

# (b) |grad f| from central differences
im = axb.pcolormesh(X, Y, GE, cmap="Greys_r", vmin=0, vmax=1.05, shading="auto", rasterized=True)
axb.plot(A * np.cos(tt), B * np.sin(tt), color=C["tangent"], lw=1.4, zorder=4)
axb.annotate(r"$|\nabla_\varepsilon f| < 1$ near the medial axis", xy=(0.4, -0.06), xytext=(0.3, -1.45), fontsize=12,
             ha="center", arrowprops=dict(arrowstyle="->", lw=1.0, color=C["main"]),
             bbox=dict(fc="white", ec="none", pad=1.0))
axb.text(-2.1, 1.55, r"$|\nabla_\varepsilon f| \approx 1$", fontsize=12.5, ha="center", bbox=dict(fc="white", ec="none", pad=1.0))
axb.set_aspect("equal")
axb.set_ylim(-2, 2)
axb.set_xlabel(r"$x$")
axa.set_ylabel(r"$y$")
axb.set_ylabel(r"$y$")
axb.set_title(r"(b) central differences, step $\varepsilon = 0.1$", fontsize=13.5)
cb2 = fig.colorbar(im, ax=axb, fraction=0.04, pad=0.02, ticks=[0, 0.5, 1])
cb2.set_label(r"$|\nabla_\varepsilon f|$", fontsize=13)

dgfig.save(fig, __file__)

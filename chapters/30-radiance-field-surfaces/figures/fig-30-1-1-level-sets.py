"""Figure 30.1.1: when is a zero level set a regular curve/surface? (Section 30.1)

Four 2D implicit functions on the same square [-2, 2]^2 (2D cross-sections of the 3D situation):
(a) f = x^2 + y^2 - 1: 0 is a regular value; vermillion arrows = N = grad f/|grad f| at 8 points of S.
(b) f = x^2 - y^2 (cross-section of the cone x^2 + y^2 - z^2): the zero set is two crossing lines; grad f = 0 at 0.
(c) f = ((x-1)^2 + y^2 - 1)((x+1)^2 + y^2 - 1): two circles touching at the origin; grad f = 0 at the contact.
(d) g = (x^2 + y^2 - 1)^2: the zero set is a perfectly good circle but grad g = 2 f grad f = 0 on it;
    blue arrows = grad g (unnormalized, scaled by 0.18) on the circles r = 0.8 and r = 1.2.
Gray curves: other level sets f = c. Black: the zero set. Orange dot: critical point on the zero set.
Self-checks: grad f != 0 on the circle in (a); grad f = 0 at the marked points; |grad g| = 0 on the circle in (d).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-1-1-level-sets.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS

L = 2.0
xs = np.linspace(-L, L, 801)
X, Y = np.meshgrid(xs, xs)


def f_a(x, y):
    return x ** 2 + y ** 2 - 1


def g_a(x, y):
    return np.stack([2 * x, 2 * y], -1)


def f_b(x, y):
    return x ** 2 - y ** 2


def g_b(x, y):
    return np.stack([2 * x, -2 * y], -1)


def f_c(x, y):
    return ((x - 1) ** 2 + y ** 2 - 1) * ((x + 1) ** 2 + y ** 2 - 1)


def g_c(x, y):
    p, q = (x - 1) ** 2 + y ** 2 - 1, (x + 1) ** 2 + y ** 2 - 1
    return np.stack([2 * (x - 1) * q + 2 * (x + 1) * p, 2 * y * q + 2 * y * p], -1)


def f_d(x, y):
    return (x ** 2 + y ** 2 - 1) ** 2


def g_d(x, y):
    return 2 * (x ** 2 + y ** 2 - 1)[..., None] * g_a(x, y)


# ---------------------------------------------------------------- self-checks
ang = np.linspace(0, 2 * np.pi, 400)
cx, cy = np.cos(ang), np.sin(ang)
assert np.min(np.linalg.norm(g_a(cx, cy), axis=-1)) > 1.99           # (a) regular on S
assert np.allclose(g_b(0.0, 0.0), 0) and f_b(0.0, 0.0) == 0           # (b) critical point on S
assert np.allclose(g_c(0.0, 0.0), 0) and f_c(0.0, 0.0) == 0           # (c) contact point is critical
assert np.allclose(g_d(cx, cy), 0, atol=1e-12)                         # (d) grad vanishes on S

fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.9), gridspec_kw=dict(wspace=0.08, hspace=0.22))
arrow_kw = dict(arrowstyle="-|>", lw=1.5, mutation_scale=12, shrinkA=0, shrinkB=0)


def base(ax, F, levels, title):
    ax.contour(X, Y, F, levels=[lv for lv in levels if lv != 0], colors=C["aux"], linewidths=0.8, zorder=1)
    ax.contour(X, Y, F, levels=[0.0], colors=C["main"], linewidths=2.2, zorder=3)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(length=2)
    ax.set_title(title, fontsize=13.5, pad=4)


# (a) regular value: unit normals
ax = axs[0, 0]
base(ax, f_a(X, Y), [-0.75, -0.4, 0.5, 1.5, 3.0], r"(a) $f = x^2 + y^2 - 1$")
for k in range(8):
    a = 2 * np.pi * k / 8 + np.pi / 8
    p = np.array([np.cos(a), np.sin(a)])
    n = g_a(*p)
    n = n / np.linalg.norm(n)
    ax.annotate("", xy=p + 0.55 * n, xytext=p, arrowprops=dict(color=C["normal"], **arrow_kw), zorder=5)
ax.text(0.0, 0.0, "0 is a\nregular value", ha="center", va="center", fontsize=11.5,
        bbox=dict(fc="white", ec="none", pad=1.0), zorder=4)
ax.text(0.0, -1.8, r"$\mathbf{N} = \nabla f/|\nabla f|$", color=C["normal"], fontsize=12, ha="center", va="center")

# (b) crossing lines (cone cross-section)
ax = axs[0, 1]
base(ax, f_b(X, Y), [-2, -0.8, -0.2, 0.2, 0.8, 2], r"(b) $f = x^2 - y^2$")
ax.plot([0], [0], "o", color=C["accent"], ms=8, zorder=6, mec=C["main"], mew=0.6)
ax.annotate(r"$\nabla f = 0$ on $f = 0$", xy=(0, 0), xytext=(0.15, 1.55), fontsize=11.5,
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]), ha="center",
            bbox=dict(fc="white", ec="none", pad=1.0))

# (c) tangent circles
ax = axs[1, 0]
base(ax, f_c(X, Y), [-0.6, -0.25, 0.4, 2.0], "(c) two touching circles")
ax.plot([0], [0], "o", color=C["accent"], ms=8, zorder=6, mec=C["main"], mew=0.6)
ax.annotate(r"$\nabla f = 0$", xy=(0, 0), xytext=(0.0, 1.55), fontsize=11.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]), bbox=dict(fc="white", ec="none", pad=1.0))

# (d) squared function: grad vanishes on S
ax = axs[1, 1]
base(ax, f_d(X, Y), [0.1, 0.5, 2.0], r"(d) $g = (x^2 + y^2 - 1)^2$")
ax.plot(np.cos(ang), np.sin(ang), color=C["main"], lw=2.2, zorder=3)      # the zero set (contour misses a double zero)
for rr_ in (0.8, 1.2):
    for k in range(8):
        a = 2 * np.pi * k / 8
        p = rr_ * np.array([np.cos(a), np.sin(a)])
        gv = 0.18 * g_d(*p)
        ax.annotate("", xy=p + gv, xytext=p, arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=5)
a_lab = 2 * np.pi * 5 / 16                                              # between two arrow directions
ax.annotate(r"$\nabla g = 0$" + "\non " + r"$g = 0$", xy=(np.cos(a_lab), np.sin(a_lab)), xytext=(-1.93, 1.93),
            ha="left", va="top", fontsize=11.5, zorder=6, bbox=dict(fc="white", ec="none", pad=1.0),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]))
ax.text(0.0, -0.02, "inward", ha="center", va="center", fontsize=10.5, color=C["tangent"])
ax.text(0.0, -1.78, "outward", ha="center", va="center", fontsize=10.5, color=C["tangent"])
ax.text(1.35, 1.72, r"$0.18\,\nabla g$", color=C["tangent"], fontsize=11.5, ha="center")

for ax in axs[:, 1]:
    ax.set_yticklabels([])
for ax in axs[1, :]:
    ax.set_xlabel(r"$x$", labelpad=1)
for ax in axs[:, 0]:
    ax.set_ylabel(r"$y$", labelpad=1, rotation=0)
for ax in axs[0, :]:
    ax.set_xticklabels([])

dgfig.save(fig, __file__)

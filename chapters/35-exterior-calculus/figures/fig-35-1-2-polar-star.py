"""Figure 35.1.2: the 1-form dtheta on R^2 minus the origin and its Hodge star star dtheta = -dr / r
(Section 35.1, Example 35.1.9).

Purple solid rays: the level lines theta = 2 pi k / 24 of the 1-form dtheta = (-y dx + x dy) / (x^2 + y^2) (each
ray is where a local angle function takes one of these values; the increment is h = 2 pi / 24).
Purple dashed circles: radii r_j = exp(h j), j = -4, ..., 3. The covector star dtheta = -dr / r kills the tangential
direction and has length 1 / r, so its stack lines are tangent to circles with spacing h r, which these circles
realize (log r = h j); star dtheta increases inward.
At the point p (r = 1.5, theta = 37.5 deg): blue (dtheta)^sharp = (-sin theta, cos theta) / r (counterclockwise,
length 1/r) and green (star dtheta)^sharp = -(cos theta, sin theta) / r (inward, length 1/r), drawn at their true
length; the cell containing p is shaded (accent): its radial side r_j (e^h - 1) and arc side r_j h differ by the factor
(e^h - 1)/h = 1.14, so the cells are close to squares.
Self-checks: star dtheta = -dr / r at random points; the two sharps are orthogonal, of equal length 1/r, and the
pair ((dtheta)^sharp, (star dtheta)^sharp) is positively oriented (a quarter turn counterclockwise).
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
h = 2 * np.pi / 24


def dtheta(p):
    x, y = p
    return np.array([-y, x]) / (x * x + y * y)


def dr(p):
    return np.asarray(p, float) / np.hypot(*p)


def star1(a):
    return np.array([-a[1], a[0]])


rng = np.random.default_rng(3512)
for p in rng.uniform(-2, 2, size=(200, 2)):
    if np.hypot(*p) < 0.1:
        continue
    r = np.hypot(*p)
    assert np.allclose(star1(dtheta(p)), -dr(p) / r)
r0, th0 = 1.5, np.deg2rad(37.5)
p = r0 * np.array([np.cos(th0), np.sin(th0)])
a, b = dtheta(p), star1(dtheta(p))
assert abs(a @ b) < 1e-14 and abs(np.linalg.norm(a) - 1 / r0) < 1e-14 and abs(np.linalg.norm(b) - 1 / r0) < 1e-14
assert a[0] * b[1] - a[1] * b[0] > 0
assert abs((np.exp(h) - 1) / h - 1.14) < 0.005

fig, ax = plt.subplots(figsize=(4.6, 4.6))
radii = np.exp(h * np.arange(-4, 4))
rmin, rmax = radii[0], radii[-1]
for k in range(24):
    t = h * k
    ax.plot([rmin * np.cos(t), rmax * np.cos(t)], [rmin * np.sin(t), rmax * np.sin(t)], color=C["covector"], lw=0.9,
            zorder=2)
ts = np.linspace(0, 2 * np.pi, 400)
for rj in radii:
    ax.plot(rj * np.cos(ts), rj * np.sin(ts), color=C["covector"], lw=0.9, ls=(0, (4, 2.5)), zorder=2)
# one shaded cell (the one containing p): between rays k = 2, 3 and circles j = 1, 2
j = 1
cell_r = (np.exp(h * j), np.exp(h * (j + 1)))
tt = np.linspace(h * 2, h * 3, 20)
poly = np.concatenate([np.stack([cell_r[0] * np.cos(tt), cell_r[0] * np.sin(tt)], 1),
                       np.stack([cell_r[1] * np.cos(tt[::-1]), cell_r[1] * np.sin(tt[::-1])], 1)])
ax.add_patch(Polygon(poly, closed=True, facecolor=C["accent"], alpha=0.5, edgecolor="none", zorder=1))
ax.plot([0], [0], "o", color=C["main"], ms=3, zorder=6)


def arrow(p, v, color, lw=1.6):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + v), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=8))


arrow(p, a, C["tangent"])
arrow(p, b, C["third"])
ax.plot(*p, "o", color=C["main"], ms=3.5, zorder=9)
ax.text(p[0] + a[0] - 0.22, p[1] + a[1] + 0.08, r"$(d\theta)^\sharp$", color=C["tangent"], fontsize=12,
        ha="center", zorder=9, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
ax.text(p[0] + b[0] + 0.05, p[1] + b[1] - 0.32, r"$(\star d\theta)^\sharp$", color=C["third"],
        fontsize=12, ha="center", zorder=9, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
ax.text(p[0] + 0.12, p[1] - 0.12, r"$p$", fontsize=12, zorder=9)
# legend
ax.plot([-2.25, -1.95], [-2.3, -2.3], color=C["covector"], lw=1.2)
ax.text(-1.88, -2.3, r"$d\theta$: rays $\theta = $ const", va="center", fontsize=10.5)
ax.plot([-2.25, -1.95], [-2.58, -2.58], color=C["covector"], lw=1.2, ls=(0, (4, 2.5)))
ax.text(-1.88, -2.58, r"$\star d\theta = -dr/r$: circles $\log r = $ const", va="center", fontsize=10.5)
dgfig.schematic_axes(ax, xlim=(-2.35, 2.35), ylim=(-2.75, 2.3))
dgfig.save(fig, __file__)

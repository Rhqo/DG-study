"""Figure 31.5.5: why point-sampled visibility has no gradient and how analytic antialiasing gives one (Section 31.5).

A row of pixels with centres at x = 0, 1, 2, 3 (pixel k covers [k - 0.5, k + 0.5]). A surface of colour 1 covers
x < p (a vertical silhouette edge at x = p), the background has colour 0.
  point sampling: pixel k shows 1 if its centre is covered (k < p), else 0;
  analytic antialiasing (the rule described for nvdiffrast [Laine20]): for each pair of horizontally adjacent pixels
    whose samples differ, the edge crosses the segment between the two centres at p; the pixel that contains the
    crossing receives the colour of the other pixel with a blend factor that is linear in the crossing position,
    0 at the midpoint and 1/2 at that pixel's centre;
  exact coverage (box filter): pixel k shows the covered fraction clamp(p - (k - 0.5), 0, 1).
(a) The row of pixels for p = 1.3 (edge inside pixel 1): top point-sampled colours, bottom antialiased colours.
(b) The colour of pixel 1 as a function of p: point-sampled (step), antialiased (ramp), exact (dashed).
Self-check: for a vertical edge the antialiased colour equals the exact coverage (as stated in the paper for vertical
and horizontal edges); its derivative with respect to p is 1 on (0.5, 1.5) and 0 elsewhere, while the
point-sampled derivative is 0 except at p = 1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-5-5-silhouette-gradient.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

C = dgfig.COLORS
K = 4


def point_sampled(p):
    return np.array([1.0 if k < p else 0.0 for k in range(K)])


def antialiased(p):
    c = point_sampled(p)
    out = c.copy()
    for k in range(K - 1):               # pair (k, k + 1)
        if c[k] != c[k + 1] and k < p < k + 1:
            mid = k + 0.5
            if p > mid:                  # crossing inside the right pixel: left colour blends into it
                w = p - mid
                out[k + 1] = (1 - w) * c[k + 1] + w * c[k]
            else:                        # crossing inside the left pixel
                w = mid - p
                out[k] = (1 - w) * c[k] + w * c[k + 1]
    return out


def exact(p):
    return np.array([np.clip(p - (k - 0.5), 0, 1) for k in range(K)])


ps = np.linspace(0.05, 2.95, 2901) + 1e-4   # avoid the measure-zero ties at pixel centres
for p in ps:
    if abs(p - round(p)) > 1e-9:     # an edge exactly through a pixel centre is a measure-zero tie
        assert np.allclose(antialiased(p), exact(p), atol=1e-12)

fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.3))
p0 = 1.3
for row, (name, vals) in enumerate((("point-sampled", point_sampled(p0)), ("antialiased", antialiased(p0)))):
    y0 = 1.25 - row * 1.35
    for k in range(K):
        g = 1 - vals[k]
        ax.add_patch(Rectangle((k - 0.5, y0 - 0.5), 1, 1, fc=(g, g, g), ec=C["aux"], lw=0.8))
        ax.text(k, y0, f"{vals[k]:.1f}", ha="center", va="center", fontsize=11,
                color="white" if vals[k] > 0.5 else "black")
    ax.text(-0.62, y0, name, ha="right", va="center", fontsize=10.5)
for y0 in (1.25, -0.1):
    ax.plot([p0, p0], [y0 - 0.62, y0 + 0.62], color=C["normal"], lw=2.0)
ax.text(p0, 1.95, r"edge at $x = p = 1.3$", ha="center", fontsize=10.5, color=C["normal"])
for k in range(K):
    ax.plot(k, -0.85, "o", color=C["main"], ms=3)
    ax.text(k, -1.05, f"pixel {k}", ha="center", va="top", fontsize=9.5)
ax.set_xlim(-2.4, 3.6)
ax.set_ylim(-1.35, 2.1)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a) one row of pixels; surface (colour 1,\nblack) on the left of the edge", fontsize=11, pad=16)

axb.plot(ps, [point_sampled(p)[1] for p in ps], color=C["tangent"], lw=2.0, label="point-sampled")
axb.plot(ps, [antialiased(p)[1] for p in ps], color=C["accent"], lw=2.4, label="antialiased")
axb.plot(ps, [exact(p)[1] for p in ps], color=C["main"], lw=1.0, ls=(0, (4, 3)), label="exact coverage")
axb.axvspan(0.5, 1.5, color=C["region"], alpha=0.2, lw=0)
axb.text(1.1, 0.1, r"$\partial c_1/\partial p = 1$", ha="left", fontsize=10.5)
axb.set_xlabel(r"edge position $p$")
axb.set_ylabel(r"colour of pixel 1, $c_1$")
axb.legend(fontsize=9.5, frameon=False, loc="center right")
axb.set_ylim(-0.08, 1.12)
axb.grid(True, lw=0.3, alpha=0.5)
axb.set_title(r"(b) $c_1$ as a function of $p$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.15)
dgfig.save(fig, __file__)

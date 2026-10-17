"""Figure 32.1.1: inscribed regular n-gons. The perimeter converges to 2 pi r at order 2, the sum of the exterior
angles is exactly 2 pi for every n (Proposition 32.1.1).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-1-1-polygon-convergence.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

import dgdec as dd  # noqa: E402

C = dgfig.COLORS
r = 1.0


def ngon(n, r=1.0):
    a = 2 * np.pi * np.arange(n) / n
    return r * np.stack([np.cos(a), np.sin(a)], 1)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) circle, hexagon and 12-gon, exterior angle of the hexagon at one vertex
s = np.linspace(0, 2 * np.pi, 400)
ax.plot(r * np.cos(s), r * np.sin(s), color=C["aux"], lw=dgfig.LW["aux"], ls="--")
P12 = ngon(12)
ax.fill(P12[:, 0], P12[:, 1], fc="none", ec=C["tangent"], lw=1.2)
P6 = ngon(6)
ax.fill(P6[:, 0], P6[:, 1], fc="none", ec=C["main"], lw=dgfig.LW["main"])
ax.plot(P6[:, 0], P6[:, 1], "o", color=C["main"], ms=3.5)
# exterior angle at vertex 0 of the hexagon: extend edge 5->0 beyond vertex 0
v0, v1, v2 = P6[5], P6[0], P6[1]
d_in = (v1 - v0) / np.linalg.norm(v1 - v0)
d_out = (v2 - v1) / np.linalg.norm(v2 - v1)
ax.plot([v1[0], v1[0] + 0.62 * d_in[0]], [v1[1], v1[1] + 0.62 * d_in[1]], color=C["aux"], lw=dgfig.LW["aux"], ls=":")
a_in = np.degrees(np.arctan2(d_in[1], d_in[0]))
a_out = np.degrees(np.arctan2(d_out[1], d_out[0]))
ext = (a_out - a_in) % 360
assert abs(ext - 60) < 1e-9, ext  # exterior angle of the hexagon = 2 pi / 6
ax.add_patch(Arc(v1, 0.8, 0.8, theta1=a_in, theta2=a_out, color=C["accent"], lw=1.6))
ax.text(v1[0] + 0.24, v1[1] + 0.10, r"$2\pi/6$", color=C["accent"],
        fontsize=11, ha="left", va="bottom")
ax.text(0.0, 0.0, r"$r = 1$", ha="center", va="center", fontsize=11, color=C["aux"])
ax.text(-0.98, -1.12, "n = 6", fontsize=10, color=C["main"])
ax.text(0.45, -1.12, "n = 12", fontsize=10, color=C["tangent"])
ax.set_aspect("equal")
ax.set_xlim(-1.25, 1.5)
ax.set_ylim(-1.25, 1.35)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

# (b) L_n / (2 pi r) and (sum of exterior angles) / (2 pi) against n
ns = np.array([3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128])
Ln = np.array([dd.curve_length(ngon(n, r), True) for n in ns])
ext_sum = np.array([dd.turning_angles(ngon(n, r), True).sum() for n in ns])
assert np.allclose(Ln, 2 * ns * r * np.sin(np.pi / ns), atol=1e-12)
assert np.allclose(ext_sum, 2 * np.pi, atol=1e-12)
err = 2 * np.pi * r - Ln
assert np.all(err > 0) and np.all(err <= np.pi ** 3 * r / (3 * ns ** 2) + 1e-15)
bx.semilogx(ns, ext_sum / (2 * np.pi), "s-", color=C["accent"], ms=4, lw=1.4, label="exterior angle sum / $2\\pi$")
bx.semilogx(ns, Ln / (2 * np.pi * r), "o-", color=C["main"], ms=4, lw=1.4, label="perimeter / $2\\pi r$")
bx.axhline(1.0, color=C["aux"], lw=dgfig.LW["aux"], ls="--", zorder=0)
for n in (6, 12):
    k = list(ns).index(n)
    bx.annotate(f"n = {n}", (n, Ln[k] / (2 * np.pi)), xytext=(n * 1.25, Ln[k] / (2 * np.pi) - 0.045),
                fontsize=10, arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
bx.set_xlabel("number of vertices $n$")
bx.set_ylim(0.8, 1.04)
bx.set_xticks([3, 6, 12, 24, 48, 96])
bx.set_xticklabels(["3", "6", "12", "24", "48", "96"])
bx.legend(loc="lower right", frameon=False, fontsize=9.5)
bx.set_title("(b)", fontsize=12)
for sp_ in ("top", "right"):
    bx.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)

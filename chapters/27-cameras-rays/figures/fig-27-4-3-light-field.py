"""Figure 27.4.3: in free space the radiance is constant along a line; NeRF's 5D field is needed inside matter.

2D scene (the plane of the line): disc A, center (2, 0.3), radius 0.6, emitted value c = 0.9; disc B, center (4.5, -0.2),
radius 0.8, c = 0.3; density sigma = 25 inside (logistic edge of width 0.02), 0 outside. The line y = 0.1, parametrized by
s = x. L(s) is the radiance at (s, 0.1) travelling in the direction d = (-1, 0), i.e. what a camera at (s, 0.1) looking
toward +x records; it is computed with the NeRF quadrature (alpha compositing, 6000 samples over 8 units).
(a) The scene: discs (gray, darker = denser), the line (black), the light direction d (arrow), three camera positions
    s = 1, 3.2, 6.5 looking toward +x.
(b) L(s) (black) and the density sigma along the line (gray area, right axis). Light-blue bands: free segments.
Self-checks: L is constant (to 1e-3) on each free segment: 0.9 before A, 0.3 between A and B, 0 after B.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-4-3-light-field.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
DISCS = ((2.0, 0.3, 0.6, 0.9), (4.5, -0.2, 0.8, 0.3))
Y0 = 0.1


def sigma_color(x):
    out_s = np.zeros(len(x))
    out_c = np.zeros(len(x))
    for (cx, cy, r, col) in DISCS:
        dd = np.hypot(x[:, 0] - cx, x[:, 1] - cy)
        w = 1 / (1 + np.exp((dd - r) / 0.02))
        out_s += 25 * w
        out_c += col * 25 * w
    return out_s, np.where(out_s > 1e-9, out_c / np.maximum(out_s, 1e-12), 0.0)


def radiance(x0, dvec, n=6000, umax=8.0):
    u = np.linspace(0, umax, n)
    du = u[1] - u[0]
    pts = x0[None, :] - u[:, None] * dvec[None, :]
    sg, cl = sigma_color(pts)
    alpha = 1 - np.exp(-sg * du)
    Tr = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]])
    return np.sum(Tr * alpha * cl)


dvec = np.array([-1.0, 0.0])
ss = np.linspace(0, 7.5, 301)
L = np.array([radiance(np.array([s, Y0]), dvec) for s in ss])
sig, _ = sigma_color(np.column_stack([ss, np.full_like(ss, Y0)]))
# free segments along the line
xa = [2.0 - np.sqrt(0.6**2 - 0.2**2), 2.0 + np.sqrt(0.6**2 - 0.2**2)]
xb = [4.5 - np.sqrt(0.8**2 - 0.3**2), 4.5 + np.sqrt(0.8**2 - 0.3**2)]
free = [(0, xa[0] - 0.2), (xa[1] + 0.2, xb[0] - 0.2), (xb[1] + 0.2, 7.5)]
vals = []
for lo, hi in free:
    msk = (ss > lo) & (ss < hi)
    assert np.ptp(L[msk]) < 1e-3, (lo, hi, np.ptp(L[msk]))
    vals.append(L[msk].mean())
assert abs(vals[0] - 0.9) < 2e-3 and abs(vals[1] - 0.3) < 2e-3 and abs(vals[2]) < 1e-6

fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.6, 6.4), gridspec_kw=dict(height_ratios=[1.0, 1.1]))
for (cx, cy, r, col) in DISCS:
    ax.add_patch(Circle((cx, cy), r, facecolor=C["surface"], edgecolor="k", lw=1.0))
    ax.text(cx, cy - 0.12, rf"$c={col}$", fontsize=11, ha="center")
ax.text(2.0, 0.95, "A", fontsize=12, ha="center")
ax.text(4.5, 0.65, "B", fontsize=12, ha="center")
ax.plot([0, 7.5], [Y0, Y0], color="k", lw=1.4)
ax.annotate("", xy=(6.3, -0.75), xytext=(7.3, -0.75), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.6))
ax.text(6.35, -0.68, "light direction $d$", fontsize=11, color=C["normal"], va="bottom")
ax.text(0.05, -0.75, "blue: cameras looking toward $+x$", fontsize=10.5, color=C["tangent"], va="center")
for s in (1.0, 3.2, 6.5):
    ax.plot(s, Y0, "s", color=C["tangent"], ms=8)
    ax.annotate("", xy=(s + 0.45, Y0), xytext=(s, Y0), arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4))
    ax.text(s - 0.08, Y0 - 0.12, rf"$s={s:g}$", fontsize=11, color=C["tangent"], ha="center", va="top")
ax.set_xlim(0, 7.5)
ax.set_ylim(-1.1, 1.15)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$ (= $s$ on the line)")
ax.set_yticks([])
ax.set_title("(a) a line through a scene with two objects", fontsize=12)

for lo, hi in free:
    bx.axvspan(lo, hi, color=C["region"], alpha=0.25, lw=0)
b2 = bx.twinx()
b2.fill_between(ss, 0, sig, color=C["surface"], alpha=0.8, lw=0)
b2.set_ylim(0, 60)
b2.set_ylabel(r"density $\sigma$ (gray)", fontsize=11)
bx.set_zorder(b2.get_zorder() + 1)
bx.patch.set_visible(False)
bx.plot(ss, L, color="k", lw=2.0)
for (lo, hi), v in zip(free, vals):
    bx.text((lo + hi) / 2, v + 0.06, rf"$L={v:.1f}$", fontsize=11, ha="center")
bx.text(0.15, 0.55, "free space:\n$L$ constant", fontsize=10.5, color=C["tangent"])
bx.set_xlim(0, 7.5)
bx.set_ylim(-0.05, 1.1)
bx.set_xlabel(r"position $s$ along the line")
bx.set_ylabel(r"radiance $L(s)$ along $d$")
bx.set_title(r"(b) $L$ changes only where $\sigma>0$", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
print("free values", vals)

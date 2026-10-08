"""Figure 27.2.2: points transform by H, lines by H^{-T} (Section 27.2).

Homography H = [[1.2, 0.3, -0.4], [0.1, 0.8, 0.5], [0.3, -0.2, 1.0]] acting on RP^2 (chart w = 1).
(a) Before: the line l through p1 = (0.2, 0.1) and p2 = (1.1, 0.7) (so l = p1 x p2), the point p3 = 0.3 p1 + 0.7 p2 on it,
    and a second line l' = (1, -2, 0.5) meeting l at q = l x l'.
(b) After: the points H p_i (black), the line H^{-T} l (blue, solid) through them, the line H^{-T} l' (green) and their
    meet H q (orange). Dashed vermillion: the line whose coefficient vector is H l, i.e. what one gets by transforming the
    line like a point; it misses the points.
Self-checks: incidence is preserved by H^{-T}; the dashed line misses all three points; H q = (H^-T l) x (H^-T l') up to scale.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-2-2-point-line.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
H = np.array([[1.2, 0.3, -0.4], [0.1, 0.8, 0.5], [0.3, -0.2, 1.0]])
Hit = np.linalg.inv(H).T
p1, p2 = np.array([0.2, 0.1, 1.0]), np.array([1.1, 0.7, 1.0])
p3 = 0.3 * p1 + 0.7 * p2
l = np.cross(p1, p2)
l2 = np.array([1.0, -2.0, 0.5])
q = np.cross(l, l2)
pts = [p1, p2, p3]
lH, l2H, wrong = Hit @ l, Hit @ l2, H @ l
assert max(abs(lH @ (H @ p)) for p in pts) < 1e-12
assert min(abs(wrong @ (H @ p)) / np.linalg.norm(wrong[:2]) for p in pts) > 0.05
qH = H @ q
assert np.linalg.norm(np.cross(qH, np.cross(lH, l2H))) < 1e-10 * np.linalg.norm(qH) * np.linalg.norm(np.cross(lH, l2H))


def dh(p):
    return p[:2] / p[2]


def draw_line(ax, lv, xlim, **kw):
    xs = np.linspace(*xlim, 200)
    if abs(lv[1]) > 1e-9:
        ax.plot(xs, -(lv[0] * xs + lv[2]) / lv[1], **kw)
    else:
        ax.axvline(-lv[2] / lv[0], **kw)


XL, YL = (-0.6, 1.8), (-0.4, 1.6)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.4))
draw_line(ax, l, XL, color=C["tangent"], lw=1.8)
draw_line(ax, l2, XL, color=C["third"], lw=1.4)
for i, p in enumerate(pts):
    ax.plot(*dh(p), "ko", ms=6, zorder=5)
    ax.text(dh(p)[0] + 0.04, dh(p)[1] - 0.12, rf"$p_{i+1}$", fontsize=12)
ax.plot(*dh(q), "o", color=C["accent"], ms=7, zorder=6)
ax.text(dh(q)[0] - 0.2, dh(q)[1] + 0.08, r"$q$", fontsize=12, color=C["accent"])
ax.text(1.45, 0.72, r"$l$", fontsize=13, color=C["tangent"])          # below the blue line
ax.text(0.25, 0.50, r"$l'$", fontsize=13, color=C["third"])          # above the green line

draw_line(bx, lH, XL, color=C["tangent"], lw=1.8)
draw_line(bx, l2H, XL, color=C["third"], lw=1.4)
draw_line(bx, wrong, XL, color=C["normal"], lw=1.4, ls="--")
for i, p in enumerate(pts):
    bx.plot(*dh(H @ p), "ko", ms=6, zorder=5)
    bx.text(dh(H @ p)[0] + 0.04, dh(H @ p)[1] - 0.13, rf"$Hp_{i+1}$", fontsize=12)
bx.plot(*dh(qH), "o", color=C["accent"], ms=7, zorder=6)
bx.text(dh(qH)[0] - 0.12, dh(qH)[1] + 0.08, r"$Hq$", fontsize=12, color=C["accent"])
bx.text(-0.55, 1.42, r"blue: $H^{-\mathsf{T}}l$ (correct)", fontsize=11, color=C["tangent"])
bx.text(-0.55, 1.27, r"green: $H^{-\mathsf{T}}l'$", fontsize=11, color=C["third"])
bx.text(-0.55, 1.12, r"dashed: $Hl$ (wrong)", fontsize=11, color=C["normal"])
for a_, ttl in ((ax, r"(a) points $p_i$ on the line $l$"), (bx, r"(b) after $p\mapsto Hp$")):
    a_.set_xlim(*XL)
    a_.set_ylim(*YL)
    a_.set_aspect("equal")
    a_.set_xlabel(r"$x/w$")
    a_.set_ylabel(r"$y/w$")
    a_.set_title(ttl, fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)

"""Figure 27.2.4: NeRF's NDC map is a projective transformation (Section 27.2).

Camera looking along -z (NeRF / OpenGL convention), image height H = 378 px, focal f = 400 px, near plane n = 1, far
plane at infinity. Shown in the (|z|, y) plane (x = 0).
(a) Camera space: rays from the center through 7 pixel rows (gray fan, frustum edges black), each ray starting at the near
    plane |z| = 1 as NeRF does. Dots: samples at s' = 0, 0.1, ..., 0.9 (uniform in NDC; the paper's t'). Orange: a line segment that does not
    pass through the center, from (|z|, y) = (1.5, -0.3) to (12, 2.2).
(b) NDC: y' = a_y y / z, z' = 1 + 2n/z with a_y = -f/(H/2). The same rays, samples and orange segment.
Self-checks: the image of every ray is a horizontal line (y' constant); the image of the orange segment is a straight
segment; the samples have z' = 2s' - 1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-2-4-ndc.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
Hpx, f, n = 378.0, 400.0, 1.0
ay = -f / (Hpx / 2)
half = (Hpx / 2) / f                     # frustum slope |y|/|z|


def ndc(y, z):                           # z < 0
    return ay * y / z, 1 + 2 * n / z


rows = np.linspace(-1, 1, 7) * half
tps = np.arange(0, 1.0, 0.1)
seg0, seg1 = np.array([1.5, -0.3]), np.array([12.0, 2.2])     # (|z|, y)
ss = np.linspace(0, 1, 300)
seg = seg0[None, :] + ss[:, None] * (seg1 - seg0)[None, :]
yp, zp = ndc(seg[:, 1], -seg[:, 0])
A = np.column_stack([zp - zp.mean(), yp - yp.mean()])
assert np.linalg.svd(A, compute_uv=False)[1] < 1e-10
for k in rows:
    zz = -np.linspace(1, 50, 40)
    y1, _ = ndc(k * np.abs(zz), zz)
    assert np.ptp(y1) < 1e-12

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw=dict(width_ratios=[1.35, 1.0]))
ZMAX = 14
for k in rows:
    edge = abs(abs(k) - half) < 1e-12
    ax.plot([0, ZMAX], [0, k * ZMAX], color="k" if edge else C["aux"], lw=1.2 if edge else 0.7)
    zs = n / (1 - tps)                    # |z| of the samples (s' = 1 - n/|z|)
    ax.plot(zs[zs < ZMAX], k * zs[zs < ZMAX], "o", color=C["tangent"], ms=3.5)
    yv, zv = ndc(k * zs, -zs)
    assert np.allclose(zv, 2 * tps - 1)
    bx.plot([-1, 1], [ay * (k * 1) / -1] * 2, color="k" if edge else C["aux"], lw=1.2 if edge else 0.7)
    bx.plot(zv, yv, "o", color=C["tangent"], ms=3.5)
ax.plot(seg[:, 0], seg[:, 1], color=C["accent"], lw=2.2)
bx.plot(zp, yp, color=C["accent"], lw=2.2)
ax.axvline(n, color=C["normal"], lw=1.0, ls="--")
ax.text(n + 0.15, -2.35, r"near plane $|z|=n=1$", fontsize=11, color=C["normal"])
ax.plot(0, 0, "ko", ms=5)
ax.annotate("center", xy=(0, 0), xytext=(-0.2, -1.35), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
ax.text(5.2, 1.9, "line not through\nthe center", fontsize=10.5, color=C["accent"])
ax.set_xlim(-0.3, ZMAX)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"depth $|z|$")
ax.set_ylabel(r"$y$")
ax.set_title("(a) camera space: a fan of rays", fontsize=12)
bx.axvline(-1, color=C["normal"], lw=1.0, ls="--")
bx.axvline(1, color=C["aux"], lw=1.0, ls=":")
bx.text(0.55, -1.25, r"$|z|=\infty$", fontsize=11, color=C["aux"])
bx.text(-0.98, -1.25, r"$|z|=1$", fontsize=11, color=C["normal"])
bx.set_xlim(-1.1, 1.15)
bx.set_ylim(-1.35, 1.35)
bx.set_xlabel(r"$z'=1+2n/z$")
bx.set_ylabel(r"$y'=a_y\,y/z$")
bx.set_title("(b) NDC: parallel lines", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)

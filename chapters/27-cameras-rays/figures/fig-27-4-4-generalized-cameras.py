"""Figure 27.4.4: central, multi-camera, and rolling-shutter cameras as maps from pixels to lines (schematic).

Side view (z horizontal = depth, y vertical). Each panel draws the ray of 7 pixels (one image column) up to depth 5.
(a) Central camera: all rays pass through one center (0, 0).
(b) Two-camera rig: centers (0, 0.6) and (0, -0.6), the second rotated by -25 deg; 4 rays each.
(c) Rolling shutter: the camera moves forward (+z) at speed v while rows are read out top to bottom; row k is exposed at
    time t_k and its ray starts at the center (v t_k, 0) (v t spans 0.9 over the readout, exaggerated). The rays do not
    meet in one point. (With motion along y instead, the rays of one column would still meet in a column-dependent point.)
Self-checks: in (a) all rays share one point; in (c) no three consecutive rays are concurrent (pairwise intersection
points differ).

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-4-4-generalized-cameras.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
rows = np.linspace(-0.45, 0.45, 7)            # normalized image y of 7 pixels


def ray(ax, c, ang0, yb, col, lw=1.0):
    a = ang0 + np.arctan(yb)
    dvec = np.array([np.cos(a), np.sin(a)])
    end = c + 5.0 / dvec[0] * dvec
    ax.plot([c[0], end[0]], [c[1], end[1]], color=col, lw=lw)
    return c, dvec


def meet(r1, r2):
    (c1, d1), (c2, d2) = r1, r2
    A = np.column_stack([d1, -d2])
    s = np.linalg.solve(A, c2 - c1)
    return c1 + s[0] * d1


fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharey=True)
ax = axes[0]
rs = [ray(ax, np.array([0.0, 0]), 0.0, y, C["accent"]) for y in rows]
ax.plot(0, 0, "ks", ms=6)
ax.text(0.1, -0.3, "center", fontsize=11)
assert all(np.allclose(r[0], [0, 0]) for r in rs)
ax.set_title("(a) central camera", fontsize=12.5)

bx = axes[1]
for c, ang, col in ((np.array([0.0, 0.6]), 0.0, C["accent"]), (np.array([0.0, -0.6]), np.radians(-25), C["tangent"])):
    for y in rows[::2]:
        ray(bx, c, ang, y, col)
    bx.plot(*c, "ks", ms=6)
bx.text(0.1, 0.85, "camera 1", fontsize=11, color=C["accent"])
bx.text(0.1, -0.95, "camera 2", fontsize=11, color=C["tangent"])
bx.set_title("(b) two-camera rig", fontsize=12.5)

cx = axes[2]
tk = np.linspace(0, 1, len(rows))
centers = [np.array([0.9 * t, 0.0]) for t in tk]
rs = []
for c, y, t in zip(centers, rows[::-1], tk):
    rs.append(ray(cx, c, 0.0, y, plt.cm.viridis(0.15 + 0.7 * t)))
    cx.plot(*c, "o", color=plt.cm.viridis(0.15 + 0.7 * t), ms=4)
cx.annotate("", xy=(1.1, -0.35), xytext=(-0.1, -0.35), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
cx.text(0.0, -0.95, "camera moves forward\nduring readout", fontsize=10.5)
cx.text(4.9, 2.05, "first row", fontsize=10.5, color=plt.cm.viridis(0.15), ha="right", va="top")
cx.text(4.9, -2.05, "last row", fontsize=10.5, color=plt.cm.viridis(0.85), ha="right", va="bottom")
pts = [meet(rs[i], rs[i + 1]) for i in range(len(rs) - 1)]
assert np.ptp(np.array(pts)[:, 0]) > 1e-3
cx.set_title("(c) rolling shutter", fontsize=12.5)
for a_ in axes:
    a_.set_xlim(-0.4, 5)
    a_.set_ylim(-2.2, 2.2)
    a_.set_xlabel(r"depth $z$")
    a_.set_aspect("equal")
axes[0].set_ylabel(r"$y$")
fig.tight_layout()
dgfig.save(fig, __file__)

"""Figure 33.1.2: a geometric simplicial complex and three collections of simplices that are not (Section 33.1).

(a) a simplicial complex in R^2: four triangles around the vertex (1, 0.9), a dangling edge and an isolated vertex.
(b) T-junction: the vertex (1, 0) of the two lower triangles lies inside the bottom edge of the upper triangle.
(c) overlap: two triangles whose intersection is a small triangle that is a face of neither.
(d) missing faces: a triangle listed without one of its edges.
Self-check: (a) every pair of triangles meets in a common vertex, a common edge, or not at all (checked by clipping);
in (b) and (c) some pair of triangles meets in a set that is not a common face; the clipped areas are as stated.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-1-2-complex-nonexamples.py``
"""

import pathlib
import sys

import dgdec as dd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from ch33 import ccw, clip, meet_is_common_face, poly_area  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
RED = C["normal"]


def draw_tri(ax, T, fc=C["region"], alpha=0.35, ec=C["main"], lw=1.3, z=2):
    ax.add_patch(Polygon(np.asarray(T), closed=True, facecolor=fc, alpha=alpha, edgecolor="none", zorder=z))
    ax.add_patch(Polygon(np.asarray(T), closed=True, fill=False, edgecolor=ec, lw=lw, zorder=z + 1))


def dots(ax, P, color=C["main"], ms=4.2, z=6):
    P = np.asarray(P)
    ax.plot(P[:, 0], P[:, 1], "o", color=color, ms=ms, zorder=z)


fig, axs = plt.subplots(2, 2, figsize=(6.0, 5.2))
(axa, axb), (axc, axd) = axs

# ---------------------------------------------------------------- (a) a simplicial complex
Va = np.array([[1.0, 0.9], [0.0, 0.3], [0.6, -0.2], [1.6, -0.15], [2.1, 0.6], [1.5, 1.7], [2.8, 1.4], [0.15, 1.5]])
Fa = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]
for f in Fa:
    draw_tri(axa, Va[list(f)])
axa.plot(*Va[[4, 6]].T, color=C["main"], lw=1.3, zorder=3)          # dangling edge
dots(axa, Va)
for f in Fa:
    for g in Fa:
        if f < g:
            assert meet_is_common_face(Va[list(f)], Va[list(g)])
K = dd.generate_complex([*Fa, (4, 6), (7,)])
assert dd.is_complex(K) and [len(dd.k_simplices(K, k)) for k in range(3)] == [8, 10, 4]
axa.set_title("(a) a simplicial complex", fontsize=12)
axa.annotate("isolated vertex", xy=Va[7], xytext=(0.0, 2.15), fontsize=10, ha="left",
             arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.7))
axa.annotate("dangling edge", xy=(Va[4] + Va[6]) / 2, xytext=(2.0, 2.15), fontsize=10, ha="left",
             arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.7))
dgfig.schematic_axes(axa, xlim=(-0.3, 3.5), ylim=(-0.5, 2.4))

# ---------------------------------------------------------------- (b) T-junction
top = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.4]])
bl = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -1.0]])
br = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, -1.0]])
for T in (top, bl, br):
    draw_tri(axb, T)
dots(axb, np.vstack([top, bl, br]))
assert not meet_is_common_face(top, bl) and not meet_is_common_face(top, br)
assert meet_is_common_face(bl, br)
axb.plot(*bl[[0, 1]].T, color=RED, lw=2.6, zorder=4)                # top ∩ bl: a face of bl only
axb.plot(1.0, 0.0, "o", color=RED, ms=6.5, zorder=7)
axb.annotate("T-junction", xy=(1.0, 0.0), xytext=(1.5, 0.75), fontsize=10.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
axb.set_title("(b) T-junction", fontsize=12)
dgfig.schematic_axes(axb, xlim=(-0.3, 2.3), ylim=(-1.2, 1.6))

# ---------------------------------------------------------------- (c) overlap
t1 = np.array([[0.0, 0.0], [2.0, 0.0], [0.9, 1.4]])
t2 = np.array([[1.1, 0.6], [2.4, 1.6], [0.6, 1.9]])
for T in (t1, t2):
    draw_tri(axc, T)
dots(axc, np.vstack([t1, t2]))
I = clip(ccw(t1), ccw(t2))
assert poly_area(I) > 0.05 and not meet_is_common_face(t1, t2)
axc.add_patch(Polygon(np.array(I), closed=True, facecolor=RED, alpha=0.45, edgecolor=RED, lw=1.2, zorder=5))
axc.annotate("overlap", xy=np.mean(I, 0), xytext=(1.75, 0.35), fontsize=10.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
axc.set_title("(c) overlap", fontsize=12)
dgfig.schematic_axes(axc, xlim=(-0.3, 2.6), ylim=(-0.4, 2.1))

# ---------------------------------------------------------------- (d) missing face
t = np.array([[0.0, 0.0], [2.0, 0.0], [0.8, 1.5]])
axd.add_patch(Polygon(t, closed=True, facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=2))
axd.plot(*t[[0, 1]].T, color=C["main"], lw=1.3, zorder=3)
axd.plot(*t[[1, 2]].T, color=C["main"], lw=1.3, zorder=3)
axd.plot(*t[[2, 0]].T, color=RED, lw=1.6, ls=(0, (3, 2.2)), zorder=3)
dots(axd, t)
Kd = {(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)}
assert not dd.is_complex(Kd)
axd.annotate("edge not listed", xy=(t[0] + t[2]) / 2, xytext=(-0.25, 1.75), fontsize=10.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
axd.set_title("(d) missing face", fontsize=12)
dgfig.schematic_axes(axd, xlim=(-0.3, 2.3), ylim=(-0.4, 2.0))

fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.08, hspace=0.18)
dgfig.save(fig, __file__)

"""Figure 36.4.1: the discrete exterior derivative by hand on two triangles (Section 36.4, Example 36.4.2).

Mesh lib/ch36.kite: x_0 = (0, 0), x_1 = (2, 0), x_2 = (1, 1.4), x_3 = (2.2, 1.6), faces f = (0, 1, 2), g = (1, 3, 2)
(counterclockwise), edges 01, 02, 12, 13, 23 oriented i -> j (i < j).
(a) the 0-form u = (3, -1, 4, 2) at the vertices and d0 u on the edges: each edge value is (value at the head) -
    (value at the tail).
(b) the 1-form alpha = (1, 2, -1, 3, 0.5) on the edges and d1 alpha in the faces: the circulation around each face in
    its own (counterclockwise) order, an edge value counting with sign -1 when its arrow runs against the face.
Self-check: d0 u = (-4, 1, 5, 3, -2), d1 alpha = (-2, 3.5), d1 d0 u = 0, and the matrices equal dgdec.d0, d1.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-4-1-discrete-d.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Polygon  # noqa: E402

V, F = ch36.kite()
E, index = dd.edge_list(F)
D0, D1 = dd.d0(F, 4), dd.d1(F)
u = np.array([3.0, -1.0, 4.0, 2.0])
alpha = np.array([1.0, 2.0, -1.0, 3.0, 0.5])
du, dal = D0 @ u, D1 @ alpha
assert du.tolist() == [-4, 1, 5, 3, -2] and dal.tolist() == [-2, 3.5] and not np.any(D1 @ du)
assert np.array_equal(D1, np.array([[1, -1, 1, 0, 0], [0, 0, -1, 1, -1]]))

C = dgfig.COLORS
BLUE, RED, GRAY = C["tangent"], C["normal"], "#777777"


def fmt(v):
    s = f"{v:g}"
    return s.replace("-", "−")


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.3))
for ax in (axa, axb):
    ch36.draw_mesh2d(ax, V, F, face="#F5F5F5", edge="#CCCCCC", lw=0.6)
    for e, (i, j) in enumerate(E):
        ch36.edge_arrow(ax, V[i], V[j], GRAY, lw=1.2, head=10, frac=(0.3, 0.7))
    for v, off in zip(range(4), ([-0.2, -0.14], [0.14, -0.18], [-0.24, 0.1], [0.2, 0.06])):
        ax.text(*(V[v] + np.array(off)), rf"$x_{v}$", fontsize=10, color="#555555", ha="center", va="center")
for v in range(4):
    axa.plot(*V[v], "o", ms=19, color="white", mec=BLUE, mew=1.8, zorder=6)
    axa.text(*V[v], fmt(u[v]), fontsize=9.5, ha="center", va="center", color=BLUE, zorder=7, fontweight="bold")
for e, (i, j) in enumerate(E):
    off = -0.16 if (i, j) != (1, 2) else 0.16
    t = ch36.normal_offset(V[i], V[j], off)
    axa.text(*t, fmt(du[e]), fontsize=10.5, ha="center", va="center", color="black", zorder=7,
             bbox=dict(boxstyle="round,pad=0.12", fc="#FFF4D6", ec="none"))
axa.set_title(r"(a) $u$ at vertices, $d_0u$ on edges", fontsize=11)
for e, (i, j) in enumerate(E):
    off = -0.16 if (i, j) != (1, 2) else 0.16
    t = ch36.normal_offset(V[i], V[j], off)
    axb.text(*t, fmt(alpha[e]), fontsize=10.5, ha="center", va="center", color=BLUE, zorder=7)
for f, tri in enumerate(F):
    c = V[tri].mean(0)
    axb.add_patch(Arc(c, 0.55, 0.55, theta1=30, theta2=320, color=RED, lw=1.4, zorder=4))
    tip = c + 0.275 * np.array([np.cos(np.radians(320)), np.sin(np.radians(320))])
    axb.annotate("", xy=tip + np.array([0.04, 0.05]), xytext=tip, arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    axb.text(*c, fmt(dal[f]), fontsize=11, ha="center", va="center", color=RED, zorder=7, fontweight="bold")
axb.set_title(r"(b) $\alpha$ on edges, $d_1\alpha$ in faces", fontsize=11)
for ax in (axa, axb):
    ax.set_xlim(-0.35, 2.55)
    ax.set_ylim(-0.35, 1.85)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)

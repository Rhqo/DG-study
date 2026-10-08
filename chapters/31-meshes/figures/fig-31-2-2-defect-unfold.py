"""Figure 31.2.2: the sign of the angle defect seen by unfolding a vertex star (Section 31.2).

Three stars with the same combinatorics: a vertex at the origin and six neighbours p_k = (cos t_k, sin t_k, z_k),
t_k = 2 pi k / 6.
(a) cone: z_k = -0.55 for all k, (b) flat: z_k = 0, (c) saddle: z_k = 0.32 (-1)^k (three up, three down).
Top row: the stars seen from the side (orthographic camera). Bottom row: the six triangles laid out in the plane
one after another around the vertex, keeping their angles and edge lengths. Orange wedge: the gap left when the
angle sum is below 2 pi (delta > 0). Blue wedge: the overlap when the angle sum exceeds 2 pi (delta < 0).
Self-check: delta = 2 pi - (sum of the six angles) equals the angle of the wedge.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-2-2-defect-unfold.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Wedge  # noqa: E402

C = dgfig.COLORS
t = 2 * np.pi * np.arange(6) / 6
stars = [("(a) cone", np.full(6, -0.55)), ("(b) flat", np.zeros(6)), ("(c) saddle", 0.32 * (-1.0) ** np.arange(6))]

fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2), gridspec_kw=dict(height_ratios=[0.8, 1.0], hspace=0.05, wspace=0.05))
cam = dm.Ortho(elev=12, azim=-70)
base = dm.hex_to_rgb(C["surface"])
for col, (title, z) in enumerate(stars):
    P = np.stack([np.cos(t), np.sin(t), z], 1)
    V = np.vstack([[0, 0, 0], P])
    F = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
    ang = dm.corner_angles(V, F)[:, 0]
    delta = 2 * np.pi - ang.sum()
    # ---- top: 3D view
    ax = axs[0, col]
    cols = dm.shade(V, F, base, light=(0.2, -0.6, 0.8), ambient=0.5)
    Pp = dm.draw_mesh_2d(ax, V, F, cam, cols, edgecolor="#444444", lw=0.7)
    q = cam(V[0])
    ax.plot(*q, "o", color=C["main"], ms=5, zorder=5)
    ax.set_xlim(Pp[:, 0].min() - 0.1, Pp[:, 0].max() + 0.1)
    ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    # ---- bottom: unfolding
    ax = axs[1, col]
    lens = np.linalg.norm(P, axis=1)
    cum = np.concatenate([[0], np.cumsum(ang)])
    for k in range(6):
        a0, a1 = cum[k], cum[k + 1]
        r0, r1 = lens[k], lens[(k + 1) % 6]
        tri = np.array([[0, 0], [r0 * np.cos(a0), r0 * np.sin(a0)], [r1 * np.cos(a1), r1 * np.sin(a1)]])
        ax.add_patch(Polygon(tri, closed=True, fc=C["surface"], ec="#444444", lw=0.8, alpha=0.85, zorder=2))
    rr = 1.25
    if delta > 1e-9:
        ax.add_patch(Wedge((0, 0), rr, np.degrees(cum[-1]), 360, fc=C["accent"], alpha=0.45, ec=C["accent"], lw=1.2,
                           zorder=1))
        txt, colr = rf"gap: $\delta = +{delta:.2f}$", C["accent"]
    elif delta < -1e-9:
        ax.add_patch(Wedge((0, 0), rr, 0, np.degrees(cum[-1] - 2 * np.pi), fc=C["tangent"], alpha=0.35,
                           ec=C["tangent"], lw=1.2, zorder=3))
        txt, colr = rf"overlap: $\delta = {delta:.2f}$", C["tangent"]
    else:
        txt, colr = r"closes: $\delta = 0$", C["main"]
    # first and last ray (the two images of the cut edge)
    for a in (0.0, cum[-1]):
        ax.plot([0, 1.35 * np.cos(a)], [0, 1.35 * np.sin(a)], color=C["main"], lw=1.6, zorder=4)
    assert abs((2 * np.pi - cum[-1]) - delta) < 1e-12
    ax.plot(0, 0, "o", color=C["main"], ms=5, zorder=5)
    ax.text(0, -1.62, txt, ha="center", fontsize=11.5, color=colr)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.8, 1.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if col == 0:
        assert delta > 0
    elif col == 1:
        assert abs(delta) < 1e-12
    else:
        assert delta < 0

fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
dgfig.save(fig, __file__)

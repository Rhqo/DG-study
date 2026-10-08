"""Figure 31.1.1: manifold and non-manifold neighbourhoods in a triangle mesh (Section 31.1).

(a) interior vertex: six triangles form a closed fan; the link (orange) is one closed loop.
(b) boundary vertex: three triangles form an open fan; the link is one path.
(c) non-manifold edge: three triangles share one edge ("three pages of a book").
(d) non-manifold vertex: two closed fans touch only at one vertex; the link is two loops.
Drawn with an orthographic camera and the painter's algorithm (schematic, but from real 3D coordinates).
Self-check: dgmesh.link_type / manifold_report classify the four stars as drawn.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-1-1-manifold-stars.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
cam = dm.Ortho(elev=28, azim=-62)
GRAY = dm.hex_to_rgb(C["surface"])


def ring(n, rad, z, a0=0.0, a1=2 * np.pi, closed=True):
    ang = np.linspace(a0, a1, n, endpoint=not closed)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang), np.full(n, z)], 1)


def draw(ax, V, F, link=(), hl_edges=(), hl_pts=(), center=None, label=None):
    order = np.argsort([cam.depth(V[f].mean(0)) for f in F])  # far to near
    cols = dm.shade(V, F, GRAY * 0.98 + 0.02, light=(0.3, -0.6, 0.9), ambient=0.55)
    for idx in order:
        P = cam(V[F[idx]])
        ax.add_patch(Polygon(P, closed=True, facecolor=cols[idx], edgecolor="#444444", lw=0.8, zorder=2))
    for a, b in link:
        P = cam(V[[a, b]])
        ax.plot(*P.T, color=C["accent"], lw=3.2, solid_capstyle="round", zorder=5)
    for a, b in hl_edges:
        P = cam(V[[a, b]])
        ax.plot(*P.T, color=C["main"], lw=3.4, solid_capstyle="round", zorder=6)
    for p in hl_pts:
        q = cam(V[p])
        ax.plot(*q, "o", ms=6.5, color=C["accent"], mec="k", mew=0.6, zorder=7)
    if center is not None:
        q = cam(V[center])
        ax.plot(*q, "o", ms=7, color=C["main"], zorder=8)
        if label:
            ax.text(q[0] + 0.07, q[1] + 0.12, label, fontsize=13, zorder=9)
    allp = cam(V)
    pad = 0.25
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad - 0.22, allp[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))

# (a) interior vertex: closed fan of 6
Va = np.vstack([[0, 0, 0.35], ring(6, 1.0, 0.0) + np.array([0, 0, 0.0])])
Va[1:, 2] += 0.12 * np.array([1, -1, 0.5, -0.4, 0.8, -0.2])
Fa = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
link_a = dm.vertex_link(Fa, 0)
assert dm.link_type(link_a) == "cycle"
draw(axs[0, 0], Va, Fa, link=link_a, center=0, label=r"$i$")
axs[0, 0].set_title("(a) interior vertex", fontsize=13)
axs[0, 0].text(0.5, 0.03, "link = one closed loop", transform=axs[0, 0].transAxes, ha="center",
               color=C["accent"], fontsize=12.5)

# (b) boundary vertex: open fan of 3
Vb = np.vstack([[0, 0, 0.2], ring(4, 1.0, 0.0, 0, np.pi, closed=False)])
Fb = np.array([[0, 1 + k, 2 + k] for k in range(3)])
link_b = dm.vertex_link(Fb, 0)
assert dm.link_type(link_b) == "path"
draw(axs[0, 1], Vb, Fb, link=link_b, hl_edges=[], center=0, label=r"$i$")
for e in ((0, 1), (0, 4)):  # the two boundary edges at i
    P = cam(Vb[list(e)])
    axs[0, 1].plot(*P.T, color=C["tangent"], lw=2.4, ls=(0, (4, 2)), zorder=6)
axs[0, 1].set_title("(b) boundary vertex", fontsize=13)
axs[0, 1].text(0.5, 0.03, "link = one path", transform=axs[0, 1].transAxes, ha="center",
               color=C["accent"], fontsize=12.5)
axs[0, 1].text(0.70, 0.30, "boundary", transform=axs[0, 1].transAxes, ha="center",
               color=C["tangent"], fontsize=12)
assert sorted(dm.boundary_vertices(Fb)) == [0, 1, 2, 3, 4]

# (c) non-manifold edge: three faces on one edge
# page directions chosen in the camera's horizontal frame so that all three pages are visible
hz = np.array([cam.view[0], cam.view[1], 0.0]) / np.linalg.norm(cam.view[:2])
pages = [(np.cos(a) * cam.right + np.sin(a) * hz, z) for a, z in ((np.radians(0), 0.1), (np.radians(120), -0.3),
                                                               (np.radians(240), 0.45))]
Vc = np.array([[0, 0, -0.55], [0, 0, 0.75]] + [list(d[:2]) + [z] for d, z in pages])
Fc = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
rep = dm.manifold_report(Fc)
assert rep["nonmanifold_edges"] == [(0, 1)]
draw(axs[1, 0], Vc, Fc, hl_edges=[(0, 1)], hl_pts=[2, 3, 4])
q = cam(Vc[1])
axs[1, 0].text(q[0] + 0.1, q[1] + 0.02, r"$e$", fontsize=13)
axs[1, 0].set_title("(c) non-manifold edge", fontsize=13)
axs[1, 0].text(0.5, 0.03, r"3 faces share edge $e$", transform=axs[1, 0].transAxes, ha="center",
               color=C["main"], fontsize=12.5)

# (d) non-manifold vertex: two fans meeting at one vertex
up = ring(5, 0.85, 0.75, 0.3)
dn = ring(5, 0.85, -0.75, 0.9)
Vd = np.vstack([[0, 0, 0], up, dn])
Fd = np.array([[0, 1 + k, 1 + (k + 1) % 5] for k in range(5)] + [[0, 6 + (k + 1) % 5, 6 + k] for k in range(5)])
link_d = dm.vertex_link(Fd, 0)
assert dm.link_type(link_d) == "other"
assert dm.manifold_report(Fd)["nonmanifold_vertices"] == [0]
assert dm.manifold_report(Fd)["nonmanifold_edges"] == []
draw(axs[1, 1], Vd, Fd, link=link_d, center=0, label=r"$i$")
axs[1, 1].set_title("(d) non-manifold vertex", fontsize=13)
axs[1, 1].text(0.5, 0.03, "link = two loops", transform=axs[1, 1].transAxes, ha="center",
               color=C["accent"], fontsize=12.5)

fig.subplots_adjust(wspace=0.05, hspace=0.12, left=0.01, right=0.99, top=0.95, bottom=0.01)
dgfig.save(fig, __file__)

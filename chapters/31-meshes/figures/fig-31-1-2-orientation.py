"""Figure 31.1.2: orientation of a triangle mesh and the Moebius strip (Section 31.1).

(a) Two triangles sharing an edge. Left: consistent orientation (the shared edge is traversed in opposite
    directions). Right: inconsistent (the same direction).
(b) The Moebius mesh in 3D (n = 24 quads, width 2 x 0.5). Starting from one triangle ("start"), the
    orientation is propagated triangle by triangle along the strip; the vermillion arrows are the resulting
    face normals (every fourth triangle, plus the last one, "end"). After one turn the normal comes back
    reversed: "start" and "end" are neighbours across the seam and point to opposite sides.
(c) The Moebius strip cut open: a rectangle [0, 6] x [0, 1] whose ends are glued with a twist,
    (6, y) ~ (0, 1 - y). Every triangle is oriented counterclockwise in the picture (small circular arrows),
    which is consistent inside the rectangle; on the glued edge both end triangles run from P to Q.
Self-check: dgmesh.orient reports a conflict for the Moebius faces, and the propagated last normal points
opposite to the first one across the seam.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-1-2-orientation.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS


def edge_arrows(ax, P, color, inset=0.13, lw=1.6):
    """Arrows along the oriented boundary of triangle P (3x2), pulled toward the centroid."""
    c = P.mean(0)
    Q = c + (1 - inset * 2.2) * (P - c)
    for k in range(3):
        a, b = Q[k], Q[(k + 1) % 3]
        a2, b2 = a + 0.12 * (b - a), b - 0.12 * (b - a)
        ax.add_patch(FancyArrowPatch(a2, b2, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, zorder=4))


def circ_arrow(ax, c, r, ccw=True, color="k"):
    ang = np.linspace(0.2, 1.75 * np.pi, 30)
    if not ccw:
        ang = ang[::-1]
    pts = c + r * np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.plot(*pts[:-1].T, color=color, lw=1.0, zorder=4)
    ax.add_patch(FancyArrowPatch(pts[-3], pts[-1], arrowstyle="-|>", mutation_scale=7, color=color, lw=1.0,
                                 zorder=4))


fig = plt.figure(figsize=(7.2, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], width_ratios=[1.15, 1.0], hspace=0.12, wspace=0.02)

# ---------------------------------------------------------------- (a)
ax = fig.add_subplot(gs[0, 0])
A, B, Cc, D = np.array([0, 0.0]), np.array([1.0, -0.15]), np.array([0.95, 0.95]), np.array([-0.1, 0.9])
for dx, consistent, title in ((0.0, True, "consistent"), (1.55, False, "inconsistent")):
    s = np.array([dx, 0])
    T1 = np.array([A, B, Cc]) + s
    T2 = np.array([A, Cc, D]) + s if consistent else np.array([A, D, Cc]) + s
    for T in (T1, T2):
        ax.add_patch(Polygon(T, closed=True, fc=C["surface"], ec="#444444", lw=0.8, alpha=0.8, zorder=1))
    edge_arrows(ax, T1, C["tangent"])
    edge_arrows(ax, T2, C["third"])
    ax.plot(*np.array([A, Cc]).T + s[:, None], color=C["main"], lw=2.2, zorder=2)
    ax.text(dx + 0.45, -0.45, title, ha="center", fontsize=12.5)
    for lab, p, off in ((r"$i$", A, (-0.12, -0.08)), (r"$j$", Cc, (0.05, 0.05))):
        ax.text(*(p + s + off), lab, fontsize=13.5)
ax.text(0.45 + 1.55, 1.1, "shared edge:\nsame direction", ha="center", fontsize=11, color=C["normal"])
ax.text(0.45, 1.1, "shared edge:\nopposite", ha="center", fontsize=11, color=C["main"])
ax.set_xlim(-0.35, 2.75)
ax.set_ylim(-0.6, 1.5)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a) orienting two triangles", fontsize=13, y=1.0)

# ---------------------------------------------------------------- (b) 3D Moebius with propagated normals
V, F = dm.mobius_mesh(n=24, w=0.5)
_, ok, conflict = dm.orient(F)
assert not ok and conflict is not None
# propagate along the strip order (consecutive faces share an edge)
Fp = [F[0].copy()]
for f in F[1:]:
    prev = Fp[-1]
    dprev = {(prev[0], prev[1]), (prev[1], prev[2]), (prev[2], prev[0])}
    df = {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}
    shared = [(a, b) for (a, b) in df if (a, b) in dprev or (b, a) in dprev]
    assert len(shared) == 1
    a, b = shared[0]
    Fp.append(f[::-1].copy() if (a, b) in dprev else f.copy())
Fp = np.array(Fp)
N = dm.face_normals(V, Fp)
# across the seam (last face vs first face) the shared edge is traversed in the same direction
last, first = Fp[-1], Fp[0]
dl = {(last[0], last[1]), (last[1], last[2]), (last[2], last[0])}
dfst = {(first[0], first[1]), (first[1], first[2]), (first[2], first[0])}
assert dl & dfst, "seam must show the orientation conflict"

cam = dm.Ortho(elev=28, azim=12)
ax3 = fig.add_subplot(gs[0, 1])
items = []
base = dm.hex_to_rgb(C["surface"])
cols = dm.shade(V, Fp, base, light=(0.2, -0.6, 0.8), ambient=0.6)
for k, f in enumerate(Fp):
    items.append(("face", cam.depth(V[f].mean(0)), k))
show = list(range(0, len(Fp) - 4, 4)) + [len(Fp) - 1]
for k in show:
    items.append(("arrow", cam.depth(V[Fp[k]].mean(0) + 0.15 * N[k]), k))
for kind, _, k in sorted(items, key=lambda t: t[1]):
    if kind == "face":
        ax3.add_patch(Polygon(cam(V[Fp[k]]), closed=True, fc=cols[k], ec="#555555", lw=0.5, zorder=1))
    else:
        c0 = V[Fp[k]].mean(0)
        special = k in (0, show[-1])
        ln = 0.45 if special else 0.26
        p0, p1 = cam(c0), cam(c0 + ln * N[k])
        ax3.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11 if special else 8,
                                      color=C["normal"], lw=2.4 if special else 1.0, alpha=1.0 if special else 0.75,
                                      zorder=3))
for k, lab in ((0, "start"), (show[-1], "end")):
    tip = cam(V[Fp[k]].mean(0) + 0.45 * N[k])
    right = (cam(V[Fp[k]].mean(0))[0] > cam(V[Fp[0 if k else show[-1]]].mean(0))[0])
    ax3.text(tip[0] + (0.06 if right else -0.06), tip[1], lab, fontsize=13, color=C["normal"],
             ha="left" if right else "right", va="center")
assert np.dot(N[0], N[-1]) < -0.5  # the normal came back reversed
P = cam(V)
ax3.set_xlim(P[:, 0].min() - 0.25, P[:, 0].max() + 0.25)
ax3.set_ylim(P[:, 1].min() - 0.25, P[:, 1].max() + 0.35)
ax3.set_aspect("equal")
ax3.set_axis_off()
ax3.set_title("(b) propagated normals", fontsize=13, y=1.0)

# ---------------------------------------------------------------- (c) development of the strip
axb = fig.add_subplot(gs[1, :])
n = 6
for k in range(n):
    p00, p10, p11, p01 = np.array([k, 0.0]), np.array([k + 1.0, 0]), np.array([k + 1.0, 1]), np.array([k, 1.0])
    for T in (np.array([p00, p10, p11]), np.array([p00, p11, p01])):
        axb.add_patch(Polygon(T, closed=True, fc=C["surface"], ec="#444444", lw=0.8, alpha=0.8, zorder=1))
        circ_arrow(axb, T.mean(0), 0.12, ccw=True, color=C["tangent"])
# glued edges
axb.plot([0, 0], [0, 1], color=C["main"], lw=2.4, zorder=2)
axb.plot([n, n], [0, 1], color=C["main"], lw=2.4, zorder=2)
axb.text(-0.12, 1.02, r"$P$", fontsize=14, ha="right")
axb.text(-0.12, -0.08, r"$Q$", fontsize=14, ha="right")
axb.text(n + 0.1, 1.02, r"$Q$", fontsize=14)
axb.text(n + 0.1, -0.08, r"$P$", fontsize=14)
# orientation of the end triangles on the glued edge
axb.add_patch(FancyArrowPatch((0.13, 0.86), (0.13, 0.14), arrowstyle="-|>", mutation_scale=13, color=C["accent"],
                              lw=2.2, zorder=5))
axb.add_patch(FancyArrowPatch((n - 0.13, 0.14), (n - 0.13, 0.86), arrowstyle="-|>", mutation_scale=13,
                              color=C["accent"], lw=2.2, zorder=5))
axb.text(n / 2, -0.2, r"glue the ends with a twist: $(6, y) \sim (0, 1-y)$" + "\n"
         + r"both orange arrows run from $P$ to $Q$: the glued edge is traversed the same way", ha="center",
         va="top", fontsize=11.5)
axb.set_xlim(-0.6, n + 0.6)
axb.set_ylim(-0.75, 1.25)
axb.set_aspect("equal")
axb.set_axis_off()
axb.set_title("(c) the Moebius strip cut open", fontsize=13, y=0.98)

fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version: Moebius + propagated normals
tr = [dm.plotly_mesh(V, Fp, color="#D0D0D0", name="Moebius strip"), dm.plotly_edges(V, Fp)]
cent = V[Fp].mean(1)
tr += dm.plotly_arrows(cent[::2], 0.3 * N[::2], color="#D55E00", cone=0.07, name="propagated normal")
tr += dm.plotly_arrows(cent[[0, -1]], 0.45 * N[[0, -1]], color="#E69F00", width=8, cone=0.1, name="start / end")
dm.plotly_save(tr, __file__, "Moebius strip (24 quads): normals propagated along the strip; "
               "orange = first and last triangle (opposite across the seam)", "fig-31-1-2-interactive")

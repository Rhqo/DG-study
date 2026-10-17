"""Figure 34.1.1: a 1-vector, a 2-vector and a 3-vector in R^3 drawn as an oriented segment, an oriented
parallelogram and an oriented parallelepiped (Section 34.1, Intuition).

Vectors: u = (1, 0.15, 0), v = (0.3, 0.9, 0), w = (0.15, 0.2, 0.85). The three objects are drawn side by side
(translated along x; translation does not change a k-vector). Curved arrows show the orientation: on u ^ v the
turn from u to v, on u ^ v ^ w the same turn on the bottom face followed by the extrusion along w.
Self-checks: |u ^ v| = |u x v|, det(u, v, w) > 0, and the orientation arc turns from u toward v.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
u = np.array([1.0, 0.15, 0.0])
v = np.array([0.3, 0.9, 0.0])
w = np.array([0.15, 0.2, 0.85])

area = np.linalg.norm(np.cross(u, v))
vol = np.linalg.det(np.column_stack([u, v, w]))
assert vol > 0, "u, v, w must be positively oriented"
assert abs(area - np.sqrt(np.dot(u, u) * np.dot(v, v) - np.dot(u, v) ** 2)) < 1e-12
print(f"|u^v| = {area:.4f}, u^v^w = {vol:.4f} e123")


def arc(center, a, b, r, t0=0.18, t1=0.82, n=40):
    """Points of a circular arc of radius r around center in the plane of a, b, turning from a toward b."""
    e1 = a / np.linalg.norm(a)
    b_perp = b - np.dot(b, e1) * e1
    e2 = b_perp / np.linalg.norm(b_perp)
    ang = np.arccos(np.clip(np.dot(e1, b / np.linalg.norm(b)), -1, 1))
    ts = np.linspace(t0, t1, n) * (2 * np.pi * 0.75)
    pts = center + r * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2)
    # the arc runs in the direction that turns e1 toward e2 (i.e., from a toward b)
    assert np.dot(np.cross(pts[1] - center, pts[2] - center), np.cross(a, b)) > 0
    assert 0 < ang < np.pi
    return pts


def draw_arc(ax, pts, color, lw=1.1, zorder=9):
    ax.plot(*pts.T, color=color, lw=lw, zorder=zorder)
    dgfig.arrow3d(ax, pts[-3], pts[-1] - pts[-3], role=color, lw=lw, head=9, zorder=zorder)


fig = plt.figure(figsize=(6.4, 3.0))
axA = fig.add_subplot(121, projection="3d", computed_zorder=False)
axB = fig.add_subplot(122, projection="3d", computed_zorder=False)
for ax in (axA, axB):
    ax.view_init(elev=24, azim=-62)
    ax.set_axis_off()
o = np.zeros(3)

# (a) 1-vectors u, v and the 2-vector u ^ v: filled parallelogram, orientation arc from u toward v
P = np.array([o, u, u + v, v])
axA.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.35, edgecolor=C["tangent"],
                                      linewidth=0.8, zorder=2))
dgfig.arrow3d(axA, o, u, role="tangent", label=r"$u$", label_offset=(0.0, -0.16, 0.0), fontsize=13)
dgfig.arrow3d(axA, o, v, role="third", label=r"$v$", label_offset=(-0.12, 0.05, 0.05), fontsize=13)
draw_arc(axA, arc(0.5 * (u + v), u, v, 0.2), C["main"])
axA.text(*(0.75 * (u + v) + np.array([0.3, -0.2, 0.0])), r"$u\wedge v$", fontsize=13, ha="center")
axA.set_title("(a) 2-vector $u\\wedge v$", fontsize=12, y=0.93)

# (b) 3-vector u ^ v ^ w: parallelepiped (faces lightly filled), bottom face = u ^ v, extrusion along w
corners = {}
for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            corners[(a, b, c)] = o + a * u + b * v + c * w
faces = [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
         [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
         [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]]
polys = [[corners[k] for k in f] for f in faces]
axB.add_collection3d(Poly3DCollection(polys, facecolor=C["region"], alpha=0.13, edgecolor="#555555",
                                      linewidth=0.6, zorder=1))
axB.add_collection3d(Poly3DCollection([polys[0]], facecolor=C["region"], alpha=0.3, edgecolor="none",
                                      zorder=1.5))
dgfig.arrow3d(axB, o, u, role="tangent", label=r"$u$", label_offset=(0.0, -0.16, 0.0), fontsize=13)
dgfig.arrow3d(axB, o, v, role="third", label=r"$v$", label_offset=(-0.13, 0.04, 0.0), fontsize=13)
dgfig.arrow3d(axB, o, w, role="accent", label=r"$w$", label_offset=(-0.1, 0.0, 0.06), fontsize=13)
draw_arc(axB, arc(0.5 * (u + v), u, v, 0.2), C["main"])
axB.set_title("(b) 3-vector $u\\wedge v\\wedge w$", fontsize=12, y=0.93)

ptsA = np.concatenate([P, np.array([[0.0, 0.0, -0.05], [0, 0, 0.3]])])
ptsB = np.array(list(corners.values()))
dgfig.equal_aspect(axA, ptsA, zoom=0.95)
dgfig.equal_aspect(axB, ptsB, zoom=0.95)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)

"""Figure 38.4.3: two planar vertex stars whose edges and triangles are all nondegenerate but which are not locally
injective at the center.

(a) folded star: a hexagonal star (link vertices at 0, 60, ..., 300 degrees on the unit circle) with link vertex 2
moved from 60 degrees to 150 degrees at radius 0.8. Face (0, 2, 3) turns clockwise (orange) and overlaps face (0, 1, 2).
The angles at the center add up to 150 - 30 + 4 * 60 = 360 degrees.
(b) branch point: valence-5 star whose link vertex k sits at 90 + 144 k degrees; the link is a pentagram, every face
turns counterclockwise with angle 144 degrees at the center, and the angles add up to 720 degrees: every direction
from the center is covered twice.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-4-3-nonimmersions.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402

C = dgfig.COLORS
ez = np.array([0.0, 0.0, 1.0])

Va, Fa = ddg38.folded_star()
Vb, Fb = ddg38.pentagram_star()
anga = ddg38.projected_star_angles(Va, Fa, 0, ez)
angb = ddg38.projected_star_angles(Vb, Fb, 0, ez)
# self-checks: nondegenerate, the claimed angle sums and signs, overlap > 0, the tests fail
for V, F in ((Va, Fa), (Vb, Fb)):
    assert np.all(dm.face_areas(V, F) > 1e-3)
    assert min(np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)) > 0.1
    assert not ddg38.star_passes(V, F, 0, ez)
    assert ddg38.max_star_overlap(V, F, 0) > 1e-3
assert np.allclose(np.rad2deg(anga), [150, -30, 60, 60, 60, 60])
assert np.allclose(np.rad2deg(angb), [144] * 5)

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.4))
for ax, (V, F, ang), title in zip(axs, ((Va, Fa, anga), (Vb, Fb, angb)), ("(a)", "(b)")):
    for f, a in zip(F, ang):
        col = C["accent"] if a < 0 else C["region"]
        ax.add_patch(Polygon(V[f][:, :2], closed=True, fc=col, alpha=0.35, ec=C["main"], lw=0.9))
    ax.plot(*V[0, :2], "o", color=C["main"], ms=5)
    for k in range(1, len(V)):
        ax.plot(*V[k, :2], "o", color=C["main"], ms=3.5)
        ax.text(*(1.14 * V[k, :2] - np.array([0.05, 0.05])), str(k), fontsize=11)
    ax.text(0.06, -0.17, "0", fontsize=11)
    dgfig.schematic_axes(ax, (-1.3, 1.3), (-1.25, 1.3))
    ax.set_title(title, loc="left", fontsize=12)
axs[0].text(-1.25, 0.95, r"$-30^\circ$", fontsize=11, color=C["accent"])
axs[1].text(-0.5, -1.25, r"$\sum\theta = 4\pi$", fontsize=11)
fig.tight_layout(w_pad=2.0)
dgfig.save(fig, __file__)

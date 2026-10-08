"""Figure 28.3.1: moving the whole reconstruction along the Sim(3) orbit does not change any image (Section 28.3).

The small BA of Figure 28.3.2 (5 cameras, 40 points). The similarity X -> s Q X + u with s = 1.4, Q = rotation by
25 degrees about the world y axis, u = (0.6, 0.2, -0.5) is applied to all points and cameras
(T_cw -> (R Q^T, s t - R Q^T u)).
(a) top view (world x-z plane; the cameras look roughly along +z): original points (gray dots) and cameras (black
    triangles = camera center and viewing direction), transformed points (light orange dots) and cameras (orange).
(b) image of camera 3 (normalized coordinates): black dots = original points in the original camera, orange circles =
    transformed points in the transformed camera (they coincide), gray crosses = transformed points seen by the
    ORIGINAL camera 3 (they do not).
Self-check: max difference between black dots and orange circles < 1e-13 for all 5 cameras.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-3-1-gauge-orbit.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgba as B  # noqa: E402
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
cams, Xs = B.make_ba()
s, Q, u = 1.4, L.expSO3(np.radians(25) * np.array([0, 1.0, 0])), np.array([0.6, 0.2, -0.5])
cams2, Xs2 = B.apply_similarity(cams, Xs, s, Q, u)
P1 = B.project_all(cams, Xs)
P2 = B.project_all(cams2, Xs2)
assert np.abs(P1 - P2).max() < 1e-13
Pwrong = B.project_all(cams, Xs2)
K = 2

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 4.6), gridspec_kw=dict(wspace=0.28, width_ratios=[1.15, 1]))


def draw_cam(axh, T, col, lab=None, size=0.65, fill="none"):
    """Camera glyph in the x-z plane: apex = camera center, opening = viewing direction (camera +z)."""
    Twc = L.inv_T(T)
    cen = Twc[:3, 3]
    f = Twc[:3, 2]
    r = Twc[:3, 0]
    tip1 = cen + size * (f + 0.5 * r)
    tip2 = cen + size * (f - 0.5 * r)
    tri = np.array([cen, tip1, tip2, cen])
    axh.fill(tri[:, 0], tri[:, 2], color=fill, alpha=0.35 if fill != "none" else 1.0, lw=0, zorder=4)
    axh.plot(tri[:, 0], tri[:, 2], color=col, lw=1.5, zorder=5)
    axh.plot(cen[0], cen[2], "o", color=col, ms=4, zorder=5)
    if lab:
        off = -0.75 * f[[0, 2]] / np.linalg.norm(f[[0, 2]])     # put the number behind the camera
        axh.text(cen[0] + off[0], cen[2] + off[1], lab, color=col, fontsize=10.5, ha="center", va="center")


ax.scatter(Xs[:, 0], Xs[:, 2], s=9, color="#8A8A8A", zorder=3, label=r"points $X_j$")
ax.scatter(Xs2[:, 0], Xs2[:, 2], s=9, color="#F3C46B", zorder=3, label=r"moved points $sQX_j + u$")
for i, T in enumerate(cams):
    draw_cam(ax, T, C["main"], str(i + 1) if i in (0, 2, 4) else None, fill="#BBBBBB")
for i, T in enumerate(cams2):
    draw_cam(ax, T, C["accent"], str(i + 1) if i in (0, 2, 4) else None, fill="#F3C46B")
ax.plot([], [], marker=">", color=C["main"], mfc="#BBBBBB", ls="none", ms=8, label="cameras (apex = center)")
ax.plot([], [], marker=">", color=C["accent"], mfc="#F3C46B", ls="none", ms=8, label="moved cameras")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, frameon=False, columnspacing=1.0)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$ (world)")
ax.set_ylabel(r"$z$ (world)")
ax.set_title("(a) top view (world $x$-$z$ plane)", fontsize=11.5)
allp = np.vstack([Xs, Xs2] + [L.inv_T(T)[:3, 3] for T in cams + cams2])
xlo, xhi = allp[:, 0].min() - 0.8, allp[:, 0].max() + 0.8
zlo, zhi = allp[:, 2].min() - 0.9, allp[:, 2].max() + 1.6
ax.text(xlo + 0.2, zhi - 0.25, r"$s = 1.4$, $Q$: $25^\circ$ about $y$," + "\n" + r"$u = (0.6, 0.2, -0.5)$", fontsize=10.5, va="top")
ax.set_xlim(xlo, xhi)
ax.set_ylim(zlo, zhi)

bx.plot(P1[K, :, 0], P1[K, :, 1], "o", color=C["main"], ms=4.5, label="original scene, original camera")
bx.plot(P2[K, :, 0], P2[K, :, 1], "o", mfc="none", mec=C["accent"], ms=9, mew=1.4,
        label="moved scene, moved camera")
bx.plot(Pwrong[K, :, 0], Pwrong[K, :, 1], "x", color=C["aux"], ms=5, mew=1.1,
        label="moved scene, original camera")
bx.invert_yaxis()
bx.set_aspect("equal")
bx.set_xlabel(r"$x/z$")
bx.set_ylabel(r"$y/z$")
bx.set_title("(b) image of camera 3", fontsize=11.5)
bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), fontsize=9.5, frameon=False)

dgfig.save(fig, __file__)

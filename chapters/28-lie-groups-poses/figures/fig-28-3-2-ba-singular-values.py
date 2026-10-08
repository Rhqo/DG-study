"""Figure 28.3.2: the singular values of the bundle-adjustment Jacobian; monocular BA has exactly 7 zero singular
values (Section 28.3).

Small BA: N = 5 cameras on an arc (T_cw, OpenCV axes) looking at M = 40 points in [-1.2, 1.2] x [-1, 1] x [4, 6].
Local coordinates: left (camera-frame) perturbations for the cameras and additive updates for the points, so the
Jacobian has 6N + 3M = 150 columns ((28.2.3)). Three Jacobians, evaluated at the true configuration:
  black : monocular reprojection residuals only (400 rows)
  blue  : plus one metric-depth residual z_c - d per observation (RGB-D, 600 rows)
  orange: monocular, with the gauge fixed by removing the 6 columns of camera 1 and the first translation column of
          camera 2 (143 columns)
All singular values are sorted in decreasing order and plotted against their index (log scale). Values below 1e-17 are
drawn at 1e-17.
Self-check: exactly 7 (black), 6 (blue) and 0 (orange) singular values below 1e-10 relative to the largest; the
black null space equals the span of the 7 gauge generators (B.gauge_generators) to 1e-12.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-3-2-ba-singular-values.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgba as B  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
cams, Xs = B.make_ba()
J = B.jacobian(cams, Xs)
Jd = B.jacobian(cams, Xs, depth=True)
keep = [c for c in range(J.shape[1]) if c >= 7]
Jf = J[:, keep]
svs = [np.linalg.svd(A, compute_uv=False) for A in (J, Jd, Jf)]
nulls = [int(np.sum(sv < 1e-10 * sv[0])) for sv in svs]
assert nulls == [7, 6, 0], nulls
G = B.gauge_generators(cams, Xs)
_, _, Vt = np.linalg.svd(J)
Qn = Vt[-7:].T
assert np.linalg.norm(G - Qn @ (Qn.T @ G)) < 1e-12 * np.linalg.norm(G)

fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06, width_ratios=[1.0, 1.1]))
FLOOR = 1e-17
styles = [(svs[0], C["main"], "-", "o", "monocular (reprojection only)"),
          (svs[1], C["tangent"], (0, (5, 2)), "s", "with metric depth (RGB-D)"),
          (svs[2], C["accent"], "-", "^", "monocular, gauge fixed")]
for ax, lo in zip(axs, [1, 131]):
    for sv, col, ls, mk, lab in styles:
        idx = np.arange(1, len(sv) + 1)
        sel = idx >= lo
        ax.semilogy(idx[sel], np.maximum(sv[sel], FLOOR), color=col, ls=ls, lw=1.8, marker=mk,
                    ms=2.5 if lo == 1 else 5, label=lab)
    ax.grid(True, which="major", color="#E5E5E5", lw=0.5)
    ax.set_xlabel("index (decreasing order)")
axs[0].set_xlim(0, 152)
axs[1].set_xlim(130.5, 150.5)
axs[1].set_xticks([132, 136, 140, 144, 148])
axs[0].set_ylim(3e-18, 30)
axs[0].set_ylabel(r"singular value of $J$")
axs[0].set_title("(a) all 150 singular values", fontsize=12)
axs[1].set_title("(b) the last 20 (zoom)", fontsize=12)
axs[0].legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=0.95)
axs[1].annotate("7 zeros\n(indices 144-150)", xy=(147, 1.5e-16), xytext=(131.5, 1e-13), fontsize=11,
                arrowprops=dict(arrowstyle="->", lw=1.0))
axs[1].annotate("6 zeros (145-150)", xy=(146, 5e-16), xytext=(131.5, 3e-10), fontsize=11, color=C["tangent"],
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C["tangent"]))
axs[1].annotate(r"gauge fixed: 143 values," + "\n" + r"smallest $%.4f$" % svs[2][-1], xy=(143, svs[2][-1]),
                xytext=(131.5, 1e-6), fontsize=11, color="#B07400",
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C["accent"]))
fig.suptitle(r"$N = 5$ cameras, $M = 40$ points, $6N + 3M = 150$ parameters", fontsize=12, y=1.02)

dgfig.save(fig, __file__)

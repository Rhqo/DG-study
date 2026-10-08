"""Figure 28.3.3: monocular scale drift and loop closure with SE(3) and Sim(3) pose graphs (Section 28.3).

Simulation (lib/dgba.drift_experiment): K = 60 keyframes on a circle of radius 10 (keyframe 60 is back at
keyframe 0). The odometry between keyframes k and k+1 is measured with rotation noise 0.3 deg and with its
translation in the local map scale s_k; log s_k is a random walk with drift -0.006 and step std 0.01 per keyframe
(seed 11), so s_60 = 0.67. Loop closure between keyframe 60 and keyframe 0:
  * SE(3) pose graph: loop edge = identity rigid motion (same place, same orientation);
  * Sim(3) pose graph [Strasdat10, MurArtal15]: odometry edges with scale 1, loop edge with scale ratio
    s_0/s_60 (what a 3D-3D alignment of the two local maps measures).
Both are solved by Gauss-Newton with keyframe 0 fixed (lib/dgba.pose_graph), identity information matrices.
(a) top view: ground truth (thick gray), dead reckoning (chained odometry, black dashed), SE(3) result (blue),
    Sim(3) result (orange).
(b) the local scale along the loop: true s_k (black) and the Sim(3) estimate 1/s~_k (orange), where s~_k is the
    scale of the optimized vertex k.
Self-check: the position RMS error (no alignment) of the Sim(3) result is < 0.4 and < 1/5 of the SE(3) result; both
pose graphs close the loop (end-to-start gap < 0.05).

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-3-3-scale-drift.py``
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
E = B.drift_experiment()
P = E["P"]
dr = np.array([T[:3, 3] for T in E["DR"]])
Xse, _ = B.pose_graph(E, "se3")
Xsim, _ = B.pose_graph(E, "sim")
pse = np.array([X[:3, 3] for X in Xse])
psim = np.array([X[:3, 3] for X in Xsim])
ssim = np.array([np.cbrt(np.linalg.det(X[:3, :3])) for X in Xsim])
e_dr, e_se, e_sim = B.rms(dr, P), B.rms(pse, P), B.rms(psim, P)
assert e_sim < 0.4 and e_sim < e_se / 5, (e_dr, e_se, e_sim)
assert np.linalg.norm(pse[-1] - pse[0]) < 0.05 and np.linalg.norm(psim[-1] - psim[0]) < 0.05

fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 4.2), gridspec_kw=dict(wspace=0.3, width_ratios=[1.1, 1]))
ax.plot(P[:, 0], P[:, 1], color="#BDBDBD", lw=5, label="ground truth", zorder=1)
ax.plot(dr[:, 0], dr[:, 1], color=C["main"], lw=1.6, ls=(0, (4, 2)), label="odometry only", zorder=2)
ax.plot(pse[:, 0], pse[:, 1], color=C["tangent"], lw=1.8, label=r"$\mathrm{SE}(3)$ pose graph", zorder=3)
ax.plot(psim[:, 0], psim[:, 1], color=C["accent"], lw=1.8, label=r"$\mathrm{Sim}(3)$ pose graph", zorder=4)
ax.plot(0, 0, "o", color=C["main"], ms=6, zorder=5)
ax.text(0.4, -1.4, "start = keyframe 0", fontsize=10.5)
ax.plot(dr[-1, 0], dr[-1, 1], "s", color=C["main"], ms=5, zorder=5)
ax.annotate("odometry end:\ngap %.1f" % np.linalg.norm(dr[-1] - dr[0]), xy=dr[-1, :2], xytext=(-12.0, 0.6),
            fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_aspect("equal")
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-2.5, 22.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5, frameon=False)
ax.set_title("(a) top view of the loop", fontsize=12)

k = np.arange(E["K"] + 1)
bx.plot(k, E["s"], color=C["main"], lw=1.8, label=r"true local scale $s_k$")
bx.plot(k, 1 / ssim, color=C["accent"], lw=1.8, ls=(0, (5, 2)), label=r"$\mathrm{Sim}(3)$ estimate $1/\tilde s_k$")
bx.axhline(1.0, color=C["aux"], lw=0.7, ls=":")
bx.set_xlabel("keyframe $k$")
bx.set_ylabel("scale")
bx.set_ylim(0.55, 1.08)
bx.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)
bx.set_title("(b) scale along the loop", fontsize=12)
bx.annotate(r"$s_{60} = %.2f$" % E["s"][-1], xy=(60, E["s"][-1]), xytext=(41, 0.6), fontsize=11,
            arrowprops=dict(arrowstyle="->", lw=0.9))
bx.text(1.0, 1.015, "no drift", fontsize=9.5, color=C["aux"])

print(f"RMS position error: odometry {e_dr:.2f}, SE(3) {e_se:.2f}, Sim(3) {e_sim:.2f}; s_K = {E['s'][-1]:.3f}")
dgfig.save(fig, __file__)

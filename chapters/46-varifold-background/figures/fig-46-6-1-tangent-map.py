"""Figure 46.6.1: the derivative of the tangent projection is the curvature.

(a) the ellipse (2 cos t, sin t) with 48 points at equal arc-length spacing, coloured by the curvature kappa.
(b) the tangent lines at these points as points of G_{1,2} (coordinates (P11 - P22, 2 P12) of Figure 46.3.1): where the ellipse
is flat the points crowd together, where it bends they spread out; consecutive points are about 2 kappa ds apart
along the circle (the angle is doubled), and the operator-norm speed |dP/ds| equals kappa.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-6-1-tangent-map.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
L = v.ellipse_length_total()
tg = np.linspace(0, 2 * np.pi, 20001)
sg = np.concatenate([[0.0], np.cumsum(0.5 * (v.ellipse_speed(tg[1:]) + v.ellipse_speed(tg[:-1])) * np.diff(tg))])
sg *= L / sg[-1]
M = 48
s = np.arange(M) * L / M
t = np.interp(s, sg, tg)
X = v.ellipse(t)
T = v.ellipse_tangent(t)
P = v.dir_proj(T)
xy = v.proj_coords(P)

# self-check: antipodal points have the same tangent line, so the 48 points give 24 points of G_{1,2}
assert np.allclose(P[: M // 2], P[M // 2:])
# horizontal line -> (1, 0), vertical line -> (-1, 0)
assert np.allclose(v.proj_coords(v.dir_proj(np.array([[1.0, 0.0], [0.0, 1.0]]))), [[1, 0], [-1, 0]])
# self-check: |dP/ds|_op = kappa (finite differences in arc length)
for t0 in (0.0, 0.4, np.pi / 2):
    h = 1e-6
    dP = (v.dir_proj(v.ellipse_tangent(t0 + h)) - v.dir_proj(v.ellipse_tangent(t0 - h))) / (2 * h * v.ellipse_speed(t0))
    assert abs(np.linalg.norm(dP, 2) - v.ellipse_kappa(t0)) < 1e-6

kap = v.ellipse_kappa(t)
cmap = plt.get_cmap("viridis")
cols = cmap((kap - kap.min()) / (kap.max() - kap.min()))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.15})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(*v.ellipse(tt).T, color=C["aux"], lw=0.8)
sc = ax.scatter(X[:, 0], X[:, 1], c=kap, cmap=cmap, s=16, zorder=3, vmin=kap.min(), vmax=kap.max())
ax.set_aspect("equal")
ax.set_xlim(-2.3, 2.75)
ax.set_ylim(-1.45, 1.57)
ax.set_axis_off()
ax.text(2.12, 0.0, r"$\kappa = 2$", fontsize=10, ha="left", va="center")
ax.text(0.0, 1.08, r"$\kappa = 1/4$", fontsize=10, ha="center", va="bottom")
ax.set_title("(a) equal arc-length points", fontsize=12)

ax = axes[1]
ax.plot(np.cos(tt), np.sin(tt), color=C["aux"], lw=0.8)
ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=22, zorder=3)
ax.text(0, 0.4, r"$G_{1,2}$", fontsize=11, ha="center", va="center", color=C["aux"])
ax.text(-0.85, 0.08, "vertical", fontsize=9, ha="left", va="center")
ax.text(0.8, -0.3, "horizontal", fontsize=9, ha="right", va="center")
ax.set_aspect("equal")
ax.set_xlim(-1.45, 1.45)
ax.set_ylim(-1.3, 1.3)
ax.set_axis_off()
ax.set_title(r"(b) the same lines in $G_{1,2}$", fontsize=12)

dgfig.save(fig, __file__)
print("kappa range", kap.min().round(4), kap.max().round(4))

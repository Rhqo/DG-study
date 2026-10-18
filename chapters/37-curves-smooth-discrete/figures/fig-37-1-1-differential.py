"""Figure 37.1.1: the differential of a parametrized plane curve stretches tangent vectors of the parameter interval.

(a) unit vectors X = 1 at twelve equally spaced parameters t_j of [0, 2 pi]; (b) the ellipse gamma(t) = (2 cos t, sin t)
with the images d gamma(X) = X gamma'(t_j) (drawn at 0.35 of their length), whose lengths |gamma'(t_j)| range from 1 to 2.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-1-1-differential.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg37  # noqa: E402

C = dgfig.COLORS
m = 12
tj = 2 * np.pi * np.arange(m) / m
P = ddg37.point("ellipse", tj)
V = ddg37.velocity("ellipse", tj)
spd = np.linalg.norm(V, axis=1)
assert abs(spd.min() - 1) < 1e-12 and abs(spd.max() - 2) < 1e-12          # |gamma'| = sqrt(4 sin^2 + cos^2)
assert np.allclose(V[:, 0], -2 * np.sin(tj)) and np.allclose(V[:, 1], np.cos(tj))
SC = 0.35

fig = plt.figure(figsize=(6.2, 4.6))
gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 3.2], hspace=0.25)
ax = fig.add_subplot(gs[0])
bx = fig.add_subplot(gs[1])

# (a) the parameter interval with unit tangent vectors
ax.plot([0, 2 * np.pi], [0, 0], color=C["main"], lw=1.4)
for t in tj:
    ax.annotate("", xy=(t + SC, 0), xytext=(t, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.3, mutation_scale=8))
    ax.plot([t], [0], "o", color=C["main"], ms=2.5)
ax.text(0, -0.75, r"$0$", ha="center", fontsize=11)
ax.text(2 * np.pi, -0.75, r"$2\pi$", ha="center", fontsize=11)
ax.text(np.pi / 2, -0.75, r"$\pi/2$", ha="center", fontsize=11)
ax.text(tj[1] + 0.05, 0.35, r"$X$", color=C["tangent"], fontsize=11)
ax.text(2 * np.pi + 0.35, -0.1, r"$t$", fontsize=11)
ax.set_xlim(-0.4, 2 * np.pi + 0.7)
ax.set_ylim(-1.1, 0.9)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")

# (b) the ellipse and the stretched vectors d gamma(X) = gamma'(t_j)
s = np.linspace(0, 2 * np.pi, 400)
S = ddg37.point("ellipse", s)
bx.plot(S[:, 0], S[:, 1], color=C["main"], lw=dgfig.LW["main"])
OUT = -(V / spd[:, None]) @ ddg37.J.T          # outward unit normal -J t (the curve is counterclockwise)
for p, v, o in zip(P, V, OUT):
    q = p + 0.13 * o                           # drawn just outside the curve, parallel to it
    bx.plot([p[0], q[0]], [p[1], q[1]], color=C["aux"], lw=0.6)
    bx.annotate("", xy=q + SC * v, xytext=q,
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=9))
    bx.plot([p[0]], [p[1]], "o", color=C["main"], ms=2.5, zorder=5)
bx.text(P[0, 0] + 0.3, P[0, 1] - 0.05, r"$t=0:\ |d\gamma(X)| = 1$", fontsize=10.5)
bx.text(P[3, 0], P[3, 1] + 0.32, r"$t=\pi/2:\ |d\gamma(X)| = 2$", fontsize=10.5, ha="center")
bx.text(0, -0.5, r"$\gamma(t) = (2\cos t,\ \sin t)$", fontsize=11, ha="center")
bx.set_aspect("equal")
bx.set_xlim(-2.6, 3.9)
bx.set_ylim(-1.3, 1.55)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")

dgfig.save(fig, __file__)

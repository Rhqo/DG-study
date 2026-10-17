"""Figure 32.2.2: two normal motions of the ellipse (2 cos t, sin t), counterclockwise, n_s pointing inward.
(a) gamma + eps kappa_s n_s (speed proportional to curvature, Proposition 32.2.3); (b) the offset curves
gamma + eps n_s (constant speed, Proposition 32.2.4), regular for |eps| < 1/max kappa_s = 1/2.

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-2-2-variation-offset.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg32  # noqa: E402

C = dgfig.COLORS
J = ddg32.J
m = 4000
t = 2 * np.pi * np.arange(m + 1) / m
P = ddg32.point("ellipse", t)
dP = np.stack([-2 * np.sin(t), np.cos(t)], 1)
T = dP / np.linalg.norm(dP, axis=1, keepdims=True)
N = T @ J.T                                   # n_s = J t (inward for this counterclockwise ellipse)
k = ddg32.kappa("ellipse", t)
assert np.all((N * (-P)).sum(1) > 0)          # n_s points toward the inside (the origin side)
assert abs(k.max() - 2) < 1e-9 and abs(k.min() - 0.25) < 1e-9


def length(Q):
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())


L = ddg32.smooth_length("ellipse")
assert abs(L - 9.688448220547675) < 1e-9
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.6))

# (a) gamma + eps kappa n
ax.plot(P[:, 0], P[:, 1], color=C["main"], lw=dgfig.LW["main"])
for eps, ls in ((0.1, "--"), (0.2, ":")):
    Q = P + eps * k[:, None] * N
    ax.plot(Q[:, 0], Q[:, 1], color=C["accent"], lw=1.4, ls=ls)
for tt in np.linspace(0, 2 * np.pi, 13)[:-1]:
    i = int(round(tt / (2 * np.pi) * m))
    v = 0.2 * k[i] * N[i]
    if np.linalg.norm(v) > 0.04:
        ax.annotate("", xy=P[i] + v, xytext=P[i], arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.2,
                                                                  mutation_scale=8))
ax.text(0, 1.22, r"$\gamma + \varepsilon\,\kappa_s\mathbf{n}_s$", ha="center", fontsize=11, color=C["accent"])
ax.set_aspect("equal")
ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-1.2, 1.5)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

# (b) offsets gamma + eps n
bx.plot(P[:, 0], P[:, 1], color=C["main"], lw=dgfig.LW["main"])
for eps in (-0.4, -0.2, 0.2, 0.4):
    Q = P + eps * N
    assert abs(length(Q) - (L - 2 * np.pi * eps)) < 1e-4      # Proposition 32.2.4 (exact for |eps| < 1/2)
    bx.plot(Q[:, 0], Q[:, 1], color=C["tangent"], lw=1.0, ls="--")
Q = P + 0.7 * N                                                # beyond 1/max kappa: cusps (swallowtails)
assert np.any(1 - 0.7 * k < 0)
bx.plot(Q[:, 0], Q[:, 1], color=C["accent"], lw=1.3)
bx.text(0, 1.52, r"$\gamma + \varepsilon\,\mathbf{n}_s$", ha="center", fontsize=11, color=C["tangent"])
bx.annotate(r"$\varepsilon = 0.7$", xy=(1.18, 0.0), xytext=(1.55, -0.95), fontsize=10, color=C["accent"],
            arrowprops=dict(arrowstyle="-", color=C["accent"], lw=0.7))
bx.set_aspect("equal")
bx.set_xlim(-2.6, 2.6)
bx.set_ylim(-1.5, 1.8)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12)

fig.tight_layout()
dgfig.save(fig, __file__)

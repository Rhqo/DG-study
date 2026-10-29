"""Figure 46.3.2: the tangential divergence div_P X is the rate at which a short piece with direction P is stretched.

Linear vector field X(x) = A x with A = [[0.8, 0.5], [-0.1, -0.4]] (div X = tr A = 0.4).
(a) unit segments through the origin in six directions (solid) and their images under the flow exp(tA) at t = 0.5
(dashed): some are stretched, some shrink.
(b) the stretching rate d/dt |exp(tA) u| at t = 0, which equals <A u, u> = tr(P(theta) A) = div_P X, against the angle
theta of the line; the dashed line is its average tr(A)/2 = 0.2, the dots are the six directions of (a).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-3-2-stretching.py
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
A = np.array([[0.8, 0.5], [-0.1, -0.4]])


def expm(M, t):
    """exp(t M) by eigen-decomposition (A is diagonalisable with real eigenvalues)."""
    w, Q = np.linalg.eig(M)
    return (Q @ np.diag(np.exp(t * w)) @ np.linalg.inv(Q)).real


def rate(theta):
    return np.array([v.tangential_div(P, A) for P in v.line_proj(np.atleast_1d(theta))])


# self-checks: rate = <Au, u> = 0.2 + 0.6 cos 2 theta + 0.2 sin 2 theta; average = tr A / 2; finite differences agree
th = np.linspace(0, np.pi, 721)
r = rate(th)
assert np.allclose(r, 0.2 + 0.6 * np.cos(2 * th) + 0.2 * np.sin(2 * th))
assert abs(np.mean(r[:-1]) - 0.5 * np.trace(A)) < 1e-12
for t0 in (0.0, 0.7, 2.0):
    u = np.array([np.cos(t0), np.sin(t0)])
    h = 1e-6
    fd = (np.linalg.norm(expm(A, h) @ u) - np.linalg.norm(expm(A, -h) @ u)) / (2 * h)
    assert abs(fd - u @ A @ u) < 1e-8

dirs = np.radians([0, 30, 60, 90, 120, 150])
T = 0.5
E = expm(A, T)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.45})
ax = axes[0]
for t0 in dirs:
    u = np.array([np.cos(t0), np.sin(t0)])
    w = E @ u
    grows = u @ A @ u > 0
    col = C["normal"] if grows else C["tangent"]
    ax.plot([-u[0], u[0]], [-u[1], u[1]], color=col, lw=1.6)
    ax.plot([-w[0], w[0]], [-w[1], w[1]], color=col, lw=1.0, ls="--")
ax.plot(0, 0, "o", color=C["main"], ms=3)
ax.set_aspect("equal")
lim = 1.6
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_axis_off()
ax.set_title(r"(a) segments, $t = 0.5$", fontsize=12)

ax = axes[1]
ax.plot(np.degrees(th), r, color=C["main"], lw=1.6)
ax.axhline(0.5 * np.trace(A), color=C["aux"], lw=0.8, ls="--")
ax.axhline(0.0, color=C["aux"], lw=0.5)
rd = rate(dirs)
ax.scatter(np.degrees(dirs), rd, c=[C["normal"] if x > 0 else C["tangent"] for x in rd], s=28, zorder=3)
ax.text(176, 0.5 * np.trace(A) - 0.04, r"$\frac{1}{2}\,\mathrm{tr}\,A$", fontsize=11, ha="right", va="top")
ax.set_xlabel(r"angle $\theta$ of the line (degrees)")
ax.set_ylabel(r"$\mathrm{div}_P X = \langle Au, u\rangle$")
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlim(0, 180)
ax.set_title(r"(b) stretching rate", fontsize=12)

dgfig.save(fig, __file__)
print("rates at 0..150:", np.round(rd, 4), " range:", round(r.min(), 4), round(r.max(), 4))

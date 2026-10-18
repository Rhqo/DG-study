"""Figure 38.1.1: the differential of the torus of revolution (R, r) = (2, 0.8) maps a circle of directions to an
ellipse in the tangent plane.

(a) parameter domain (u, v) in [-pi, pi] x [-pi, pi] with two points q1 = (0, -1.0) (outer equator) and
q2 = (2 pi / 3, -0.25); at each point a circle of radius rho = 0.45 of vectors X and one vector X at 30 degrees.
(b) the torus with the images: x(q) + dx_q(rho (cos a, sin a)), an ellipse with semi-axes rho |x_u| = 0.8 rho
(meridian direction) and rho |x_v| = rho (R + r cos u) (2.8 rho at q1, 1.6 rho at q2), and the vector dx_q(X).

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-1-1-differential-ellipses.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg38  # noqa: E402

C = dgfig.COLORS
R, r = ddg38.R_TORUS, ddg38.r_TORUS
rho = 0.45
alpha = np.deg2rad(30.0)
Q = [(0.0, -1.0), (2 * np.pi / 3, -0.25)]
labels = [r"$q_1$", r"$q_2$"]


def jac(u, v):
    """columns x_u, x_v (exact)."""
    xu = np.array([-r * np.sin(u) * np.cos(v), -r * np.sin(u) * np.sin(v), r * np.cos(u)])
    xv = np.array([-(R + r * np.cos(u)) * np.sin(v), (R + r * np.cos(u)) * np.cos(v), 0.0])
    return np.stack([xu, xv], 1)


# self-checks: exact partials agree with finite differences; semi-axes are r and R + r cos u
for (u, v) in Q:
    J = jac(u, v)
    xu, xv = ddg38.partials(ddg38.torus, np.array(u), np.array(v))
    assert np.allclose(J[:, 0], xu, atol=1e-7) and np.allclose(J[:, 1], xv, atol=1e-7)
    sv = np.linalg.svd(J, compute_uv=False)
    assert np.allclose(sorted(sv), sorted([r, R + r * np.cos(u)]))
    assert abs(J[:, 0] @ J[:, 1]) < 1e-12                       # F = 0: the ellipse axes are x_u and x_v
assert np.isclose(R + r * np.cos(Q[0][0]), 2.8) and np.isclose(R + r * np.cos(Q[1][0]), 1.6)

fig = plt.figure(figsize=(7.2, 3.7))
ax = fig.add_axes([0.02, 0.10, 0.36, 0.80])
uu = np.linspace(-np.pi, np.pi, 2)
ax.fill_between([-np.pi, np.pi], -np.pi, np.pi, color=C["region"], alpha=0.25, lw=0)
for t in np.arange(-np.pi, np.pi + 1e-9, np.pi / 4):
    ax.plot([t, t], [-np.pi, np.pi], color=C["aux"], lw=0.3)
    ax.plot([-np.pi, np.pi], [t, t], color=C["aux"], lw=0.3)
ax.add_patch(plt.Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi, fill=False, ec=C["main"], lw=0.8))
a = np.linspace(0, 2 * np.pi, 200)
for (u, v), lab in zip(Q, labels):
    vv = v
    ax.plot(u + rho * np.cos(a), vv + rho * np.sin(a), color=C["tangent"], lw=1.3)
    ax.annotate("", xy=(u + rho * np.cos(alpha), vv + rho * np.sin(alpha)), xytext=(u, vv),
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=11))
    ax.plot(u, vv, "o", color=C["main"], ms=3)
    ax.text(u - 0.42, vv - 0.62, lab, fontsize=12, ha="right")
    ax.text(u + rho * np.cos(alpha) + 0.08, vv + rho * np.sin(alpha) + 0.02, r"$X$", color=C["tangent"], fontsize=12)
ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
ax.set_aspect("equal")
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$v$", rotation=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", loc="left", fontsize=12)

ax3 = dgfig.axes3d(fig, 111, elev=38, azim=-62)
ax3.set_position([0.40, 0.02, 0.60, 0.96])
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, 2 * np.pi, 97), indexing="ij")
X, Y, Z = ddg38.torus(U, V)
dgfig.surface(ax3, X, Y, Z, alpha=0.30, grid_every=4)
for (u, v), lab in zip(Q, labels):
    p = ddg38.torus(u, v)
    J = jac(u, v)
    E = p[:, None] + J @ (rho * np.stack([np.cos(a), np.sin(a)]))
    ax3.plot(*E, color=C["tangent"], lw=1.6, zorder=8)
    dgfig.arrow3d(ax3, p, J @ (rho * np.array([np.cos(alpha), np.sin(alpha)])), role="tangent", zorder=9, head=9)
    dgfig.point3d(ax3, p, None, size=10)
ax3.text(*(ddg38.torus(*Q[0]) + np.array([0.1, -0.45, -0.42])), r"$\mathbf{x}(q_1)$", fontsize=12)
ax3.text(*(ddg38.torus(*Q[1]) + np.array([-0.35, -0.2, 0.42])), r"$\mathbf{x}(q_2)$", fontsize=12)
p1 = ddg38.torus(*Q[0])
tip = p1 + jac(*Q[0]) @ (rho * np.array([np.cos(alpha), np.sin(alpha)]))
ax3.text(*(tip + np.array([0.25, 0.1, 0.15])), r"$d\mathbf{x}(X)$", color=C["tangent"], fontsize=12)
dgfig.equal_aspect(ax3, np.stack([X, Y, Z], -1), zoom=1.38)
ax3.set_title("(b)", loc="left", fontsize=12, y=0.92)
dgfig.map_arrow(fig, ax, ax3, r"$\mathbf{x}$", xy_from=(0.98, 0.70), xy_to=(0.10, 0.70), rad=-0.3)
dgfig.save(fig, __file__)

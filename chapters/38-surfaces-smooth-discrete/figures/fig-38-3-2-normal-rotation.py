"""Figure 38.3.2: N x dx rotates tangent vectors by +90 degrees; on 1-forms the Hodge star is star(dx) = -N x dx.

Saddle x(u, v) = (u, v, u^2 - v^2) at q = (0.5, -0.4): g = [[2, 0.8], [0.8, 1.64]], N = (-1, -0.8, 1)/sqrt(2.64).
X = (1, 0) (drawn at half length). X' is the g-rotation of X by +90 degrees: g(X', X') = g(X, X), g(X, X') = 0,
(X, X') positively oriented; X' = (-0.4 b, b), b = sqrt(2 / 1.32). Proposition 38.3.6: dx(X') = N x dx(X) and
(star dx)(X) = -dx(X') = -N x dx(X).
(a) the parameter plane near q: X (blue), X' (green), the Euclidean rotation of X (gray dashed), and the curve
{Y : g(Y, Y) = g(X, X)} (light blue). (b) the saddle near x(q): dx(X) (blue), N (vermilion), N x dx(X) (green),
(star dx)(X) (purple), all at half length, and the tangent plane.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-3-2-normal-rotation.py
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
q = np.array([0.5, -0.4])
J = np.array([[1.0, 0.0], [0.0, 1.0], [2 * q[0], -2 * q[1]]])       # columns x_u, x_v
g = J.T @ J
N = np.cross(J[:, 0], J[:, 1])
N = N / np.linalg.norm(N)
X = np.array([1.0, 0.0])
b = np.sqrt(2 / 1.32)
Xp = np.array([-0.4 * b, b])

# self-checks
assert np.allclose(g, [[2, 0.8], [0.8, 1.64]]) and np.allclose(N, np.array([-1, -0.8, 1]) / np.sqrt(2.64))
assert abs(X @ g @ Xp) < 1e-12 and abs(Xp @ g @ Xp - X @ g @ X) < 1e-12 and np.linalg.det(np.stack([X, Xp], 1)) > 0
assert np.allclose(J @ Xp, np.cross(N, J @ X))                       # Proposition 38.3.6(a)
xu, xv = ddg38.partials(ddg38.saddle, np.array(q[0]), np.array(q[1]))
assert np.allclose(np.stack([xu, xv], 1), J, atol=1e-8)

fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_axes([0.04, 0.12, 0.40, 0.80])
sc = 0.5
ax.plot(*q, "o", color=C["main"], ms=4)
w, Q = np.linalg.eigh(g)
a = np.linspace(0, 2 * np.pi, 200)
E = q[:, None] + sc * (Q @ (np.sqrt(X @ g @ X) * np.stack([np.cos(a), np.sin(a)]) / np.sqrt(w)[:, None]))
ax.plot(*E, color=C["region"], lw=1.2)
for vec, col, ls, lab, off in ((X, C["tangent"], "-", r"$X$", (0.03, -0.09)),
                               (Xp, C["third"], "-", r"$X'$", (-0.2, 0.0)),
                               (np.array([0.0, 1.0]), C["aux"], "--", r"$90^\circ$", (0.03, 0.0))):
    tip = q + sc * vec
    ax.annotate("", xy=tip, xytext=q, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, ls=ls, mutation_scale=11))
    ax.text(tip[0] + off[0], tip[1] + off[1], lab, color=col, fontsize=12)
ax.set_aspect("equal")
ax.set_xlim(q[0] - 0.85, q[0] + 0.85)
ax.set_ylim(q[1] - 0.62, q[1] + 0.82)
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$v$", rotation=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", loc="left", fontsize=12)

ax3 = dgfig.axes3d(fig, 111, elev=12, azim=-112)
ax3.set_position([0.44, 0.0, 0.56, 1.0])
U, V = np.meshgrid(np.linspace(q[0] - 0.55, q[0] + 0.55, 23), np.linspace(q[1] - 0.55, q[1] + 0.55, 23), indexing="ij")
Xs, Ys, Zs = ddg38.saddle(U, V)
dgfig.surface(ax3, Xs, Ys, Zs, alpha=0.30, grid_every=4)
p = ddg38.saddle(*q)
dgfig.tangent_plane(ax3, p, J[:, 0], J[:, 1] - (J[:, 1] @ J[:, 0]) / (J[:, 0] @ J[:, 0]) * J[:, 0], size=0.55)
vecs = [(J @ X, "tangent", r"$d\mathbf{x}(X)$", (0.05, 0.0, -0.08)),
        (N, "normal", r"$\mathbf{N}$", (0.0, 0.0, 0.08)),
        (np.cross(N, J @ X), "third", r"$\mathbf{N}\times d\mathbf{x}(X)$", (-0.25, 0.0, 0.1)),
        (-np.cross(N, J @ X), "covector", r"$(\star d\mathbf{x})(X)$", (0.0, 0.05, -0.12))]
for vec, role, lab, off in vecs:
    dgfig.arrow3d(ax3, p, 0.5 * vec, role=role, label=lab, label_offset=off, zorder=10, head=9, fontsize=11)
dgfig.point3d(ax3, p, None, size=10)
dgfig.equal_aspect(ax3, np.stack([Xs, Ys, Zs], -1), zoom=1.15)
ax3.set_title("(b)", loc="left", fontsize=12, y=0.93)
dgfig.save(fig, __file__)

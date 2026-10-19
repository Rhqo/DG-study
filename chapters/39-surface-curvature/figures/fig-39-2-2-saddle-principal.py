"""Figure 39.2.2: principal directions of the saddle are orthogonal in R^3 but not in the parameter plane.

Saddle x(u, v) = (u, v, u^2 - v^2), upward N, point q = (0.5, -0.4) (Examples 39.1.4 and 39.2.9).
k1 = 0.6783 (blue), k2 = -0.8461 (green). Lines of curvature: integral curves of the two principal line fields on
[-1, 1]^2 (RK2), drawn in the parameter plane (a) and on the surface (b). At q: X1, X2 (g-unit, drawn at length 0.3
in (a)); dashed gray = the Euclidean perpendicular of X1; light blue curve = {Y : g_q(Y, Y) = 0.09}. In (b):
dx(X1), dx(X2) at length 0.4 and N (vermilion, length 0.4).

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-2-2-saddle-principal.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg39  # noqa: E402

C = dgfig.COLORS
q = np.array(ddg39.SADDLE_Q)


def derivs(x, y):
    return 2 * x, -2 * y, 2.0, 0.0, -2.0


k1, k2, X1, X2, g = ddg39.principal(ddg39.saddle, *q)
assert abs(k1 - 0.678284) < 1e-5 and abs(k2 + 0.846136) < 1e-5
assert abs(X1 @ g @ X2) < 1e-8 and abs(X1 @ g @ X1 - 1) < 1e-8
cosE = X1 @ X2 / np.linalg.norm(X1) / np.linalg.norm(X2)
angE = np.degrees(np.arccos(abs(cosE)))
assert 60 < angE < 70                                   # Euclidean angle in the parameter plane is not 90 degrees
xu, xv = ddg39.partials(ddg39.saddle, *q)
D = np.column_stack([xu, xv])
assert abs((D @ X1) @ (D @ X2)) < 1e-8                  # orthogonal in R^3
# graph principal directions agree with the general routine
gk1, gk2, gX1, gX2 = ddg39.graph_principal(derivs, *q)
assert abs(gk1 - k1) < 1e-6 and abs(abs(gX1 @ X1 / np.linalg.norm(X1)) - 1) < 1e-6

lines = {0: [], 1: []}
for which, seeds in ((0, [(0.0, s) for s in np.linspace(-0.9, 0.9, 7)]), (1, [(s, 0.0) for s in np.linspace(-0.9, 0.9, 7)])):
    for sd in seeds + [tuple(q)]:
        segs = [ddg39.trace_line(derivs, sd, which, step=0.01, nmax=400, box=1.0, direction=dd) for dd in (1, -1)]
        lines[which].append((np.vstack([segs[1][::-1], segs[0][1:]]), sd == tuple(q)))

fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_subplot(121)
cols = {0: C["tangent"], 1: C["third"]}
for which in (0, 1):
    for L, through_q in lines[which]:
        ax.plot(L[:, 0], L[:, 1], color=cols[which], lw=1.6 if through_q else 0.6, alpha=1.0 if through_q else 0.6)
tt = np.linspace(0, 2 * np.pi, 200)
Ci = np.linalg.inv(np.linalg.cholesky(g))
ell = 0.3 * (Ci.T @ np.stack([np.cos(tt), np.sin(tt)])).T
ax.plot(q[0] + ell[:, 0], q[1] + ell[:, 1], color=C["region"], lw=1.2)
for X, col, lab, off in ((X1, C["tangent"], r"$X_1$", (0.03, 0.04)), (X2, C["third"], r"$X_2$", (0.05, 0.0))):
    ax.annotate("", xy=q + 0.3 * X, xytext=q, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=12,
                                                             shrinkA=0, shrinkB=0))
    ax.text(*(q + 0.3 * X + np.array(off)), lab, color=col, fontsize=12)
perp = np.array([-X1[1], X1[0]]) / np.linalg.norm(X1) * 0.3
ax.plot([q[0], q[0] + perp[0]], [q[1], q[1] + perp[1]], color=C["aux"], lw=1.0, ls=(0, (3, 2)))
ax.plot(*q, "o", color=C["main"], ms=4)
ax.text(q[0] + 0.05, q[1] - 0.12, r"$q$", fontsize=12)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$v$", rotation=0)
ax.set_title("(a)", loc="left", fontsize=12)

ax = dgfig.axes3d(fig, 122, elev=22, azim=-100)
U, V = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41), indexing="ij")
dgfig.surface(ax, U, V, U ** 2 - V ** 2, alpha=0.35, grid=False)
for which in (0, 1):
    for L, through_q in lines[which]:
        P = ddg39.saddle(L[:, 0], L[:, 1]).T
        ax.plot(*P.T, color=cols[which], lw=1.6 if through_q else 0.6, alpha=1.0 if through_q else 0.6, zorder=5 if through_q else 3)
p = ddg39.saddle(*q)
n = ddg39.unit_normal(ddg39.saddle, *q)
dgfig.arrow3d(ax, p, 0.55 * D @ X1, role="tangent", zorder=12, lw=1.8)
dgfig.arrow3d(ax, p, 0.55 * D @ X2, role="third", zorder=12, lw=1.8)
dgfig.arrow3d(ax, p, 0.6 * n, role="normal", zorder=12, lw=1.6)
ax.text(*(p + 0.62 * D @ X1 + np.array([0.0, 0.0, -0.17])), r"$d\mathbf{x}(X_1)$", color=C["tangent"], fontsize=11)
ax.text(*(p + 0.62 * D @ X2 + np.array([0.05, 0.0, 0.05])), r"$d\mathbf{x}(X_2)$", color=C["third"], fontsize=11)
ax.text(*(p + 0.65 * n + np.array([0, 0, 0.04])), r"$\mathbf{N}$", color=C["normal"], fontsize=12)
dgfig.equal_aspect(ax, np.stack([U.ravel(), V.ravel(), (U ** 2 - V ** 2).ravel()], 1), zoom=1.25)
ax.set_title("(b)", loc="left", fontsize=12, y=0.96)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.92, wspace=0.08)
dgfig.save(fig, __file__)
print(f"k1 {k1:.6f} k2 {k2:.6f} X1 {X1} X2 {X2} Euclidean angle {angE:.2f}")

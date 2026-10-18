"""Figure 36.5.1: a primal 1-form measures circulation along an edge, a dual 1-form measures flux across the dual
edge (Section 36.5).

Two counterclockwise triangles sharing the edge e = [x_i, x_j]: x_i = (0, 0), x_j = (2, 0), f = [x_i, x_j, x_k] with
x_k = (1.05, 2.0) above, g = [x_j, x_i, x_l] with x_l = (0.9, -1.9) below. Constant vector field X = (0.8, 0.5),
i.e. the 1-form alpha = X^flat = 0.8 dx + 0.5 dy.
(a) the primal edge e (blue, oriented i -> j) and the field X (gray arrows); the value of the primal 1-form is the
    circulation int_e alpha = <X, x_j - x_i> = 1.6.
(b) the circumcentric dual edge e* (orange polyline c_g -> m_e -> c_f, oriented from the right face g to the left
    face f, i.e. along e turned by +90 degrees); the value of the dual 1-form star1 alpha is the flux of X across e*,
    int_{e*} star alpha = <X, t> |e*| (t = unit tangent of e, the normal of e*).
Self-check: e* is perpendicular to e; |e*| / |e| = (cot alpha + cot beta) / 2 = dgdec.hodge1; the flux equals
star1 x circulation (exact for a constant form, Theorem 36.5.16).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-5-1-circulation-flux.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

V = np.array([[0.0, 0.0], [2.0, 0.0], [1.05, 2.0], [0.9, -1.9]])
F = np.array([[0, 1, 2], [1, 0, 3]])
E, index = dd.edge_list(F)
e = index[(0, 1)]
X = np.array([0.8, 0.5])
J = np.array([[0.0, -1.0], [1.0, 0.0]])
t = (V[1] - V[0]) / np.linalg.norm(V[1] - V[0])
C = dd.circumcenters(V, F)
start, mid, end = ch36.dual_edge_polylines(V, F)[e]
assert np.allclose(start, C[1]) and np.allclose(end, C[0])           # from g (right) to f (left)
assert abs((end - mid) @ t) < 1e-12 and abs((start - mid) @ t) < 1e-12   # perpendicular to e
dual_len = (end - mid) @ (J @ t) + (mid - start) @ (J @ t)               # signed, along +Jt
star1 = dd.hodge1(V, F)[e]
assert abs(dual_len / 2.0 - star1) < 1e-12
circulation = X @ (V[1] - V[0])
star_alpha = lambda v: X @ (-J @ v)  # noqa: E731   (star alpha)(v) = alpha(-J v) in R^2
flux = star_alpha(mid - start) + star_alpha(end - mid)
assert abs(flux - star1 * circulation) < 1e-12 and abs(circulation - 1.6) < 1e-12
print(f"circulation {circulation:.3f}, |e*| {dual_len:.4f}, star1 {star1:.4f}, flux {flux:.4f}")

col = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 4.0))
gx, gy = np.meshgrid(np.linspace(-0.2, 2.2, 7), np.linspace(-2.0, 2.0, 9))
for ax in (axa, axb):
    for tri in F:
        ax.add_patch(Polygon(V[tri], closed=True, facecolor="#F4F4F4", edgecolor="#BBBBBB", lw=0.8, zorder=0))
    for x0, y0 in zip(gx.ravel(), gy.ravel()):
        ax.add_patch(FancyArrowPatch((x0, y0), (x0 + 0.28 * X[0], y0 + 0.28 * X[1]), arrowstyle="-|>",
                                     mutation_scale=7, color="#A0A0A0", lw=0.9, zorder=1))
    for v, off, name in ((0, (-0.15, -0.1), r"$x_i$"), (1, (0.15, -0.1), r"$x_j$")):
        ax.text(*(V[v] + np.array(off)), name, fontsize=11, ha="center")
    ax.text(*(V[F[0]].mean(0) + np.array([-0.25, 0.25])), r"$f$", fontsize=12)
    ax.text(*(V[F[1]].mean(0) + np.array([-0.35, -0.3])), r"$g$", fontsize=12)
ax = axa
ax.add_patch(FancyArrowPatch(V[0], V[1], arrowstyle="-|>", mutation_scale=16, color=col["tangent"], lw=2.6, zorder=4))
ax.text(1.0, -0.22, r"$e$", fontsize=13, color=col["tangent"], ha="center")
ax.set_title(r"(a) primal: $\int_e\alpha = \langle X, x_j - x_i\rangle$", fontsize=11)
ax.text(1.0, -2.4, rf"circulation $= {circulation:.2f}$", fontsize=11, ha="center", color=col["tangent"])
ax = axb
ax.plot(*V[[0, 1]].T, color=col["tangent"], lw=1.2, zorder=3)
P = np.array([start, mid, end])
ax.plot(*P.T, color=col["accent"], lw=2.6, zorder=4)
ax.add_patch(FancyArrowPatch(mid, mid + 0.6 * (end - mid), arrowstyle="-|>", mutation_scale=16, color=col["accent"],
                             lw=2.6, zorder=5))
for p, name, off in ((start, r"$c_g$", (-0.35, -0.05)), (end, r"$c_f$", (-0.35, -0.05)), (mid, r"$m_e$", (0.1, -0.28))):
    ax.plot(*p, "o", color=col["accent"], ms=4, zorder=6)
    ax.text(*(p + np.array(off)), name, fontsize=11)
for s in np.linspace(0.12, 0.88, 5):
    q = start + s * (end - start)
    ax.add_patch(FancyArrowPatch(q - 0.18 * t, q + 0.22 * t, arrowstyle="-|>", mutation_scale=9, color=col["normal"],
                                 lw=1.4, zorder=6))
ax.text(*(end + np.array([0.12, -0.35])), r"$e^\star$", fontsize=13, color=col["accent"])
ax.set_title(r"(b) dual: $\int_{e^\star}\star\alpha = \langle X, t\rangle\,|e^\star|$", fontsize=11)
ax.text(1.0, -2.4, rf"flux $= {flux:.2f} = \star_1 \times {circulation:.1f}$", fontsize=11, ha="center",
        color=col["accent"])
for ax in (axa, axb):
    ax.set_xlim(-0.35, 2.4)
    ax.set_ylim(-2.55, 2.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

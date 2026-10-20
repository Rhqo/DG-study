"""Figure 42.4.1: (L u)_i is the flux of the piecewise linear gradient out of the circumcentric dual cell (Section 42.4,
Proposition 42.4.1).

A planar star: center x_0 = (0, 0) and six neighbours; the triangles (0, 1, 2) and (0, 6, 1) have obtuse angles at x_2
and x_6 (115 and 127 degrees), so edge 01 is non-Delaunay (L_01 < 0) and the two pieces of the dual cell C_0 next to
edge 01 have negative signed area (hatched vermilion).
Values u = W_0 u with u_i = x_i^2 + 0.6 y_i (vertex values).
(a) the star (gray), the circumcentric dual cell C_0 (orange polygon through edge midpoints and circumcenters; the
    triangle pieces with negative signed area are hatched), and the gradient of W_0 u on every face (blue, scaled 0.25).
(b) the six edge terms L_0j (u_j - u_0) (bars; blue if L_0j > 0, vermilion if L_0j < 0) and their sum (L u)_0 (black
    line), compared with the flux computed segment by segment along the boundary of C_0 (orange dot).
Self-check: the sum of the segment fluxes (int over [m, c] of <grad u, n> with signed lengths) equals (L u)_0 to 1e-14,
the cell area equals dgdec.dual_areas(..., "circumcentric")[0].

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-4-1-dual-cell-flux.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

col = dgfig.COLORS
nb = np.array([[1.0, 0.0], [0.35, 0.3], [-0.55, 0.85], [-1.05, -0.1], [-0.4, -0.95], [0.55, -0.25]])
V = np.c_[np.vstack([[0.0, 0.0], nb]), np.zeros(7)]
F = np.array([[0, k + 1, (k + 1) % 6 + 1] for k in range(6)])
assert np.all(dm.face_cross(V, F)[:, 2] > 0)
ang = dm.corner_angles(V, F)
assert ang[0, 2] > np.pi / 2 and ang[5, 1] > np.pi / 2   # obtuse at x_2 in (0, 1, 2) and at x_6 in (0, 6, 1)
assert dm.cotan_laplacian(V, F)[0, 1] < 0             # so edge 01 is non-Delaunay
u = V[:, 0] ** 2 + 0.6 * V[:, 1]
L = dm.cotan_laplacian(V, F)
C = dd.circumcenters(V, F)
J = np.array([[0.0, -1.0], [1.0, 0.0]])

# boundary of C_0 as the oriented chain sum_f [m_0j, c_f] + [c_f, m_0k]; flux of the constant face gradient
flux_total, pieces = 0.0, []
for f, (i, j, k) in enumerate(F):
    P = V[[i, j, k]][:, :2]
    g = dd.barycentric_gradients(V[[i, j, k]])[:, :2]
    grad = u[[i, j, k]] @ g
    mij, mik, c = 0.5 * (P[0] + P[1]), 0.5 * (P[0] + P[2]), C[f, :2]
    for a, b in ((mij, c), (c, mik)):
        seg = b - a
        flux_total += grad @ (-J @ seg)            # the outward side of a counterclockwise boundary is on the right
    pieces.append((P[0], mij, c, mik, grad, f))
assert abs(flux_total - (L @ u)[0]) < 1e-14
areas = [0.5 * ((m1 - x0)[0] * (c - x0)[1] - (m1 - x0)[1] * (c - x0)[0]) + 0.5 * ((c - x0)[0] * (m2 - x0)[1] - (c - x0)[1] * (m2 - x0)[0])
         for x0, m1, c, m2, _, _ in pieces]
assert abs(sum(areas) - dd.dual_areas(V, F, "circumcentric")[0]) < 1e-14
terms = [L[0, j] * (u[j] - u[0]) for j in range(1, 7)]
print("cotan weights L_0j:", np.round(L[0, 1:], 3), " (L u)_0 =", round(float((L @ u)[0]), 4),
      " cell area:", round(float(sum(areas)), 4), " Delta u = 2, A_0 * 2 =", round(2 * float(sum(areas)), 4))

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.3), gridspec_kw={"width_ratios": [1.15, 1.0]})
for f in F:
    axa.fill(*V[f][:, :2].T, color="#EEEEEE", zorder=0)
for (a, b) in dm.edges(F):
    axa.plot(*V[[a, b]][:, :2].T, color="#888888", lw=0.8, zorder=1)
for x0, m1, c, m2, grad, f in pieces:
    for tri in ((x0, m1, c), (x0, c, m2)):
        T = np.array(tri)
        sgn = (T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0]
        if sgn > 0:
            axa.add_patch(Polygon(T, closed=True, facecolor=col["accent"], alpha=0.35, edgecolor="none", zorder=2))
        else:
            axa.add_patch(Polygon(T, closed=True, facecolor="none", hatch="////", edgecolor=col["normal"], lw=0.0,
                                  zorder=2))
    axa.plot([m1[0], c[0], m2[0]], [m1[1], c[1], m2[1]], color=col["accent"], lw=1.5, zorder=3)
    cen = V[F[f]][:, :2].mean(0)
    axa.annotate("", xy=cen + 0.25 * grad, xytext=cen, arrowprops=dict(arrowstyle="->", color=col["tangent"], lw=1.2), zorder=4)
axa.plot(*C[:, :2].T, "o", color=col["accent"], ms=3, zorder=5)
axa.plot(V[:, 0], V[:, 1], "o", color=col["main"], ms=3.5, zorder=6)
for idx in range(7):
    off = (0.05, 0.05) if idx else (-0.17, -0.13)
    axa.text(V[idx, 0] + off[0], V[idx, 1] + off[1], rf"${idx}$", fontsize=10, zorder=7)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title(r"(a) dual cell $C_0$ and $\mathrm{grad}\, W_0u$", fontsize=11)

xs = np.arange(1, 7)
axb.bar(xs, terms, color=[col["tangent"] if L[0, j] > 0 else col["normal"] for j in range(1, 7)], width=0.6)
axb.set_ylim(min(min(terms), 0) - 0.15, max(max(terms), (L @ u)[0]) + 0.25)
axb.axhline((L @ u)[0], color=col["main"], lw=1.2)
axb.axhline(0, color="#999999", lw=0.6)
axb.plot([7], [flux_total], "o", color=col["accent"], ms=6)
axb.text(4.6, (L @ u)[0] + 0.05, rf"$(Lu)_0 = {(L @ u)[0]:.3f}$", fontsize=9.5)
axb.set_xticks(list(xs) + [7], [rf"${j}$" for j in xs] + ["flux"])
axb.set_xlabel(r"neighbor $j$")
axb.set_ylabel(r"$L_{0j}\,(u_j - u_0)$")
axb.set_title("(b) edge terms and their sum", fontsize=11)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.1, wspace=0.25)
dgfig.save(fig, __file__)

"""Figure 36.6.2: a Poisson problem solved with the cotan Laplacian (Section 36.6, Implementation 36.6).

Problem: Delta u = f on the unit square with u = 0 on the boundary, f = -2 pi^2 sin(pi x) sin(pi y), exact solution
u* = sin(pi x) sin(pi y). Discretization: L u = star0 f at the interior vertices of dgmesh.grid_mesh (n x n squares,
alternating diagonals), L = -d0^T star1 d0, star0 the circumcentric dual areas (h^2 at interior vertices).
(a) the discrete solution for n = 8 (color on the triangles, piecewise linear) and its vertices (dots). The
    piecewise-linear coloring is drawn by refining every triangle into 16 small flat pieces carrying the linear
    interpolant (matplotlib's SVG gouraud shading renders as nearly flat triangles in browsers).
(b) largest vertex error |u - u*| for n = 4, 8, 16, 32 (log-log, h = 1/n); the dotted line has slope 2.
Self-check: errors decrease by about 4 per halving of h (fitted slope 2 to within 0.05); the discrete solution is
exact up to rounding for the quadratic-free harmonic test u = x^2 - y^2 (Lu = 0 inside).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-6-2-poisson.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402


def solve(n, exact, rhs):
    xs = np.linspace(0.0, 1.0, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    D0 = dd.d0(F, len(V))
    L = -D0.T @ (dd.hodge1(V, F)[:, None] * D0)
    star0 = dd.hodge0(V, F, "circumcentric")
    inner = (V[:, 0] > 0) & (V[:, 0] < 1) & (V[:, 1] > 0) & (V[:, 1] < 1)
    ue = exact(V)
    u = ue.copy()                                          # boundary values from the exact solution
    b = star0[inner] * rhs(V)[inner] - L[np.ix_(inner, ~inner)] @ ue[~inner]
    u[inner] = np.linalg.solve(L[np.ix_(inner, inner)], b)
    return V, F, u, ue


ex = lambda V: np.sin(np.pi * V[:, 0]) * np.sin(np.pi * V[:, 1])  # noqa: E731
rh = lambda V: -2 * np.pi ** 2 * ex(V)  # noqa: E731
ns = np.array([4, 8, 16, 32])
errs = np.array([np.abs(np.subtract(*solve(int(n), ex, rh)[2:])).max() for n in ns])
slope = np.polyfit(np.log(1.0 / ns), np.log(errs), 1)[0]
assert abs(slope - 2) < 0.05 and np.all(errs[:-1] / errs[1:] > 3.8)
Vh, Fh, uh, ueh = solve(8, lambda V: V[:, 0] ** 2 - V[:, 1] ** 2, lambda V: 0.0 * V[:, 0])
assert np.abs(uh - ueh).max() < 1e-12
print("errors", [f"{e:.3e}" for e in errs], f"slope {slope:.3f}")

V, F, u, ue = solve(8, ex, rh)
col = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.0, 1.1]})
tri = mtri.Triangulation(V[:, 0], V[:, 1], F)
fine, ufine = mtri.UniformTriRefiner(tri).refine_field(u, triinterpolator=mtri.LinearTriInterpolator(tri, u), subdiv=2)
tpc = axa.tripcolor(fine, ufine, shading="flat", cmap="Blues", vmin=0, vmax=1, edgecolors="face", lw=0.2)
axa.triplot(tri, color="#888888", lw=0.4)
axa.plot(V[:, 0], V[:, 1], "o", color="black", ms=2)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title(r"(a) discrete $u$, $n = 8$", fontsize=11)
cb = fig.colorbar(tpc, ax=axa, fraction=0.046, pad=0.03)
cb.set_label(r"$u$", rotation=0, labelpad=6)
axb.loglog(1.0 / ns, errs, "o-", color=col["tangent"], ms=4, lw=1.4, label=r"$\max|u - u^*|$")
axb.loglog(1.0 / ns, 1.6 * errs[-1] * (ns[-1] / ns) ** 2, ":", color="#999999", lw=1.0, label="slope 2")
axb.set_xticks(1.0 / ns, [rf"$1/{n}$" for n in ns])
axb.xaxis.set_minor_formatter(plt.NullFormatter())
axb.set_xlabel(r"spacing $h = 1/n$")
axb.set_ylabel("largest error")
axb.legend(fontsize=9, frameon=False, loc="lower right")
axb.set_title("(b) second-order convergence", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.16, wspace=0.55)
dgfig.save(fig, __file__)

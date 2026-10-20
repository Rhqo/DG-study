"""Verification for Section 42.5 (Boundary Conditions).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-5-boundary-conditions.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch42  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "42-5-boundary-conditions.html"
x, a, b, c0, c1, g0, g1 = sp.symbols("x a b c0 c1 g0 g1", real=True)

# ---------------------------------------------------------------- Example 42.5.3, Figure 42.5.1
u = x ** 2 / 2 + c1 * x + c0
sol = sp.solve([u.subs(x, 0) - a, u.subs(x, 1) - b], [c0, c1])
check("Example 42.5.3(a): Dirichlet gives c0 = a, c1 = b - a - 1/2", sp.simplify(sol[c0] - a) == 0
      and sp.simplify(sol[c1] - (b - a - sp.Rational(1, 2))) == 0)
eqs = [-sp.diff(u, x).subs(x, 0) - g0, sp.diff(u, x).subs(x, 1) - g1]
check("Example 42.5.3(b): Neumann -u'(0) = g0, u'(1) = g1 is solvable iff g0 + g1 = 1 (c0 free)",
      sp.solve(eqs, [c1, g1], dict=True)[0][g1] == 1 - g0 and sp.solve([e.subs({g0: 1, g1: 1}) for e in eqs], [c1]) == [])
un = (x ** 2 - x) / 2
check("Example 42.5.3, Figure 42.5.1: u = (x^2 - x)/2 + c has -u'(0) = u'(1) = 1/2 and u'' = 1",
      sp.diff(un, x, 2) == 1 and -sp.diff(un, x).subs(x, 0) == sp.Rational(1, 2) and sp.diff(un, x).subs(x, 1) == sp.Rational(1, 2))

# ---------------------------------------------------------------- Proposition 42.5.4, Example 42.5.5 on the unit disk
r, t = sp.symbols("r t", positive=True)
X, Y = r * sp.cos(t), r * sp.sin(t)
lap = lambda f: sp.simplify(sp.diff(r * sp.diff(f, r), r) / r + sp.diff(f, t, 2) / r ** 2)  # noqa: E731
check("Example 42.5.5(a): u = x is harmonic with d_r u = cos t on r = 1, and the boundary integral of cos t is 0",
      lap(X) == 0 and sp.simplify(sp.diff(X, r).subs(r, 1) - sp.cos(t)) == 0 and sp.integrate(sp.cos(t), (t, 0, 2 * sp.pi)) == 0)
check("Example 42.5.5(b): u = r^2/2 has Delta u = 2, d_r u = 1; int 2 dA = 2 pi = boundary integral of 1",
      lap(r ** 2 / 2) == 2 and sp.diff(r ** 2 / 2, r).subs(r, 1) == 1
      and sp.integrate(sp.integrate(2 * r, (r, 0, 1)), (t, 0, 2 * sp.pi)) == 2 * sp.pi)
check("Definition 42.5.2 remark: u = 0 and d_r u = 1 together are impossible for the Laplace equation (unique u = 0)",
      sp.diff(sp.Integer(0), r) != 1)

# ---------------------------------------------------------------- Proposition 42.5.6 on the unit disk (weak form with Neumann)
for uf, vf in ((r ** 2 * sp.cos(2 * t) + r ** 3 * sp.sin(t), 1 + r * sp.cos(t)), (r ** 2 / 2, r ** 2 * sp.sin(t) ** 2)):
    gu_gv = sp.diff(uf, r) * sp.diff(vf, r) + sp.diff(uf, t) * sp.diff(vf, t) / r ** 2
    lhs = sp.integrate(sp.integrate(sp.expand(gu_gv * r), (r, 0, 1)), (t, 0, 2 * sp.pi))
    f = lap(uf)
    rhs = -sp.integrate(sp.integrate(sp.expand(f * vf * r), (r, 0, 1)), (t, 0, 2 * sp.pi)) \
        + sp.integrate((sp.diff(uf, r) * vf).subs(r, 1), (t, 0, 2 * sp.pi))
    check(f"Proposition 42.5.6: (42.5.4) holds on the unit disk for u = {uf}", sp.simplify(lhs - rhs) == 0)

# ---------------------------------------------------------------- Proposition 42.5.7 (reduced Dirichlet matrix)
Vj, Fj = dm.grid_mesh(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
rng = np.random.default_rng(7)
I0 = np.setdiff1d(np.arange(len(Vj)), dm.boundary_vertices(Fj))
Vj = Vj.copy()
Vj[I0, :2] += 0.2 / 8 * rng.uniform(-1, 1, (len(I0), 2))
V0, F0 = dm.icosphere(3)
hemi = dm.submesh(V0, F0, V0[F0].mean(1)[:, 2] > 0)[:2]
for name, (V, F) in {"jittered square": (Vj, Fj), "disk mesh": ch42.disk_mesh(10), "hemisphere": hemi}.items():
    L = dm.cotan_laplacian(V, F)
    B = np.array(dm.boundary_vertices(F))
    I = np.setdiff1d(np.arange(len(V)), B)
    lamI = np.linalg.eigvalsh(-L[np.ix_(I, I)])
    check(f"Proposition 42.5.7 on {name}: -L_II is positive definite (min eigenvalue {lamI[0]:.3e}); negative weights "
          f"{(dd.hodge1(V, F) < 0).sum()}", lamI[0] > 1e-8)
    try:
        np.linalg.cholesky(-L[np.ix_(I, I)])
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
    check(f"Proposition 42.5.7 on {name}: Cholesky succeeds", chol_ok)

# ---------------------------------------------------------------- Proposition 42.5.8, 42.5.9, Figure 42.5.2
Vc = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.45, 0.85, 0], [-0.5, 0.8, 0], [-1.0, 0, 0]])
Fc = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
Lc = dm.cotan_laplacian(Vc, Fc)
uc = 2 * Vc[:, 0] + 3 * Vc[:, 1]
C = dd.circumcenters(Vc, Fc)
flux = 0.0
for f, (i, j, k) in enumerate(Fc):
    g = (uc[[i, j, k]] @ dd.barycentric_gradients(Vc[[i, j, k]]))[:2]
    for p0, p1 in ((0.5 * (Vc[i] + Vc[j]), C[f]), (C[f], 0.5 * (Vc[i] + Vc[k]))):
        s = (p1 - p0)[:2]
        flux += g @ np.array([s[1], -s[0]])
close("Proposition 42.5.9(a), Figure 42.5.2: dual-edge flux = (L u)_0 = 3 for u = 2x + 3y", [flux, (Lc @ uc)[0]], [3.0, 3.0],
      tol=1e-13)
close("Figure 42.5.2: boundary pieces carry -3, total 0 = int Delta u", flux + np.array([2.0, 3.0]) @ [0.0, -1.0], 0.0, tol=1e-13)
check("Figure 42.5.2: all faces acute", dm.corner_angles(Vc, Fc).max() < np.pi / 2)
# Proposition 42.5.9 on a whole disk mesh: (L u)_i + int over the half boundary edges of <grad u, n> = 0 for affine u
V, F = ch42.disk_mesh(8)
L = dm.cotan_laplacian(V, F)
ua = 1.3 * V[:, 0] - 0.7 * V[:, 1] + 0.2
bg = ch42.boundary_load(V, F, lambda p, n: np.array([1.3, -0.7, 0.0]) @ n)
close("Proposition 42.5.9(b): for affine u, L u + b_g = 0 = M Delta u at every vertex (interior and boundary)", L @ ua + bg,
      np.zeros(len(V)), tol=1e-12)
check("Proposition 42.5.8: with g = 0 (do nothing), boundary rows of L u are generally nonzero for affine u",
      np.abs((L @ ua)[dm.boundary_vertices(F)]).max() > 1e-2)
# Galerkin weak form with Neumann data assembled by quadrature equals L u - (M^G f - b_g) residual structure
gfun = lambda p, n: np.sin(p[0]) * n[0] + p[1] * n[1]  # noqa: E731
bgq = ch42.boundary_load(V, F, gfun, order=6)
# direct quadrature of oint g phi_i over every boundary edge (orientation independent)
bdirect = np.zeros(len(V))
tq, wq = dd.gauss_legendre_01(6)
loop = dm.boundary_loop(F)
for p0, p1 in zip(loop, np.roll(loop, -1)):
    e = V[p1] - V[p0]
    nrm = np.array([e[1], -e[0], 0.0]) / np.linalg.norm(e)
    for tk, wk in zip(tq, wq):
        gv = gfun(V[p0] + tk * e, nrm) * np.linalg.norm(e) * wk
        bdirect[p0] += gv * (1 - tk)
        bdirect[p1] += gv * tk
close("(42.5.6): ch42.boundary_load agrees with a direct quadrature along the boundary loop", bgq, bdirect, tol=1e-13)

# ---------------------------------------------------------------- Proposition 42.5.10
MG = ch42.galerkin_mass(V, F)
fv = np.cos(V[:, 0]) + V[:, 1] ** 2
bb = MG @ fv - bgq
area_int = MG.sum(0) @ fv
bd_int = 0.0
for p0, p1 in zip(loop, np.roll(loop, -1)):
    e = V[p1] - V[p0]
    nrm = np.array([e[1], -e[0], 0.0]) / np.linalg.norm(e)
    bd_int += sum(gfun(V[p0] + tk * e, nrm) * wk for tk, wk in zip(tq, wq)) * np.linalg.norm(e)
close("Proposition 42.5.10: 1^T (M^G f - b_g) = int W0 f - oint g", bb.sum(), area_int - bd_int, tol=1e-12)
uu = np.linalg.lstsq(L, bb - bb.sum() / len(V), rcond=None)[0]
check("Proposition 42.5.10: after removing the mean, L u = b is solvable (residual 0)",
      np.abs(L @ uu - (bb - bb.sum() / len(V))).max() < 1e-10)
check("Proposition 42.5.10: L has a one-dimensional kernel on the disk mesh", np.sum(np.abs(np.linalg.eigvalsh(L)) < 1e-10) == 1)

# ---------------------------------------------------------------- Example 42.5.11
g2 = lambda P: P[:, 0] ** 2 - P[:, 1] ** 2  # noqa: E731


def dirichlet(V, F, g):
    L = dm.cotan_laplacian(V, F)
    B = np.array(dm.boundary_vertices(F))
    I = np.setdiff1d(np.arange(len(V)), B)
    u = np.zeros(len(V))
    u[B] = g(V[B])
    u[I] = np.linalg.solve(L[np.ix_(I, I)], -L[np.ix_(I, B)] @ u[B])
    return u


dmax, drms = [], []
for n in (8, 16, 32, 64):
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    check(f"Example 42.5.11 (n = {n}): grid_mesh reproduces x^2 - y^2 exactly", np.abs(dirichlet(V, F, g2) - g2(V)).max() < 1e-12)
    rng = np.random.default_rng(7)
    I = np.setdiff1d(np.arange(len(V)), dm.boundary_vertices(F))
    W = V.copy()
    W[I, :2] += 0.2 / n * rng.uniform(-1, 1, (len(I), 2))
    e = dirichlet(W, F, g2) - g2(W)
    m = dm.mass_barycentric(W, F)
    dmax.append(np.abs(e).max())
    drms.append(np.sqrt((e ** 2 * m).sum() / m.sum()))
    if n == 32:
        check("Example 42.5.11: 508 negative cotan weights on the jittered 32 x 32 grid", (dd.hodge1(W, F) < 0).sum() == 508)
close("Example 42.5.11: jittered max errors 8.8e-4, 3.1e-4, 1.1e-4, 3.3e-5 (relative)", np.array(dmax) / [8.8e-4, 3.1e-4, 1.05e-4, 3.25e-5],
      np.ones(4), tol=0.02)
close("Example 42.5.11: RMS errors 2.8e-4 ... 5.7e-6", [drms[0], drms[-1]], [2.75e-4, 5.66e-6], tol=1e-6)
hs = 1.0 / np.array([8, 16, 32, 64])
close("Figure 42.5.3(a): fitted slopes 1.59 (max) and 1.86 (RMS)",
      [np.polyfit(np.log(hs), np.log(dmax), 1)[0], np.polyfit(np.log(hs), np.log(drms), 1)[0]], [1.59, 1.86], tol=6e-3)

# ---------------------------------------------------------------- Example 42.5.12
nmax = []
for n in (8, 16, 32):
    V, F = ch42.disk_mesh(n)
    L, m = dm.cotan_laplacian(V, F), dm.mass_barycentric(V, F)
    bx = -ch42.boundary_load(V, F, lambda p, nn: nn[0])
    ux = np.linalg.solve(-L + np.outer(m, m), -bx)
    check(f"Example 42.5.12(a) (n = {n}): Neumann data n_x reproduces x exactly; 1^T b = 0",
          np.abs(ux - (V[:, 0] - m @ V[:, 0] / m.sum())).max() < 1e-12 and abs(bx.sum()) < 1e-12)
    br = 2 * m - ch42.boundary_load(V, F, lambda p, nn: p @ nn)
    ur = np.linalg.solve(-L + np.outer(m, m), -br)
    ex = 0.5 * (V[:, 0] ** 2 + V[:, 1] ** 2)
    nmax.append(np.abs(ur - (ex - m @ ex / m.sum())).max())
    check(f"Example 42.5.12(b) (n = {n}): 1^T b = 0 (int 2 = oint <x, n> = 2 area)", abs(br.sum()) < 1e-12)
close("Example 42.5.12(b): max errors 6.9e-3, 1.7e-3, 4.3e-4", nmax, [6.94e-3, 1.74e-3, 4.34e-4], tol=6e-5)

# ---------------------------------------------------------------- Implementation 42.5
ns, same, printed = ch42.run_printed(PAGE, "code-42-5-1", "out-42-5-1")
check("Implementation 42.5: code-42-5-1 prints exactly out-42-5-1", same, printed)
check("Implementation 42.5: code-42-5-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-42-5-1") <= 30)

# ---------------------------------------------------------------- exercises
uu1 = x ** 2 / 2 - x
check("Exercise 42.5.1: u = x^2/2 - x solves u'' = 1, u(0) = 0, u'(1) = 0",
      sp.diff(uu1, x, 2) == 1 and uu1.subs(x, 0) == 0 and sp.diff(uu1, x).subs(x, 1) == 0)
check("Exercise 42.5.2: oint cos^2 = pi; (x^2 - y^2)/2 is harmonic with d_r = cos 2t",
      sp.integrate(sp.cos(t) ** 2, (t, 0, 2 * sp.pi)) == sp.pi and lap(r ** 2 * sp.cos(2 * t) / 2) == 0
      and sp.simplify(sp.diff(r ** 2 * sp.cos(2 * t) / 2, r).subs(r, 1) - sp.cos(2 * t)) == 0)
xs = np.linspace(0, 1, 9)
V, F = dm.grid_mesh(xs, xs)
L = dm.cotan_laplacian(V, F)
B = np.array(dm.boundary_vertices(F))
I = np.setdiff1d(np.arange(len(V)), B)
check("Exercise 42.5.3: (L(x^2 + y^2))_i = 4 l^2 at interior vertices (not harmonic)",
      np.allclose((L @ (V[:, 0] ** 2 + V[:, 1] ** 2))[I], 4 / 64))
ud = dirichlet(V, F, lambda P: np.sin(3 * P[:, 0]) + P[:, 1] ** 3)
E0 = -0.5 * ud @ L @ ud
ok = True
rng = np.random.default_rng(5)
for _ in range(20):
    v = np.zeros(len(V))
    v[I] = rng.normal(size=len(I))
    ok &= -0.5 * (ud + v) @ L @ (ud + v) > E0 + 1e-12
check("Exercise 42.5.4: the Dirichlet solution minimizes -1/2 w^T L w among w with w_B = g", ok)
xs = np.linspace(0, 1, 9)
V, F = dm.grid_mesh(xs, xs)
m = dm.mass_barycentric(V, F)
close("Exercise 42.5.5: 1^T M f = area = 1 for f = 1 on the unit square", m.sum(), 1.0, tol=1e-14)
ns6, same6, printed6 = ch42.run_printed(PAGE, "code-ex-42-5-6", "out-ex-42-5-6")
check("Exercise 42.5.6: code-ex-42-5-6 prints exactly out-ex-42-5-6", same6, printed6)
check("Exercise 42.5.6: code block is at most 30 lines", dgcode.code_lines(PAGE, "code-ex-42-5-6") <= 30)

summary()

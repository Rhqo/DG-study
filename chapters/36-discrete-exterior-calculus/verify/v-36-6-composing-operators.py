"""Verification for Section 36.6 (Composing Operators).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-6-composing-operators.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch36  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "36-6-composing-operators.html"
rng = np.random.default_rng(3606)
J = np.array([[0.0, -1.0], [1.0, 0.0]])

# a jittered acute planar patch
V, F = ch36.equilateral_patch(3)
V = V + 0.06 * rng.uniform(-1, 1, size=V.shape)
V3 = np.c_[V, np.zeros(len(V))]
assert dm.corner_angles(V3, F).max() < np.pi / 2
E, idx = dd.edge_list(F)
D0, D1 = dd.d0(F, len(V)), dd.d1(F)
s0, s1, s2 = dd.dual_areas(V, F, "circumcentric"), dd.hodge1(V, F), dd.hodge2(V, F)
dt1 = ch36.dual_d1(F, len(V))
interior = np.ones(len(V), bool)
interior[np.unique(dm.boundary_edges(F))] = False

# ---------------------------------------------------------------- Proposition 36.6.1
u = lambda p: np.sin(p[0]) * np.cos(0.7 * p[1]) + p[0] * p[1]  # noqa: E731
du = lambda p: np.array([np.cos(p[0]) * np.cos(0.7 * p[1]) + p[1],  # noqa: E731
                         -0.7 * np.sin(p[0]) * np.sin(0.7 * p[1]) + p[0]])
close("Proposition 36.6.1(a): d0 R0 u = R1(du) (gradient, exact)", D0 @ dd.derham0(u, V), dd.derham1(du, V, E, order=12),
      tol=1e-12)
X = lambda p: np.array([p[0] * p[1] ** 2 - np.sin(p[1]), np.exp(0.3 * p[0]) + p[1]])  # noqa: E731
curlX = lambda p: 0.3 * np.exp(0.3 * p[0]) - (2 * p[0] * p[1] - np.cos(p[1]))  # noqa: E731
close("Proposition 36.6.1(b): star2 d1 R1(X^flat) = face averages of curl X (exact)",
      s2 * (D1 @ dd.derham1(X, V, E, order=12)), s2 * dd.derham2(curlX, V, F, order=8), tol=1e-11)
Xc = rng.normal(size=2)
div_c = (dt1 @ (s1 * dd.derham1(lambda p: Xc, V, E))) / s0
check("Proposition 36.6.1(c): constant X has discrete divergence 0 at every interior vertex",
      np.abs(div_c[interior]).max() < 1e-12)
# (c) the dual value star1 alpha is the flux to the right of e*, and the sum around i* is the outward flux
polys = ch36.dual_edge_polylines(V, F)
Xl = lambda p: np.array([0.4 * p[0] + 0.1, -0.2 * p[1] + 0.3 * p[0]])  # noqa: E731   affine field, div = 0.2
flux = np.array([sum(Xl(0.5 * (p + q)) @ (-J @ (q - p)) for p, q in zip(P[:-1], P[1:])) for P in polys])
outward = dt1 @ flux
close("Proposition 36.6.1(c): d~1 of the exact dual fluxes = div X x |i*| for an affine X (divergence theorem)",
      outward[interior], 0.2 * s0[interior], tol=1e-12)

# ---------------------------------------------------------------- Definition 36.6.2, Proposition 36.6.3
ds1 = (D0.T @ (s1[:, None] * np.eye(len(E)))) / s0[:, None]          # star0^{-1} d0^T star1
ds2 = (D1.T @ (s2[:, None] * np.eye(len(F)))) / s1[:, None]          # star1^{-1} d1^T star2
ok = True
for _ in range(10):
    a0, a1, a2 = rng.normal(size=len(V)), rng.normal(size=len(E)), rng.normal(size=len(F))
    ok &= abs((D0 @ a0) @ (s1 * a1) - a0 @ (s0 * (ds1 @ a1))) < 1e-10
    ok &= abs((D1 @ a1) @ (s2 * a2) - a1 @ (s1 * (ds2 @ a2))) < 1e-10
check("Proposition 36.6.3(a): <d a, b>_k = <a, d* b>_{k-1} for k = 1, 2 (random)", ok)
check("Proposition 36.6.3(c): d*_1 d*_2 = 0", np.abs(ds1 @ ds2).max() < 1e-12)
close("Proposition 36.6.3: d*_1 = -star0^{-1} d~1 star1", ds1, -(dt1 @ np.diag(s1)) / s0[:, None], tol=1e-12)

# ---------------------------------------------------------------- Example 36.6.4, Non-example 36.6.5, Example 36.6.8(a)
for ell in (1.0, 0.5, 0.25):
    Ve, Fe = ch36.equilateral_patch(2, h=ell)
    Ee, _ = dd.edge_list(Fe)
    Le = dm.cotan_laplacian(np.c_[Ve, np.zeros(len(Ve))], Fe)
    se0 = dd.dual_areas(Ve, Fe, "circumcentric")
    D0e = dd.d0(Fe, len(Ve))
    g = 0.5 * (Ve[:, 0] ** 2 + Ve[:, 1] ** 2)
    codiff = (D0e.T @ (dd.hodge1(Ve, Fe) * dd.derham1(lambda p: p, Ve, Ee))) / se0
    close(f"Example 36.6.4 (l = {ell}): d*_1 R1(x dx + y dy) = -2 at the center vertex", codiff[0], -2.0, tol=1e-12)
    close(f"Example 36.6.8(a) (l = {ell}): star0^{{-1}} L x^2 = 2 at the center; row = 1/sqrt3 (x6), -2 sqrt3",
          [(Le @ Ve[:, 0] ** 2)[0] / se0[0], Le[0, 0], np.sort(Le[0][Le[0] > 0])[0], (Le[0] > 0).sum()],
          [2.0, -2 * np.sqrt(3), 1 / np.sqrt(3), 6], tol=1e-12)
    close(f"Non-example 36.6.5 (l = {ell}): graph Laplacian d0^T d0 x^2 = -3 l^2 at the center",
          (D0e.T @ D0e @ Ve[:, 0] ** 2)[0], -3 * ell ** 2, tol=1e-12)
    close(f"Example 36.6.8(a) (l = {ell}): star0 = (sqrt3/2) l^2 at the center", se0[0], np.sqrt(3) / 2 * ell ** 2, 1e-12)

# ---------------------------------------------------------------- Definition 36.6.6, Proposition 36.6.7
for name, (Vm, Fm) in (("jittered patch", (V3, F)), ("non-Delaunay pair", (np.c_[ch36.non_delaunay_pair()[0], np.zeros(4)],
                                                                       ch36.non_delaunay_pair()[1])),
                       ("icosphere(2)", dm.icosphere(2)), ("torus", dm.torus_mesh(2, 0.8, 8, 12))):
    Dm = dd.d0(Fm, len(Vm))
    Lm = -Dm.T @ (dd.hodge1(Vm, Fm)[:, None] * Dm)
    check(f"Proposition 36.6.7: -d0^T star1 d0 = dgmesh.cotan_laplacian = -dgdec.laplacian_psd on {name}",
          np.allclose(Lm, dm.cotan_laplacian(Vm, Fm)) and np.allclose(Lm, -dd.laplacian_psd(Vm, Fm)))
    check(f"Proposition 36.6.9(a): L symmetric, L 1 = 0 on {name}",
          np.allclose(Lm, Lm.T) and np.abs(Lm @ np.ones(len(Vm))).max() < 1e-12)
Lp = dm.cotan_laplacian(V3, F)
ok = True
for _ in range(10):
    uu = rng.normal(size=len(V))
    ok &= abs(-uu @ Lp @ uu - np.sum(s1 * (D0 @ uu) ** 2)) < 1e-10
check("Proposition 36.6.9(b): -u^T L u = sum w_ij (u_j - u_i)^2", ok)
ev = np.linalg.eigvalsh(-Lp)
check("Proposition 36.6.9(c): Delaunay patch: -L PSD with a one-dimensional kernel (constants)",
      s1.min() > 0 and ev.min() > -1e-12 and (np.abs(ev) < 1e-10).sum() == 1)
Vgm, Fgm = ch36.grid(4)
wgm = dd.hodge1(Vgm, Fgm)
evg = np.linalg.eigvalsh(-dm.cotan_laplacian(np.c_[Vgm, np.zeros(len(Vgm))], Fgm))
check("Proposition 36.6.9 (remark): grid_mesh has zero diagonal weights, yet -L is PSD with kernel = constants",
      np.abs(wgm).min() < 1e-12 and wgm.min() > -1e-12 and evg.min() > -1e-12 and (np.abs(evg) < 1e-10).sum() == 1)
# (36.6.1): on a planar mesh, R(a)^T star_k R(b) equals the L2 inner product for constant forms (k = 0, 1, 2)
area = dm.face_areas(V3, F).sum()
a1, b1 = rng.normal(size=2), rng.normal(size=2)
c0, c2 = rng.normal(size=2)
check("(36.6.1): constant 0-, 1-, 2-forms on the planar patch: a^T star_k b = L2 inner product",
      abs(c0 * c2 * s0.sum() - c0 * c2 * area) < 1e-12
      and abs((D0 @ V @ a1) @ (s1 * (D0 @ V @ b1)) - (a1 @ b1) * area) < 1e-12
      and abs((c0 / s2) @ (s2 * (c2 / s2)) - c0 * c2 * area) < 1e-10)     # R2(c dA)_f = c |f| = c / star2_f
Vn, Fn = ch36.non_delaunay_pair()
Ln = dm.cotan_laplacian(np.c_[Vn, np.zeros(4)], Fn)
close("Example 36.6.10: non-Delaunay pair, w01 = -1.696, eigenvalues of -L = 0, 0.275, 3.607, 7.393",
      np.round(np.concatenate([[Ln[0, 1]], np.linalg.eigvalsh(-Ln)]), 3), [-1.696, 0, 0.275, 3.607, 7.393], tol=1e-12)

# ---------------------------------------------------------------- Example 36.6.8(b),(c), Proposition 36.6.11
Vg, Fg = ch36.grid(4)
Lg = dm.cotan_laplacian(np.c_[Vg, np.zeros(len(Vg))], Fg)
inner_g = (Vg[:, 0] > 0) & (Vg[:, 0] < 1) & (Vg[:, 1] > 0) & (Vg[:, 1] < 1)
check("Example 36.6.8(b): square grid interior rows are the 5-point stencil (1, 1, 1, 1, -4)",
      all(sorted(np.round(Lg[i][np.abs(Lg[i]) > 1e-12], 12).tolist()) == [-4.0, 1.0, 1.0, 1.0, 1.0]
          for i in np.nonzero(inner_g)[0]))
check("Example 36.6.8(b): (L u)_i = 0 at interior vertices for u = x^2 - y^2",
      np.abs((Lg @ (Vg[:, 0] ** 2 - Vg[:, 1] ** 2))[inner_g]).max() < 1e-13)
Vi, Fi = dm.icosphere(2)
check("Example 36.6.8(c): L 1 = 0 on icosphere(2)", np.abs(dm.cotan_laplacian(Vi, Fi).sum(1)).max() < 1e-12)
Vr = ch36.grid(5)[0] + 0.03 * rng.uniform(-1, 1, size=(36, 2))
Fr = ch36.grid(5)[1]
assert (ch36.cross2(Vr[Fr[:, 1]] - Vr[Fr[:, 0]], Vr[Fr[:, 2]] - Vr[Fr[:, 0]]) > 0).all()   # no flipped triangle
Lr = dm.cotan_laplacian(np.c_[Vr, np.zeros(36)], Fr)
inner_r = np.ones(36, bool)
inner_r[np.unique(dm.boundary_edges(Fr))] = False
aff = 1.3 * Vr[:, 0] - 0.4 * Vr[:, 1] + 2.0
check("Proposition 36.6.11(b): linear precision (L u)_i = 0 for affine u on a jittered grid",
      np.abs((Lr @ aff)[inner_r]).max() < 1e-12)
Vt = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.45], [0.5, 0.55]])
Ft = np.array([[0, 1, 4], [1, 3, 4], [4, 3, 5], [3, 2, 5], [2, 0, 5], [0, 4, 5]])
Ft = np.array([t if ch36.cross2(Vt[t[1]] - Vt[t[0]], Vt[t[2]] - Vt[t[0]]) > 0 else t[[0, 2, 1]] for t in Ft])
w_t = dd.hodge1(Vt, Ft)
Lt = dm.cotan_laplacian(np.c_[Vt, np.zeros(6)], Ft)
aff_t = 0.7 * Vt[:, 0] + 1.1 * Vt[:, 1]
check("Proposition 36.6.11(b): linear precision also with a negative weight (non-Delaunay interior edge)",
      w_t.min() < 0 and np.abs((Lt @ aff_t)[[4, 5]]).max() < 1e-12)
close("Proposition 36.6.11(a): star0^{-1} L = -d*_1 d0", Lp / s0[:, None], -ds1 @ D0, tol=1e-10)

# ---------------------------------------------------------------- Implementation 36.6 / Figure 36.6.2
ns, same, printed = ch36.run_printed(PAGE, "code-36-6-1", "out-36-6-1")
check("Implementation 36.6: code-36-6-1 prints exactly out-36-6-1", same, printed)
check("Implementation 36.6: code-36-6-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-6-1") <= 30)
e4 = ns["solve_poisson"](4)[0]
close("Figure 36.6.2: n = 4 error 5.30e-02", round(float(e4), 4), 0.0530, tol=1e-12)
_, Lq, st0, Vq = ns["solve_poisson"](8)
inq = (Vq[:, 0] > 0) & (Vq[:, 0] < 1) & (Vq[:, 1] > 0) & (Vq[:, 1] < 1)
close("Implementation 36.6: interior star0 = h^2 and star0^{-1} L is the 5-point difference / h^2",
      [st0[inq].min(), st0[inq].max(), (Lq[inq] / st0[inq][:, None])[0][inq].min()], [1 / 64, 1 / 64, -4 * 64], 1e-9)

# ---------------------------------------------------------------- Exercises
Ve1, Fe1 = ch36.equilateral_triangle(1.3)
L1 = dm.cotan_laplacian(np.c_[Ve1, np.zeros(3)], Fe1)
close("Exercise 36.6.1: single equilateral triangle, L = (1/(2 sqrt3)) [[-2,1,1],...], eig(-L) = 0, sqrt3/2, sqrt3/2",
      np.concatenate([L1.ravel(), np.linalg.eigvalsh(-L1)]),
      np.concatenate([(np.ones((3, 3)) - 3 * np.eye(3)).ravel() / (2 * np.sqrt(3)), [0, np.sqrt(3) / 2, np.sqrt(3) / 2]]),
      tol=1e-12)
ok = True
for _ in range(5):
    uu, vv = rng.normal(size=len(V)), rng.normal(size=len(V))
    ok &= abs(uu @ Lp @ vv - vv @ Lp @ uu) < 1e-10 and abs(uu @ Lp @ vv + np.sum(s1 * (D0 @ uu) * (D0 @ vv))) < 1e-10
check("Exercise 36.6.3: u^T L v = v^T L u = -sum w (u_j - u_i)(v_j - v_i)", ok)
w = rng.normal(size=len(F))
beta = (ch36.dual_d0(F) @ w) / s1
check("Exercise 36.6.4: star2 d1 d0 = 0 and div(star1^{-1} d~0 w) = 0",
      np.abs(s2[:, None] * (D1 @ D0)).max() < 1e-14 and np.abs(dt1 @ (s1 * beta)).max() < 1e-12)
# (b) with w = psi(c_f) for an affine psi, beta = R1 of the 1-form of -J grad psi on interior edges (not +J grad psi)
gpsi = np.array([0.7, -1.3])
beta_psi = (ch36.dual_d0(F) @ (dd.circumcenters(V, F) @ gpsi)) / s1
int_e = np.abs(D1).sum(0) == 2
check("Exercise 36.6.4(b): star1^{-1} d~0 psi(c_f) = R1((-J grad psi)^flat) on interior edges, != R1((J grad psi)^flat)",
      np.abs(beta_psi - D0 @ V @ (-J @ gpsi))[int_e].max() < 1e-12
      and np.abs(beta_psi - D0 @ V @ (J @ gpsi))[int_e].max() > 0.1)
ok = True
for i in np.nonzero(interior)[0]:
    nb = [j for j in range(len(V)) if j != i and abs(Lp[i, j]) > 1e-14]
    uu = rng.normal(size=len(V))
    uu[i] = -sum(Lp[i, j] * uu[j] for j in nb) / Lp[i, i]                # (L u)_i = 0
    ok &= min(uu[nb]) - 1e-12 <= uu[i] <= max(uu[nb]) + 1e-12
check("Exercise 36.6.5: positive weights: (Lu)_i = 0 implies min <= u_i <= max over the neighbors", ok)
ns6, same6, printed6 = ch36.run_printed(PAGE, "code-ex-36-6-6", "out-ex-36-6-6")
check("Exercise 36.6.6: code-ex-36-6-6 prints exactly out-ex-36-6-6", same6, printed6)
lam = ns6["lam"]
check("Exercise 36.6.6: multiplicities 1, 3, 5, 7 near 0, 2, 6, 12 (relative error < 7%)",
      abs(lam[0]) < 1e-10 and np.allclose(lam[1:4], 2, rtol=1e-3) and np.allclose(lam[4:9], 6, rtol=0.03)
      and np.allclose(lam[9:16], 12, rtol=0.07) and lam[16] > 15)

summary()

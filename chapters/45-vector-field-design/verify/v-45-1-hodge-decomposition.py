"""Verification for Section 45.1 (Hodge Decomposition).

Run from the project root:
``PYTHONPATH=tools python3 chapters/45-vector-field-design/verify/v-45-1-hodge-decomposition.py``
"""

import pathlib
import sys
import warnings

import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch45  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

warnings.simplefilter("ignore")
PAGE = HERE / "45-1-hodge-decomposition.html"
rng = np.random.default_rng(45)


def proj(W, B):
    """W-orthogonal projector onto the column space of B (W symmetric positive definite)."""
    G = B.T @ W @ B
    return B @ np.linalg.pinv(G) @ B.T @ W


# ---------------------------------------------------------------- Proposition 45.1.1 and 45.1.2 (linear algebra)
for trial in range(3):
    m, k = 7, 3
    A = rng.normal(size=(m, k))
    Wv = np.diag(rng.uniform(0.5, 2.0, m))
    Wu = np.diag(rng.uniform(0.5, 2.0, k))
    Astar = np.linalg.inv(Wu) @ A.T @ Wv
    x = rng.normal(size=k)
    y = rng.normal(size=m)
    check(f"Proposition 45.1.1 (trial {trial}): <A x, y>_V = <x, A* y>_U with A* = Wu^-1 A^T Wv",
          abs((A @ x) @ Wv @ y - x @ Wu @ (Astar @ y)) < 1e-12)
    K = np.linalg.svd(Astar)[2][np.linalg.matrix_rank(Astar):].T          # basis of ker A*
    check(f"Proposition 45.1.1 (trial {trial}): ker A* is W-orthogonal to im A and dim im A + dim ker A* = m",
          np.abs(A.T @ Wv @ K).max() < 1e-12 and np.linalg.matrix_rank(np.hstack([A, K])) == m)
# a chain complex U --A--> V --B--> Y with B A = 0
U0 = rng.normal(size=(9, 3))
A = U0[:, :2]
Bt = np.linalg.svd(A.T)[2][2:].T[:, :3]                                 # columns orthogonal to im A (Euclidean)
B = (Bt @ rng.normal(size=(3, 3))).T
check("Proposition 45.1.2 setup: B A = 0", np.abs(B @ A).max() < 1e-12)
Wv = np.diag(rng.uniform(0.5, 2.0, 9))
Wy = np.diag(rng.uniform(0.5, 2.0, 3))
Bstar = np.linalg.inv(Wv) @ B.T @ Wy
Pa, Pb = proj(Wv, A), proj(Wv, Bstar)
w = rng.normal(size=9)
z = w - Pa @ w - Pb @ w
check("Proposition 45.1.2: im A and im B* are W-orthogonal", np.abs(A.T @ Wv @ Bstar).max() < 1e-12)
check("Proposition 45.1.2: the remainder z lies in ker B and in ker A*",
      np.abs(B @ z).max() < 1e-12 and np.abs(A.T @ Wv @ z).max() < 1e-12)
check("Proposition 45.1.2: dim Z = 9 - 2 - 3 = 4", 9 - np.linalg.matrix_rank(np.vstack([B, A.T @ Wv])) == 4)

# ---------------------------------------------------------------- Example 45.1.3 and Non-example 45.1.4
F3 = np.array([[0, 1, 2]])
D0 = dd.d0(F3, 3)                    # edges [0,1], [0,2], [1,2]
E3, idx3 = dd.edge_list(F3)
check("Example 45.1.3: edge order [0,1], [0,2], [1,2]", E3.tolist() == [[0, 1], [0, 2], [1, 2]])
om = np.zeros(3)
om[idx3[(0, 1)]], om[idx3[(1, 2)]], om[idx3[(0, 2)]] = 1.0, 2.0, 4.0
cyc = np.zeros(3)
cyc[idx3[(0, 1)]], cyc[idx3[(1, 2)]], cyc[idx3[(0, 2)]] = 1.0, 1.0, -1.0
check("Example 45.1.3: the loop c = [0,1] + [1,2] - [0,2] spans ker d0^T", np.abs(D0.T @ cyc).max() < 1e-15)
gam = (om @ cyc) / (cyc @ cyc) * cyc
check("Example 45.1.3: gamma = -(1/3) c", np.allclose(gam, -cyc / 3))
exact = om - gam
alpha = np.array([0.0, 4 / 3, 11 / 3])
check("Example 45.1.3: omega - gamma = d0 alpha with alpha = (0, 4/3, 11/3)", np.allclose(D0 @ alpha, exact))
check("Example 45.1.3: with the face [0,1,2], d1 = c^T and the harmonic part becomes the co-exact part",
      np.allclose(dd.d1(F3)[0], cyc))
Wd = np.diag([1.0, 1.0, 4.0])                     # weights 1, 1, 4 on [0,1], [1,2], [0,2]
Wfull = np.zeros((3, 3))
Wfull[idx3[(0, 1)], idx3[(0, 1)]], Wfull[idx3[(1, 2)], idx3[(1, 2)]], Wfull[idx3[(0, 2)], idx3[(0, 2)]] = 1.0, 1.0, 4.0
hW = np.linalg.solve(Wfull, cyc)                  # ker d0^T W = W^{-1} ker d0^T
gW = (om @ Wfull @ hW) / (hW @ Wfull @ hW) * hW
expected = np.zeros(3)
expected[idx3[(0, 1)]], expected[idx3[(1, 2)]], expected[idx3[(0, 2)]] = -4 / 9, -4 / 9, 1 / 9
check("Non-example 45.1.4: with weights (1, 1, 4) the harmonic part is (-4/9, -4/9, 1/9)", np.allclose(gW, expected))
check("Non-example 45.1.4: both harmonic parts have period -1 around c", np.isclose(cyc @ gW, -1) and np.isclose(cyc @ gam, -1))
check("Non-example 45.1.4: omega - gamma_W is exact", np.linalg.matrix_rank(np.column_stack([D0, om - gW])) == 2)

# ---------------------------------------------------------------- Lemma 45.1.5, Example 45.1.7: forms on T_{2,0.8}
# coordinates (u, v); positively oriented orthonormal coframe (rho dv, r du) for the outward normal
R, r = 2.0, 0.8
n = 256
uu, vv = np.meshgrid(2 * np.pi * np.arange(n) / n, 2 * np.pi * np.arange(n) / n, indexing="ij")
rho = R + r * np.cos(uu)
dA = r * rho * (2 * np.pi / n) ** 2                # area weights of the periodic trapezoid rule
U_, V_ = sp.symbols("u v", real=True)
rho_s = R + r * sp.cos(U_)
f_s = sp.sin(U_) * sp.cos(2 * V_) + sp.cos(U_ + V_)            # a function
a_s = sp.cos(V_) + sp.sin(U_) ** 2                              # 1-form a du + b dv
b_s = sp.sin(U_ - V_) * sp.cos(U_)


def star1(a, b):
    """star(a du + b dv) = -(rho/r) a dv + (r/rho) b du  (star du = -(rho/r) dv, star dv = (r/rho) du)."""
    return (r / rho_s) * b, -(rho_s / r) * a


def codiff1(a, b):
    """d* = -star d star on 1-forms: star(a du + b dv) = A du + B dv, d of it = (B_u - A_v) du^dv,
    du^dv = -dA/(r rho) (dA = r rho dv^du), star of it = -(B_u - A_v)/(r rho)."""
    A_, B_ = star1(a, b)
    return (sp.diff(B_, U_) - sp.diff(A_, V_)) / (r * rho_s)


def L2(a1, b1, a2, b2):
    """<<a1 du + b1 dv, a2 du + b2 dv>> = int (a1 a2 / r^2 + b1 b2 / rho^2) dA."""
    g = sp.lambdify((U_, V_), a1 * a2 / r ** 2 + b1 * b2 / rho_s ** 2, modules="numpy")
    return float(np.sum(g(uu, vv) * dA))


df = (sp.diff(f_s, U_), sp.diff(f_s, V_))
lhs = L2(*df, a_s, b_s)
cd = sp.lambdify((U_, V_), codiff1(a_s, b_s), modules="numpy")
rhs = float(np.sum(sp.lambdify((U_, V_), f_s, modules="numpy")(uu, vv) * cd(uu, vv) * dA))
close("Lemma 45.1.5: <<df, beta>> = <<f, d* beta>> on T_{2,0.8} (spectral quadrature)", lhs, rhs, 1e-10)
check("Example 45.1.7: d* dv = 0 on T_{R,r} (dv is harmonic)", sp.simplify(codiff1(0, 1)) == 0)
check("Example 45.1.7: d* du != 0 (du is closed but not harmonic)", sp.simplify(codiff1(1, 0)) != 0)
cR = sp.sqrt(R ** 2 - r ** 2)
check("Example 45.1.7: d*(c du/(R + r cos u)) = 0", sp.simplify(codiff1(cR / rho_s, 0)) == 0)
close("Example 45.1.7: int_0^{2 pi} sqrt(R^2 - r^2)/(R + r cos u) du = 2 pi",
      float(sp.Integral(cR / rho_s, (U_, 0, 2 * sp.pi)).evalf(30)), 2 * np.pi, 1e-12)

# ---------------------------------------------------------------- Proposition 45.1.10 and the Conventions box (plane)
x, y = sp.symbols("x y", real=True)
v = sp.Function("v")(x, y)
u = sp.Function("u")(x, y)
grad = lambda f: sp.Matrix([sp.diff(f, x), sp.diff(f, y)])  # noqa: E731
J = lambda X: sp.Matrix([-X[1], X[0]])  # noqa: E731       (+90 degrees)
star_1form = lambda P, Q: (-Q, P)  # noqa: E731            star(P dx + Q dy) = -Q dx + P dy (star dx = dy, star dy = -dx)
div = lambda X: sp.diff(X[0], x) + sp.diff(X[1], y)  # noqa: E731
curl = lambda X: sp.diff(X[1], x) - sp.diff(X[0], y)  # noqa: E731
Sd = star_1form(*grad(v))
check("Proposition 45.1.10 (plane): star dv = (J grad v)^flat", sp.simplify(sp.Matrix(Sd) - J(grad(v))) == sp.zeros(2, 1))
check("Proposition 45.1.10: div (J grad v) = 0", sp.simplify(div(J(grad(v)))) == 0)
check("Proposition 45.1.10: curl (grad u) = 0", sp.simplify(curl(grad(u))) == 0)
check("Proposition 45.1.10: curl (J grad v) = Delta v", sp.simplify(curl(J(grad(v))) - div(grad(v))) == 0)
phi = sp.Function("phi")(x, y)
# d*(phi dA) = -star d star(phi dA) = -star d phi; our star, and Crane's section 7.3 star (star_C a = a o J = -star a)
dstar_ours = [-c for c in star_1form(*grad(phi))]
check("Conventions: d*(phi dA) = -(J grad phi)^flat (clockwise, Example 35.5.9(a))",
      sp.simplify(sp.Matrix(dstar_ours) + J(grad(phi))) == sp.zeros(2, 1))
delta_C_sec4 = star_1form(*grad(phi))                    # delta = star d star with the section 4.1.3 star (= ours)
delta_C_sec7 = [-c for c in star_1form(*grad(phi))]      # with star_C = -star on 1-forms
check("Conventions: with the Cr section 4.1.3 star, delta(phi dA) = +(J grad phi)^flat = -d*(phi dA) (counterclockwise)",
      sp.simplify(sp.Matrix(delta_C_sec4) - J(grad(phi))) == sp.zeros(2, 1))
check("Conventions: with the Cr section 7.3 star, delta(phi dA) = d*(phi dA) (clockwise)",
      sp.simplify(sp.Matrix(delta_C_sec7) - sp.Matrix(dstar_ours)) == sp.zeros(2, 1))
# on 1-forms in 2D: delta = star d star equals -d* with our star and +d* with star_C = -star
P_, Q_ = sp.Function("P")(x, y), sp.Function("Q")(x, y)
A1, B1 = star_1form(P_, Q_)
ours_dstar = -(sp.diff(B1, x) - sp.diff(A1, y))         # -star d star (P dx + Q dy) = -(div) ... check below
check("Conventions: d*(P dx + Q dy) = -div(P, Q) (Theorem 35.5.8(a))", sp.simplify(ours_dstar + sp.diff(P_, x) + sp.diff(Q_, y)) == 0)

# ---------------------------------------------------------------- Warning: zero and negative star1
for nu, nv in ((12, 24), (24, 48)):
    V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
    s1 = dd.hodge1(V, F)
    check(f"Warning: torus_mesh(2, 0.8, {nu}, {nv}) has star1 = 0 (|.| < 1e-12) exactly on its {nu * nv} diagonals",
          int((np.abs(s1) < 1e-12).sum()) == nu * nv and s1[np.abs(s1) >= 1e-12].min() > 0.05)
Vt = np.array([[0, 0, 0], [2, 0, 0], [1, 0.3, 0.1], [1, -0.25, -0.1]], float)
Ft = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
check("Warning: the folded tetrahedron is closed and consistently oriented",
      dm.oriented_consistently(Ft) and dm.euler_characteristic(Ft) == 2 and dm.signed_volume(Vt, Ft) > 0)
s1t = dd.hodge1(Vt, Ft)
Et, idt = dd.edge_list(Ft)
close("Warning: star1 on [0,1] of the folded tetrahedron = -1.573", s1t[idt[(0, 1)]], -1.5727, 6e-5)
evt = np.linalg.eigvalsh(dd.d1(Ft) @ (dd.d1(Ft).T / s1t[:, None]))
close("Warning: d1 star1^-1 d1^T has the eigenvalue -0.199", evt[0], -0.1989, 6e-5)
check("Warning: -L of the folded tetrahedron is still positive semidefinite (Corollary 42.3.6)",
      np.linalg.eigvalsh(dd.laplacian_psd(Vt, Ft)).min() > -1e-12)
for h in (0.2,):
    Vg, Fg = dm.genus2_mesh(h)
    s1g = dd.hodge1(Vg, Fg)
    check(f"Warning: genus2_mesh({h}) has negative star1 entries ({int((s1g < 0).sum())} of {len(s1g)})",
          int((s1g < 0).sum()) == 880 and len(s1g) == 7578)

# ---------------------------------------------------------------- Theorem 45.1.12, Proposition 45.1.13, Example 45.1.14
res_ico = {}
for lev in (2, 3, 4):
    V, F = dm.icosphere(lev)
    D0, D1, s1 = dd.d0(F, len(V)), dd.d1(F), dd.hodge1(V, F)
    omega = ch45.sphere_example(V, F)
    al, bt, ga = ch45.hodge_decomposition(D0, D1, s1, omega)
    c = ch45.unit(V[F].mean(1))
    e_a = np.abs(al - (V[:, 2] - V[:, 2].mean())).max()
    e_b = np.abs(bt + (c[:, 0] - c[:, 0].mean())).max()
    res_ico[lev] = (len(V), e_a, e_b, np.abs(ga).max())
    if lev == 3:
        ex, co = D0 @ al, D1.T @ bt / s1
        ip = lambda a, b: a @ (s1 * b)  # noqa: E731
        check("Theorem 45.1.12 (icosphere(3)): the three parts are <.,.>_1-orthogonal",
              max(abs(ip(ex, co)), abs(ip(ex, ga)), abs(ip(co, ga))) < 1e-12)
        check("Theorem 45.1.12 (icosphere(3)): the co-exact part is d*_2 beta with beta = star2^{-1} bt",
              np.allclose(co, (D1.T @ (dd.hodge2(V, F) * (bt / dd.hodge2(V, F)))) / s1))
        check("Proposition 45.1.13: alpha and bt solve the two systems",
              np.abs(D0.T @ (s1 * (D0 @ al)) - D0.T @ (s1 * omega)).max() < 1e-12
              and np.abs(D1 @ (D1.T @ bt / s1) - D1 @ omega).max() < 1e-12)
        n0, n1, n2 = dm.counts(F)
        check("Theorem 45.1.12 (icosphere(3)): dim im d0 + dim im d1^T = n1 (so H = 0)",
              np.linalg.matrix_rank(D0) + np.linalg.matrix_rank(D1) == n1)
        # the same decomposition by explicit W-orthogonal projectors
        Wd = np.diag(s1)
        pe = proj(Wd, D0) @ omega
        pc = proj(Wd, (D1.T / s1[:, None])) @ omega
        check("Proposition 45.1.13: the systems give the W-orthogonal projections", np.allclose(pe, ex) and np.allclose(pc, co))
for lev, (nv_, e_a, e_b, g_) in res_ico.items():
    print(f"   icosphere({lev}): n0 = {nv_}, |alpha - z| = {e_a:.2e}, |bt + x| = {e_b:.2e}, |gamma| = {g_:.1e}")
check("Example 45.1.14: |gamma| < 1e-12 on icosphere(2..4)", all(g < 1e-12 for *_, g in res_ico.values()))
check("Example 45.1.14: |alpha - z| = 5.3e-4, 1.4e-4, 3.5e-5",
      [round(res_ico[k][1], 5) for k in (2, 3)] == [0.00053, 0.00014] and abs(res_ico[4][1] - 3.5e-5) < 5e-7)
check("Example 45.1.14: |bt + x| = 2.6e-2, 1.0e-2, 5.1e-3",
      [round(res_ico[k][2], 3) for k in (2, 3)] == [0.026, 0.010] and abs(res_ico[4][2] - 5.1e-3) < 5e-5)

# ---------------------------------------------------------------- Example 45.1.15: the flat torus
res_ft = {}
for nn in (12, 24, 48):
    ft = ch45.FlatTorus(nn, nn)
    F = ft.F
    D0, D1 = dd.d0(F), dd.d1(F)
    s0, s1, s2 = ch45.stars_from_corners(F, ft.C)
    if nn == 12:
        check("Example 45.1.15: FlatTorus(12, 12) has (n0, n1, n2) = (144, 432, 288), chi = 0, all star1 = 1/sqrt 3",
              dm.counts(F) == (144, 432, 288) and np.allclose(s1, 1 / np.sqrt(3)) and dm.is_manifold(F)
              and dm.oriented_consistently(F))
        dx = ft.edge_vec[:, 0]
        check("Example 45.1.15 / Exercise 45.1.3: R_1(dx) is discrete harmonic on the equilateral flat torus",
              np.abs(D1 @ dx).max() < 1e-15 and np.abs(D0.T @ (s1 * dx)).max() < 1e-14)
    omega, uv, vf, h = ch45.flat_torus_example(ft)
    al, bt, ga = ch45.hodge_decomposition(D0, D1, s1, omega)
    res_ft[nn] = (np.abs(al - (uv - uv.mean())).max(), np.abs(bt + (vf - vf.mean())).max(), np.abs(ga - h).max())
for nn, (a_, b_, g_) in res_ft.items():
    print(f"   FlatTorus({nn}, {nn}): |alpha - u| = {a_:.2e}, |bt + v| = {b_:.2e}, |gamma - h| = {g_:.1e}")
check("Example 45.1.15: gamma = R_1(h) to 1e-14 on all three meshes", all(g < 1e-14 for *_, g in res_ft.values()))
check("Example 45.1.15: |alpha - u| = 1.06e-5, 6.29e-7, 3.92e-8 (factor about 16)",
      abs(res_ft[12][0] - 1.06e-5) < 5e-8 and abs(res_ft[24][0] - 6.29e-7) < 5e-10 and abs(res_ft[48][0] - 3.92e-8) < 5e-11)
check("Example 45.1.15: |bt + v| = 8.99e-5, 1.16e-5, 1.45e-6 (factor about 8)",
      abs(res_ft[12][1] - 8.99e-5) < 5e-8 and abs(res_ft[24][1] - 1.16e-5) < 5e-8 and abs(res_ft[48][1] - 1.45e-6) < 5e-9)

# ---------------------------------------------------------------- Theorem 45.1.16: harmonic forms and cohomology
cases = [("icosphere(2)", *dm.icosphere(2)), ("torus_mesh(2, 0.8, 12, 24)", *dm.torus_mesh(2, 0.8, 12, 24)),
         ("genus2_mesh(0.2)", *dm.genus2_mesh(0.2))]
for name, V, F in cases:
    D0, D1, s1 = dd.d0(F, len(V)), dd.d1(F), dd.hodge1(V, F)
    chi = dm.euler_characteristic(F)
    if not name.startswith("genus2"):
        dimH = (D0.shape[0] - np.linalg.matrix_rank(D1)) - np.linalg.matrix_rank(D0)
        check(f"Theorem 45.1.16(a) ({name}): dim ker d1 / im d0 = 2 - chi = {2 - chi}", dimH == 2 - chi)
        A = D0.T @ (s1[:, None] * D0)
        check(f"Theorem 45.1.16(a) ({name}): -L = d0^T star1 d0 is PSD with a 1-dimensional kernel",
              np.linalg.eigvalsh(A)[0] > -1e-10 and np.linalg.eigvalsh(A)[1] > 1e-8)
    else:
        tc, C, Cs = ch45.generators(F)
        G = ch45.harmonic_basis(D0, s1, Cs)
        check(f"Theorem 45.1.16(a) ({name}): with negative star1 entries the 4 harmonic parts are closed, "
              "co-closed and independent",
              np.abs(D1 @ G.T).max() < 1e-9 and np.abs(D0.T @ (s1[:, None] * G.T)).max() < 1e-8
              and np.linalg.matrix_rank(G) == 4)
ft = ch45.FlatTorus(12, 12)
D0, D1 = dd.d0(ft.F), dd.d1(ft.F)
s0, s1, s2 = ch45.stars_from_corners(ft.F, ft.C)
gam = ft.edge_vec @ np.array([0.3, 0.1])
better = all(((gam + D0 @ a) @ (s1 * (gam + D0 @ a))) > gam @ (s1 * gam) for a in rng.normal(size=(20, D0.shape[1])))
check("Theorem 45.1.16(b): on the flat torus the harmonic form is shorter than every other form in its class (20 trials)",
      better)

# ---------------------------------------------------------------- Example 45.1.17: torus of revolution
R, r = 2.0, 0.8
cR = np.sqrt(R * R - r * r)
res_t = {}
for nu, nv in ((12, 24), (24, 48), (48, 96)):
    V, F = dm.torus_mesh(R, r, nu, nv)
    E, _ = dd.edge_list(F)
    D0, s1 = dd.d0(F, len(V)), dd.hodge1(V, F)
    uang = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]) - R)
    vang = np.arctan2(V[:, 1], V[:, 0])
    du, dv = ch45.wrap(D0 @ uang), ch45.wrap(D0 @ vang)
    gdu, gdv = ch45.harmonic_basis(D0, s1, np.array([du, dv]))
    exs = np.zeros(len(E))
    for t, w in zip(ch45.GL_T, ch45.GL_W):
        exs += w * cR / (R + r * np.cos(uang[E[:, 0]] + t * du)) * du
    res_t[(nu, nv)] = (np.abs(gdu - exs).max() / np.abs(exs).max(), np.abs(gdv - dv).max())
for k_, (rel, dvh) in res_t.items():
    print(f"   torus_mesh(2, 0.8, {k_[0]}, {k_[1]}): relative error of harm(du) = {rel:.2e}, |harm(dv) - dv| = {dvh:.1e}")
check("Example 45.1.17: R_1(dv) is exactly discrete harmonic (to 1e-14)", all(d < 1e-14 for _, d in res_t.values()))
check("Example 45.1.17: relative error of the harmonic part of R_1(du) = 7.3e-3, 1.8e-3, 4.6e-4",
      [round(res_t[k_][0], 4) for k_ in ((12, 24), (24, 48), (48, 96))] == [0.0073, 0.0018, 0.0005]
      and abs(res_t[(48, 96)][0] - 4.6e-4) < 5e-6)

# ---------------------------------------------------------------- Implementation box
ns, same, out = ch45.run_printed(PAGE, "code-45-1-1", "out-45-1-1")
check("code-45-1-1 runs and prints exactly out-45-1-1", same)
import dgcode  # noqa: E402
check("code-45-1-1 has at most 30 lines", dgcode.code_lines(PAGE, "code-45-1-1") <= 30)

# ---------------------------------------------------------------- Exercises
# Exercise 45.1.2: omega = (2, -1, 3) on [0,1], [1,2], [0,2]
om2 = np.zeros(3)
om2[idx3[(0, 1)]], om2[idx3[(1, 2)]], om2[idx3[(0, 2)]] = 2.0, -1.0, 3.0
g2 = (om2 @ cyc) / 3 * cyc
check("Exercise 45.1.2: gamma = -(2/3) c, exact part (8/3, -1/3, 7/3) with alpha = (0, 8/3, 7/3)",
      np.allclose(g2, -2 / 3 * cyc) and np.allclose(om2 - g2, dd.d0(F3, 3) @ np.array([0, 8 / 3, 7 / 3])))
# Exercise 45.1.4: Hodge Laplacian kernel on icosphere(2) (zero) and on the flat torus (dimension 2)
for name, F, s0, s1, s2 in (("flat torus", ft.F, *ch45.stars_from_corners(ft.F, ft.C)),):
    D0, D1 = dd.d0(F), dd.d1(F)
    DH = D0 @ ((D0.T * s1[None, :]) / s0[:, None]) + (D1.T * s2[None, :]) @ D1 / s1[:, None]
    sym = np.diag(np.sqrt(s1)) @ DH @ np.diag(1 / np.sqrt(s1))
    ev = np.linalg.eigvalsh((sym + sym.T) / 2)
    check(f"Exercise 45.1.4 ({name}): the discrete Hodge Laplacian has a 2-dimensional kernel",
          (np.abs(ev) < 1e-9).sum() == 2 and np.allclose(sym, sym.T))
# Exercise 45.1.5 (coding) and 45.1.6 (coding): printed code
for blk in ("code-ex-45-1-5", "code-ex-45-1-6"):
    ns, same, out = ch45.run_printed(PAGE, blk, blk.replace("code", "out"))
    check(f"{blk} runs and prints exactly {blk.replace('code', 'out')}", same)

summary()

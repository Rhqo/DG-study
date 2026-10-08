"""Verification for Section 31.3 (The Cotangent Laplacian).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/verify/v-31-3-cotangent-laplacian.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

from dgcheck import check, close, summary, sym_equal  # noqa: E402

rng = np.random.default_rng(31)

# ---------------------------------------------------------------- Lemma 31.3.1: one triangle (symbolic)
a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
xi, xj, xk = sp.Matrix([0, 0]), sp.Matrix([a1, a2]), sp.Matrix([b1, b2])
A2 = a1 * b2 - a2 * b1                       # twice the signed area


def hat_grad(p, q, r):
    """Gradient of the linear function equal to 1 at p and 0 at q, r."""
    M = sp.Matrix([[q[0] - p[0], q[1] - p[1]], [r[0] - p[0], r[1] - p[1]]])
    return M.inv() * sp.Matrix([-1, -1])


gi, gj, gk = hat_grad(xi, xj, xk), hat_grad(xj, xk, xi), hat_grad(xk, xi, xj)


def cot_at(p, q, r):
    u, v = q - p, r - p
    return (u.dot(v)) / (u[0] * v[1] - u[1] * v[0])


area = A2 / 2
cot_k = cot_at(xk, xi, xj)
cot_j = cot_at(xj, xk, xi)
# with the orientation (i, j, k) counterclockwise the cot formula uses |area|; take A2 > 0
sym_equal("Lemma 31.3.1: area * <grad phi_i, grad phi_j> = -cot(theta_k)/2", area * gi.dot(gj), -cot_k / 2, {a1: (0.5, 2), a2: (-0.3, 0.3), b1: (0.2, 1.2), b2: (0.5, 2)})
sym_equal("Lemma 31.3.1: area * |grad phi_i|^2 = (cot theta_j + cot theta_k)/2", area * gi.dot(gi), (cot_j + cot_k) / 2,
          {a1: (0.5, 2), a2: (-0.3, 0.3), b1: (0.2, 1.2), b2: (0.5, 2)})
sym_equal("Lemma 31.3.1: the three hat gradients sum to zero", gi + gj + gk, sp.Matrix([0, 0]),
          {a1: (0.5, 2), a2: (-0.3, 0.3), b1: (0.2, 1.2), b2: (0.5, 2)})
# the numeric triangle of Figure 31.3.1(b)
P = np.array([[0, 0], [2.4, 0], [0.7, 1.5]])
gnum = lambda p, q, r: np.linalg.solve(np.array([q - p, r - p]), [-1.0, -1.0])  # noqa: E731
Gi, Gj = gnum(P[0], P[1], P[2]), gnum(P[1], P[2], P[0])
ar = 0.5 * abs((P[1] - P[0])[0] * (P[2] - P[0])[1] - (P[1] - P[0])[1] * (P[2] - P[0])[0])
tk = np.arccos(np.dot(P[0] - P[2], P[1] - P[2]) / np.linalg.norm(P[0] - P[2]) / np.linalg.norm(P[1] - P[2]))
close("Fig 31.3.1(b): area <grad phi_i, grad phi_j> = -cot(theta_k)/2 (numeric triangle)", ar * Gi @ Gj, -0.5 / np.tan(tk),
      tol=1e-12)
e = P[2] - P[1]
foot = P[1] + np.dot(P[0] - P[1], e) / np.dot(e, e) * e
close("Fig 31.3.1(b): |grad phi_i| = 1/h_i", np.linalg.norm(Gi), 1 / np.linalg.norm(P[0] - foot), tol=1e-12)

# ---------------------------------------------------------------- Proposition 31.3.2: Dirichlet energy = -u^T L u
for name, (V, F) in (("icosphere 2", dm.icosphere(2)), ("torus 8x12", dm.torus_mesh(2, 0.8, 8, 12)),
                     ("random planar", dm.grid_mesh(np.sort(rng.uniform(0, 1, 7)), np.sort(rng.uniform(0, 1, 6))))):
    L = dm.cotan_laplacian(V, F)
    u = rng.normal(size=len(V))
    close(f"Prop 31.3.2: int |grad u|^2 = -u^T L u on {name}", dm.dirichlet_energy(V, F, u) / (-u @ L @ u), 1.0,
          tol=1e-10)
    check(f"Prop 31.3.3: L symmetric and L 1 = 0 on {name}", np.allclose(L, L.T) and np.allclose(L.sum(1), 0))
    ev = np.linalg.eigvalsh(-L)
    check(f"Prop 31.3.3: -L positive semidefinite on {name} (min eig {ev.min():.1e})", ev.min() > -1e-10)
V, F = dm.icosphere(2)
ev = np.linalg.eigvalsh(-dm.cotan_laplacian(V, F))
check("Prop 31.3.3: kernel = constants on a connected mesh (one zero eigenvalue)", (np.abs(ev) < 1e-9).sum() == 1)
V2 = np.vstack([V, V + 3.0])
F2 = np.vstack([F, F + len(V)])
ev2 = np.linalg.eigvalsh(-dm.cotan_laplacian(V2, F2))
check("Prop 31.3.3: two components -> two-dimensional kernel", (np.abs(ev2) < 1e-9).sum() == 2)
# PSD even with negative weights: the max-principle mesh of Figure 31.3.2(c)
B = np.array([[1.6, 0], [0.8, 0.25], [0, 1], [-1, 0.5], [-1, -0.5], [0, -1], [0.8, -0.25]])
Vm = np.c_[np.vstack([[0, 0], B]), np.zeros(8)]
Fm = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
Lm = dm.cotan_laplacian(Vm, Fm)
check("Prop 31.3.3: -L stays PSD although an edge weight is negative (Figure 31.3.2(c))",
      Lm[0, 1] < 0 and np.linalg.eigvalsh(-Lm).min() > -1e-12)

# linear precision on a planar mesh (also a non-Delaunay one)
for name, (Vp, Fp) in (("graded grid", dm.grid_mesh((np.linspace(0, 1, 9) ** 1.7) * 2, np.linspace(0, 1.4, 7))),
                       ("max-principle fan", (Vm, Fm))):
    Lp = dm.cotan_laplacian(Vp, Fp)
    inner = [i for i in range(len(Vp)) if i not in dm.boundary_vertices(Fp)]
    check(f"Prop 31.3.3: linear precision (L x)_i = 0 at interior vertices of a planar {name}",
          np.abs((Lp @ Vp)[inner]).max() < 1e-12)

# ---------------------------------------------------------------- non-Delaunay weights
for _ in range(200):
    al, be = rng.uniform(0.1, 3.0, 2)
    w = 0.5 * (1 / np.tan(al) + 1 / np.tan(be))
    if abs(al + be - np.pi) < 1e-6:
        continue
    ok = abs(w - np.sin(al + be) / (2 * np.sin(al) * np.sin(be))) < 1e-9 and ((w < 0) == (al + be > np.pi))
    if not ok:
        break
check("Remark 31.3.4: w = sin(alpha + beta)/(2 sin alpha sin beta) < 0 iff alpha + beta > pi (200 random pairs)", ok)


def circ_inside(a, b, c, d):
    """d strictly inside the circumcircle of a, b, c (2D)."""
    M = np.array([[a[0] - d[0], a[1] - d[1], (a - d) @ (a - d)], [b[0] - d[0], b[1] - d[1], (b - d) @ (b - d)],
                  [c[0] - d[0], c[1] - d[1], (c - d) @ (c - d)]])
    o = np.linalg.det(M)
    orient = (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]
    return o * np.sign(orient) > 0


ok = True
for _ in range(300):
    xi_, xj_ = np.array([0.0, 0.0]), np.array([2.0, 0.0])
    xl = np.array([rng.uniform(0.2, 1.8), rng.uniform(0.1, 1.5)])
    xm = np.array([rng.uniform(0.2, 1.8), -rng.uniform(0.1, 1.5)])
    ang = lambda p, a, b: np.arccos((a - p) @ (b - p) / np.linalg.norm(a - p) / np.linalg.norm(b - p))  # noqa: E731
    w = 0.5 * (1 / np.tan(ang(xl, xi_, xj_)) + 1 / np.tan(ang(xm, xi_, xj_)))
    ok &= (w < 0) == circ_inside(xi_, xj_, xl, xm)
check("Remark 31.3.4: w_ij < 0 iff x_m is inside the circumcircle of x_i x_j x_l (300 random quads)", ok)
ang = lambda p, a, b: np.degrees(np.arccos((a - p) @ (b - p) / np.linalg.norm(a - p) / np.linalg.norm(b - p)))  # noqa: E731
for xl_, xm_, aplusb, wv in ((np.array([1.0, 1.0]), np.array([1.0, -1.3]), 165, 0.13),
                             (np.array([1.0, 0.45]), np.array([1.0, -0.35]), 273, -1.07)):
    s_ = ang(xl_, np.array([0., 0]), np.array([2., 0])) + ang(xm_, np.array([0., 0]), np.array([2., 0]))
    w_ = 0.5 * (1 / np.tan(np.radians(ang(xl_, np.array([0., 0]), np.array([2., 0]))))
                + 1 / np.tan(np.radians(ang(xm_, np.array([0., 0]), np.array([2., 0])))))
    check(f"Fig 31.3.2(a)(b): alpha + beta = {aplusb} deg, w = {wv:+.2f}", round(s_) == aplusb and round(w_, 2) == wv)

# Figure 31.3.2(c): discrete maximum principle fails
u = np.ones(8)
u[1] = 0.0
u0 = -(Lm[0, 1:] @ u[1:]) / Lm[0, 0]
check(f"Fig 31.3.2(c): harmonic value u_0 = {u0:.3f} > 1 = max of the boundary values", round(u0, 2) == 1.29)
close("Fig 31.3.2(c): weight of the edge 0-1 = -1.44", round(Lm[0, 1], 2), -1.44, tol=1e-12)
close("Fig 31.3.2(c): its opposite angles are 145.3 deg", round(ang(Vm[2, :2], Vm[0, :2], Vm[1, :2]), 1), 145.3, tol=1e-9)
# flip the edge 0-1 to 2-7 (the Delaunay choice): vertex 0 is no longer adjacent to the 0-valued vertex
Ff = np.array([f for f in Fm.tolist() if not (0 in f and 1 in f)] + [[0, 7, 2], [7, 1, 2]])
Lf = dm.cotan_laplacian(Vm, Ff)
u0f = -(Lf[0, [2, 3, 4, 5, 6, 7]] @ u[[2, 3, 4, 5, 6, 7]]) / Lf[0, 0]
check("Fig 31.3.2(c): after flipping to the Delaunay edge, all weights at 0 are positive and u_0 = 1",
      (Lf[0, [2, 3, 4, 5, 6, 7]] > 0).all() and abs(u0f - 1) < 1e-12)
# uniform weights would give 6/7
check("Fig 31.3.2(c): with uniform weights u_0 = 6/7 (inside the range)", abs(np.mean(u[1:]) - 6 / 7) < 1e-12)

# ---------------------------------------------------------------- Proposition 31.3.5: area gradient = -L x
for name, (V, F) in (("jittered icosphere", dm.icosphere(2)), ("torus", dm.torus_mesh(2, 0.8, 7, 11))):
    V = V + 0.03 * rng.normal(size=V.shape)
    L = dm.cotan_laplacian(V, F)
    g = np.zeros_like(V)
    for i in (0, 5, 17):
        for k in range(3):
            Vp, Vn = V.copy(), V.copy()
            Vp[i, k] += 1e-6
            Vn[i, k] -= 1e-6
            g[i, k] = (dm.face_areas(Vp, F).sum() - dm.face_areas(Vn, F).sum()) / 2e-6
    close(f"Prop 31.3.5: grad of the total area w.r.t. x_i = -(L x)_i on a {name}", g[[0, 5, 17]], -(L @ V)[[0, 5, 17]],
          tol=1e-6)

# ---------------------------------------------------------------- Example 31.3.6: spheres
for lv in (2, 3, 4):
    V, F = dm.icosphere(lv)
    L = dm.cotan_laplacian(V, F)
    nv = (((L @ V) / dm.mass_voronoi(V, F)[:, None]) * V).sum(1)
    check(f"Example 31.3.6: level {lv}: <(M^-1 L x)_i, N_i> = -2 exactly with the Voronoi mass", np.allclose(nv, -2, atol=1e-10))
for lv, val in ((2, -2.27), (3, -2.29)):
    Vl, Fl = dm.icosphere(lv)
    nbl = (((dm.cotan_laplacian(Vl, Fl) @ Vl) / dm.mass_barycentric(Vl, Fl)[:, None]) * Vl).sum(1)
    check(f"Example 31.3.6: level {lv}, barycentric mass: worst normal component {nbl.min():.3f} ~ {val}",
          round(nbl.min(), 2) == val)
V, F = dm.icosphere(4)
L = dm.cotan_laplacian(V, F)
Hv = (L @ V) / dm.mass_voronoi(V, F)[:, None]
nb = (((L @ V) / dm.mass_barycentric(V, F)[:, None]) * V).sum(1)
deg = np.array([len(x) for x in dm.neighbors(F)])
check("Example 31.3.6: level 4, barycentric mass: -2.29 at the 12 valence-5 vertices, others in [-2.03, -1.99]",
      np.allclose(nb[deg == 5], nb[deg == 5][0]) and round(nb[deg == 5][0], 2) == -2.29
      and nb[deg == 6].min() > -2.03 and round(nb[deg == 6].max(), 2) == -1.99)
tang = np.linalg.norm(Hv - (Hv * V).sum(1)[:, None] * V, axis=1)
check(f"Example 31.3.6: level 4: tangential part of M^-1 L x below 0.01 (max {tang.max():.4f})", tang.max() < 0.01)
# the inscribed-sphere identity with the pure circumcentric Voronoi area, on a jittered sphere
V, F = dm.icosphere(3)
h = np.mean([np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)])
J = V + 0.1 * h * rng.normal(size=V.shape)
J /= np.linalg.norm(J, axis=1, keepdims=True)
L = dm.cotan_laplacian(J, F)
pureV = np.zeros(len(J))
for (i, j), w in dm.cotan_weights(J, F).items():
    pureV[i] += w * np.sum((J[i] - J[j]) ** 2) / 4
    pureV[j] += w * np.sum((J[i] - J[j]) ** 2) / 4
close("Example 31.3.6: inscribed mesh: <(L x)_i, x_i> = -(1/2) sum_j w_ij |x_j - x_i|^2 = -2 A_i(circumcentric)",
      ((L @ J) * J).sum(1), -2 * pureV, tol=1e-12)
# Figure 31.3.3(b): the jittered level-4 sphere of the figure (same random stream). The spread of the normal component
# with the mixed area comes only from vertices next to obtuse triangles (review R1, 2026-10-08).
rgf = np.random.default_rng(1)
for lv in range(1, 5):
    V0f, F0f = dm.icosphere(lv)
    hf = np.mean([np.linalg.norm(V0f[a] - V0f[b]) for a, b in dm.edges(F0f)])
    Jf = V0f + 0.1 * hf * rgf.normal(size=V0f.shape)
    Jf /= np.linalg.norm(Jf, axis=1, keepdims=True)
LJf = dm.cotan_laplacian(Jf, F0f) @ Jf
mixf = dm.mass_voronoi(Jf, F0f)
njf = (LJf * Jf).sum(1) / mixf
angf = dm.corner_angles(Jf, F0f)
obtf = np.zeros(len(Jf), bool)
for f in np.where(angf.max(1) > np.pi / 2)[0]:
    obtf[F0f[f]] = True
pureVf = np.zeros(len(Jf))
for (i, j), w in dm.cotan_weights(Jf, F0f).items():
    pureVf[i] += w * np.sum((Jf[i] - Jf[j]) ** 2) / 4
    pureVf[j] += w * np.sum((Jf[i] - Jf[j]) ** 2) / 4
check(f"Fig 31.3.3(b): jittered level 4, mixed mass: std {njf.std():.3f}, range [{njf.min():.2f}, {njf.max():.2f}]",
      round(njf.std(), 3) == 0.020 and round(njf.min(), 2) == -2.52 and round(njf.max(), 2) == -1.79)
check(f"Example 31.3.6 / Fig 31.3.3(b): the {obtf.sum()} vertices off -2 are exactly those next to the "
      f"{(angf.max(1) > np.pi / 2).sum()} obtuse triangles",
      obtf.sum() == 439 and (angf.max(1) > np.pi / 2).sum() == 178 and np.allclose(njf[~obtf], -2, atol=1e-9)
      and (np.abs(njf[obtf] + 2) > 1e-6).all())
check("Example 31.3.6: with the circumcentric Voronoi area the jittered sphere gives exactly -2 at every vertex",
      np.allclose((LJf * Jf).sum(1) / pureVf, -2, atol=1e-9))
tangf = np.linalg.norm(LJf / mixf[:, None] - njf[:, None] * Jf, axis=1)
check(f"Example 31.3.6: jittered: the tangential part is what fluctuates (max {tangf.max():.3f})",
      round(tangf.max(), 2) == 0.05)

# ---------------------------------------------------------------- Figure 31.3.4: smoothing drift
nx, ny = 17, 13
Vg, Fg = dm.grid_mesh((np.linspace(0, 1, nx) ** 1.7) * 2.0, np.linspace(0, 1.4, ny), pattern="alternate")
bnd = np.array([i in (0, nx - 1) or j in (0, ny - 1) for i in range(nx) for j in range(ny)])
I = ~bnd
rg = np.random.default_rng(3)
V0 = Vg.copy()
V0[I, 2] += 0.04 * rg.normal(size=I.sum())
Lu = dm.uniform_laplacian(Fg)


def step(X, kind, tau):
    if kind == "uniform":
        A, rhs = np.eye(len(X)) - tau * Lu, X
    else:
        m = dm.mass_barycentric(X, Fg)
        A, rhs = np.diag(m) - tau * dm.cotan_laplacian(X, Fg), m[:, None] * X
    Xn = X.copy()
    Xn[I] = np.linalg.solve(A[np.ix_(I, I)], rhs[I] - A[np.ix_(I, ~I)] @ X[~I])
    return Xn


res = {}
for kind, tau in (("uniform", 0.5), ("cotan", 0.005)):
    X = V0.copy()
    for s_ in range(40):
        X = step(X, kind, tau)
    res[kind] = (np.sqrt((X[I, 2] ** 2).mean()), np.linalg.norm((X - V0)[I, :2], axis=1).mean())
check(f"Fig 31.3.4: after 40 steps uniform drift {res['uniform'][1]:.4f} (noise {res['uniform'][0]:.4f}), "
      f"cotan drift {res['cotan'][1]:.4f} (noise {res['cotan'][0]:.4f})",
      round(res["uniform"][1], 3) == 0.048 and round(res["cotan"][1], 4) == 0.0063 and res["cotan"][0] < res["uniform"][0])
Xc = Vg.copy()
for s_ in range(5):
    Xc = step(Xc, "cotan", 0.005)
Xu = Vg.copy()
for s_ in range(5):
    Xu = step(Xu, "uniform", 0.5)
check("Fig 31.3.4: noise-free plane: cotan moves nothing, uniform slides vertices",
      np.abs(Xc - Vg).max() < 1e-12 and np.abs(Xu - Vg).max() > 0.01)

# ---------------------------------------------------------------- PyTorch3D definitions (pytorch3d/ops/laplacian_matrices.py)
def p3d_cot_laplacian(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A_, B_, C_ = (np.linalg.norm(v1 - v2, axis=1), np.linalg.norm(v0 - v2, axis=1), np.linalg.norm(v0 - v1, axis=1))
    s = 0.5 * (A_ + B_ + C_)
    area_ = np.sqrt(np.clip(s * (s - A_) * (s - B_) * (s - C_), 1e-12, None))
    cota = (B_ ** 2 + C_ ** 2 - A_ ** 2) / area_ / 4
    cotb = (A_ ** 2 + C_ ** 2 - B_ ** 2) / area_ / 4
    cotc = (A_ ** 2 + B_ ** 2 - C_ ** 2) / area_ / 4
    n = len(V)
    Lp = np.zeros((n, n))
    for (ii, jj, cc) in ((F[:, 1], F[:, 2], cota), (F[:, 2], F[:, 0], cotb), (F[:, 0], F[:, 1], cotc)):
        np.add.at(Lp, (ii, jj), cc)
    Lp = Lp + Lp.T
    inv_areas = np.zeros(n)
    for c_ in range(3):
        np.add.at(inv_areas, F[:, c_], area_)
    return Lp, 1 / inv_areas


V, F = dm.icosphere(4)
Lp, inv_a = p3d_cot_laplacian(V, F)
L = dm.cotan_laplacian(V, F)
off = ~np.eye(len(V), dtype=bool)
check("Text: PyTorch3D cot_laplacian off-diagonal = cot a + cot b = 2 L_ij, no diagonal",
      np.allclose(Lp[off], 2 * L[off]) and np.allclose(np.diag(Lp), 0))
check("Text: PyTorch3D inv_areas = 1 / (area of all faces around i) = 1 / (3 A_i barycentric)",
      np.allclose(inv_a, 1 / (3 * dm.mass_barycentric(V, F))))
cotcurv = np.linalg.norm((Lp @ V - Lp.sum(1)[:, None] * V) * 0.25 * inv_a[:, None], axis=1)
Hb = np.linalg.norm((L @ V) / dm.mass_barycentric(V, F)[:, None], axis=1) / 2
close("Text: PyTorch3D 'cotcurv' per-vertex term = |H_i| / 3 (barycentric H_i)", cotcurv, Hb / 3, tol=1e-12)
check(f"Text: on the unit icosphere (level 4) the 'cotcurv' loss is {cotcurv.mean():.4f} ~ 1/3 (text: 0.333)",
      abs(cotcurv.mean() - 1 / 3) < 0.003 and round(cotcurv.mean(), 3) == 0.333)
cotw = (Lp @ V) / Lp.sum(1)[:, None] - V
check("Text: PyTorch3D 'cot' term = cot-weighted centroid of the neighbours minus x_i (points inward on a sphere)",
      ((cotw * V).sum(1) < 0).all())

# ---------------------------------------------------------------- losses: homogeneity and normal consistency
V0s, Fs = dm.icosphere(3)
rg2 = np.random.default_rng(0)
V0s = V0s * (1 + 0.03 * rg2.normal(size=(len(V0s), 1)))
Es = dm.edges(Fs)
Lus = dm.uniform_laplacian(Fs)
pairs = []
for (i, j), fs in dm.edge_faces(Fs).items():
    a = [x for x in Fs[fs[0]] if x not in (i, j)][0]
    b = [x for x in Fs[fs[1]] if x not in (i, j)][0]
    pairs.append((i, j, a, b))
pairs = np.array(pairs)


def nc_value(X, P_=pairs):
    v0, v1, a, b = (X[P_[:, k]] for k in range(4))
    n0, n1 = np.cross(v1 - v0, a - v0), np.cross(b - v0, v1 - v0)
    return (1 - (n0 * n1).sum(1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(n1, axis=1)).mean()


lap_value = lambda X: np.linalg.norm(Lus @ X, axis=1).mean()  # noqa: E731
edge_value = lambda X: ((X[Es[:, 0]] - X[Es[:, 1]]) ** 2).sum(1).mean()  # noqa: E731
for name, fn, k in (("Laplacian (uniform)", lap_value, 1), ("edge length", edge_value, 2), ("normal consistency", nc_value, 0)):
    s_ = 1.37
    c_ = np.array([0.2, -0.1, 0.4])
    close(f"Insight box: {name} loss is homogeneous of degree {k} and translation invariant",
          fn(c_ + s_ * (V0s - c_)), s_ ** k * fn(V0s), tol=1e-12)
# Euler: <grad L, X - c> = k L  -> d/dt sum |x - c|^2 = -2 k L
for name, fn, k in (("Laplacian (uniform)", lap_value, 1), ("edge length", edge_value, 2), ("normal consistency", nc_value, 0)):
    h_ = 1e-6
    d = (fn(V0s * (1 + h_)) - fn(V0s * (1 - h_))) / (2 * h_)
    close(f"Insight box: <grad L, X> = {k} L for {name} (Euler's theorem)", d, k * fn(V0s), tol=1e-6)
# normal consistency does not depend on the stored winding of the faces
Fflip = Fs.copy()
Fflip[::3] = Fflip[::3, ::-1]
pairs_f = []
for (i, j), fs in dm.edge_faces(Fflip).items():
    a = [x for x in Fflip[fs[0]] if x not in (i, j)][0]
    b = [x for x in Fflip[fs[1]] if x not in (i, j)][0]
    pairs_f.append((i, j, a, b))
close("Text: PyTorch3D normal consistency uses the edge and the two opposite vertices, not the face winding",
      nc_value(V0s, np.array(pairs_f)), nc_value(V0s), tol=1e-12)
# small bending: 1 - cos theta ~ theta^2 / 2
th = sp.symbols("theta")
sym_equal("Text: 1 - cos(theta) = theta^2/2 + O(theta^4)", sp.series(1 - sp.cos(th), th, 0, 4).removeO(), th ** 2 / 2)
# edge loss gradient is a uniform (graph) Laplacian
Lg = np.zeros((len(V0s), len(V0s)))
for i, j in Es:
    Lg[i, j] = Lg[j, i] = 1
np.fill_diagonal(Lg, -Lg.sum(1))
d = V0s[Es[:, 0]] - V0s[Es[:, 1]]
g = np.zeros_like(V0s)
np.add.at(g, Es[:, 0], 2 * d / len(Es))
np.add.at(g, Es[:, 1], -2 * d / len(Es))
close("Exercise 31.3.4: gradient of the mean squared edge length = -(2/n1) L_graph X", g, -(2 / len(Es)) * Lg @ V0s,
      tol=1e-14)

# ---------------------------------------------------------------- Figure 31.3.5 numbers
exec_ns = {}
src = pathlib.Path(__file__).resolve().parents[1] / "figures" / "fig-31-3-5-losses.py"
check("Fig 31.3.5: figure script exists", src.exists())


def run(fn_grad, steps):
    X = V0s.copy()
    r = np.linalg.norm(V0s - V0s.mean(0), axis=1)
    r0, s0 = r.std() / r.mean(), r.mean()
    tr = []
    for _ in range(steps):
        g_ = fn_grad(X)
        X = X - 0.0005 * g_ / np.abs(g_).max()
        rr = np.linalg.norm(X - X.mean(0), axis=1)
        tr.append((rr.std() / rr.mean() / r0, rr.mean() / s0))
        if tr[-1][0] < 0.25:
            break
    return np.array(tr)


def g_edge(X):
    d_ = X[Es[:, 0]] - X[Es[:, 1]]
    g_ = np.zeros_like(X)
    np.add.at(g_, Es[:, 0], 2 * d_ / len(Es))
    np.add.at(g_, Es[:, 1], -2 * d_ / len(Es))
    return g_


def g_lap(X):
    R = Lus @ X
    return Lus.T @ (R / np.linalg.norm(R, axis=1)[:, None]) / len(X)


te = run(g_edge, 6000)
tl = run(g_lap, 6000)
check(f"Fig 31.3.5: edge loss reaches 25% roughness at size {te[-1, 1]:.3f}", round(te[-1, 1], 3) == 0.965)
check(f"Fig 31.3.5: Laplacian loss stalls at roughness {tl[:, 0].min():.2f} and ends at size {tl[-1, 1]:.3f}",
      round(tl[:, 0].min(), 2) == 0.56 and round(tl[-1, 1], 3) == 0.936)
check(f"Fig 31.3.5: after the stall the Laplacian loss makes the mesh rougher again (end roughness {tl[-1, 0]:.2f})",
      round(tl[-1, 0], 2) == 0.62)

# Exercise 31.3.1: regular triangular lattice
hh = 0.7
pts = np.array([[0, 0]] + [[hh * np.cos(t), hh * np.sin(t)] for t in np.arange(6) * np.pi / 3])
Vt = np.c_[pts, np.zeros(7)]
Ft = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
Lt = dm.cotan_laplacian(Vt, Ft)
close("Exercise 31.3.1: equilateral lattice: w = 1/sqrt(3)", Lt[0, 1:], np.full(6, 1 / np.sqrt(3)), tol=1e-12)
uq = Vt[:, 0] ** 2
close("Exercise 31.3.1: (M^-1 L u)_0 = 2 = Delta(x^2) exactly (barycentric A_0 = (sqrt3/2) h^2)",
      (Lt @ uq)[0] / dm.mass_barycentric(Vt, Ft)[0], 2.0, tol=1e-12)
close("Exercise 31.3.2: alpha = beta = 100 deg: w = cot(100 deg) = -0.176", round(1 / np.tan(np.radians(100)), 3), -0.176,
      tol=1e-12)

summary()

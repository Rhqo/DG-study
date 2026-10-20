"""Verification for Section 40.3 (Curvature Normals).

Run from the project root: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/verify/v-40-3-curvature-normals.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgsym  # noqa: E402
import ddg40  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "40-3-curvature-normals.html"

# ---------------------------------------------------------------- Implementation 40.3 and Exercise 40.3.6
check("Implementation 40.3: at most 30 lines", dgcode.code_lines(PAGE, "code-40-3-1") <= 30)
ns, out = ddg40.run_printed(PAGE, "code-40-3-1")
check("Implementation 40.3: printed output matches the page", out.strip() == ddg40.printed_block(PAGE, "out-40-3-1").strip(), out)
ns2, out2 = ddg40.run_printed(PAGE, "code-ex-40-3-6", dict(ns))
check("Exercise 40.3.6: printed output matches the page", out2.strip() == ddg40.printed_block(PAGE, "out-ex-40-3-6").strip(), out2)
cn = ns["curvature_normals"]
for name, (V, F) in [("icosphere(2)", dm.icosphere(2)), ("torus_mesh(2, 0.8, 12, 20)", dm.torus_mesh(2, 0.8, 12, 20))]:
    a = cn(V, F)
    b = ddg40.curvature_normals(V, F)
    check(f"Implementation 40.3: curvature_normals = lib implementation on {name}", all(np.allclose(x, y, atol=1e-12) for x, y in zip(a, b)))
    close(f"Implementation 40.3: (L x)_i = dgmesh.cotan_laplacian @ x on {name}", a[1], dm.cotan_laplacian(V, F) @ V, 1e-12)

# ---------------------------------------------------------------- Proposition 40.3.1 and Example 40.3.2 (symbolic)
for key in ("sphere", "cylinder", "torus", "saddle"):
    ex = dgsym.EXAMPLES[key]
    X = sp.Matrix(ex["expr"])
    u, v = ex["coords"]
    pos = ex.get("positive", ())
    Xu, Xv = X.diff(u), X.diff(v)
    Nn = dgsym.unit_normal(X, u, v, pos)
    Nu, Nv = Nn.diff(u), Nn.diff(v)
    K, H = dgsym.K_H(X, u, v, pos)
    n = Xu.cross(Xv)
    area = sp.sqrt(n.dot(n))
    dom = {s: ex["domain"][s] for s in ex["domain"]} if "domain" in ex else None
    lhs_H = Xu.cross(Nv) - Xv.cross(Nu)                                  # (dx ^ dN)(d_u, d_v)
    lhs_K = Nu.cross(Nv)                                                 # (1/2)(dN ^ dN)(d_u, d_v)
    for c in range(3):
        sym_equal(f"Proposition 40.3.1: {key}: (dx ^ dN)(d_u, d_v) = -2 H N |x_u x x_v| (component {c})",
                  lhs_H[c], -2 * H * Nn[c] * area, symbols_domain=dom)
        sym_equal(f"Proposition 40.3.1: {key}: (1/2)(dN ^ dN)(d_u, d_v) = K N |x_u x x_v| (component {c})",
                  lhs_K[c], K * Nn[c] * area, symbols_domain=dom)
ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
check("Example 40.3.2(a): unit sphere H = -1, K = 1 with the outward normal",
      sp.simplify(dgsym.K_H(ex["expr"].subs(ex["params"][0], 1), th, ph, ex["positive"])[1] + 1) == 0)
ex = dgsym.EXAMPLES["cylinder"]
check("Example 40.3.2(b): cylinder H = -1/(2r), K = 0",
      sp.simplify(dgsym.K_H(ex["expr"], *ex["coords"])[1] + 1 / (2 * ex["params"][0])) == 0
      and sp.simplify(dgsym.K_H(ex["expr"], *ex["coords"])[0]) == 0)

# ---------------------------------------------------------------- (40.3.2): Delta x = 2 H N (coordinate formula on the torus) and Exercise 40.3.3
ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
X = sp.Matrix(ex["expr"])
E, Fc, G = dgsym.first_ff(X, u, v, ex["positive"])
g = sp.Matrix([[E, Fc], [Fc, G]])
sg = sp.sqrt(sp.simplify(g.det()))
gi = g.inv()
q = (u, v)
K, H = dgsym.K_H(X, u, v, ex["positive"])
Nn = dgsym.unit_normal(X, u, v, ex["positive"])
for c in range(3):
    lap = sum(sp.diff(sg * gi[a, b] * sp.diff(X[c], q[b]), q[a]) for a in range(2) for b in range(2)) / sg
    sym_equal(f"(40.3.2): torus: Delta x = 2 H N by the coordinate formula (component {c})", lap, 2 * H * Nn[c],
              symbols_domain={u: (0, 6.2), v: (0, 6.2), ex["params"][0]: (2, 3), ex["params"][1]: (0.3, 0.9)})
ph_, t_ = sp.symbols("varphi t", real=True)
Xm = sp.Matrix([sp.sech(t_) * sp.cos(ph_), sp.sech(t_) * sp.sin(ph_), sp.tanh(t_)])
Em, Fm, Gm = dgsym.first_ff(Xm, ph_, t_)
check("Exercise 40.3.3: Mercator coordinates are conformal with e^{2 mu} = sech^2 t",
      sp.simplify(Em - sp.sech(t_) ** 2) == 0 and sp.simplify(Gm - sp.sech(t_) ** 2) == 0 and sp.simplify(Fm) == 0)
for c in range(3):
    sym_equal(f"Exercise 40.3.3: Mercator: x_phiphi + x_tt = -2 sech^2 t x (component {c})",
              Xm[c].diff(ph_, 2) + Xm[c].diff(t_, 2), -2 * sp.sech(t_) ** 2 * Xm[c])
uu, vv = sp.symbols("u v", real=True)
Xc = sp.Matrix([sp.cosh(vv) * sp.cos(uu), sp.cosh(vv) * sp.sin(uu), vv])
Ec, Fcc, Gc = dgsym.first_ff(Xc, uu, vv)
check("Exercise 40.3.3: catenoid is conformal (E = G = cosh^2 v, F = 0) and x_uu + x_vv = 0",
      sp.simplify(Ec - Gc) == 0 and sp.simplify(Fcc) == 0 and sp.simplify(Xc.diff(uu, 2) + Xc.diff(vv, 2)) == sp.zeros(3, 1))

# ---------------------------------------------------------------- Theorem 40.3.4, Propositions 40.3.5 and 40.3.6
rng = np.random.default_rng(7)
Vj, Fj = dm.icosphere(2)
Vj = Vj + rng.normal(0, 0.04, Vj.shape)
obtuse = dm.corner_angles(Vj, Fj).max() > np.pi / 2
check("Theorem 40.3.4: the jittered icosphere(2) has obtuse triangles (circumcenters outside)", obtuse)
tests = [("icosphere(2)", dm.icosphere(2)), ("jittered icosphere(2)", (Vj, Fj)), ("torus_mesh(2, 0.8, 12, 20)", dm.torus_mesh(2, 0.8, 12, 20)),
         ("perturbed icosahedron", ddg40.perturbed_icosahedron()), ("unit cube (max)", ddg40.unit_cube("max"))]
for name, (V, F) in tests:
    L = dm.cotan_laplacian(V, F) @ V
    worst_c = max(np.abs(-ddg40.boundary_integral_N_cross_dx(V, F, i, "circumcentric") - L[i]).max() for i in range(len(V)))
    worst_b = max(np.abs(-ddg40.boundary_integral_N_cross_dx(V, F, i, "barycentric") - L[i]).max() for i in range(len(V)))
    check(f"Theorem 40.3.4: {name}: -oint N x dx = (L x)_i (circumcentric and barycentric cells)", worst_c < 1e-12 and worst_b < 1e-12,
          (worst_c, worst_b))
    close(f"Proposition 40.3.6: {name}: (L x)_i = -sum l sin(theta/2) Nbar", ddg40.mean_normal_edge_form(V, F), L, 1e-12)
    D = dd.d0(F, len(V))
    star1 = dd.hodge1(V, F)
    close(f"Proposition 40.3.5: {name}: (L x) = tilde d1 star1 d0 x = -d0^T star1 d0 x", -D.T @ (star1[:, None] * (D @ V)), L, 1e-12)
# per dual edge: (star1 d0 x)_e = -int_{e*} N x dx, e* from g (o = -1) to f (o = +1)
V, F = Vj, Fj
Nf = dm.face_normals(V, F)
cc = dd.circumcenters(V, F)
E_list, _ = dd.edge_list(F)
star1 = dd.hodge1(V, F)
ef = dm.edge_faces(F)
worst = 0.0
for e, (i, j) in enumerate(E_list):
    fs = ef[(int(i), int(j))]
    f = [h for h in fs if any(F[h, c] == i and F[h, (c + 1) % 3] == j for c in range(3))][0]
    gg = [h for h in fs if h != f][0]
    m = 0.5 * (V[i] + V[j])
    integral = np.cross(Nf[gg], m - cc[gg]) + np.cross(Nf[f], cc[f] - m)
    worst = max(worst, np.abs(star1[e] * (V[j] - V[i]) + integral).max())
check("Proposition 40.3.5: on every dual edge of the jittered icosphere, (star1 d0 x)_e = -int_{e*} N x dx", worst < 1e-12, worst)

# ---------------------------------------------------------------- Example 40.3.7
errs = []
for lev in (1, 2, 3, 4):
    V, F = dm.icosphere(lev)
    A = dm.mass_voronoi(V, F)
    check(f"Example 40.3.7(a): icosphere({lev}) has no obtuse triangle (mixed = circumcentric area)",
          dm.corner_angles(V, F).max() < np.pi / 2 and np.allclose(A, dd.dual_areas(V, F, "circumcentric")))
    errs.append(np.abs(dm.cotan_laplacian(V, F) @ V / A[:, None] + 2 * V).max())
check("Example 40.3.7(a): level 1 error at round-off", errs[0] < 1e-13)
close("Example 40.3.7(a): levels 2-4: 0.0284, 0.0150, 0.0076", errs[1:], [0.0284, 0.0150, 0.0076], 1e-4)
V, F = ddg40.octahedron()
close("Example 40.3.7(b): octahedron vertex e3: (L x)_i = -(4/sqrt 3) e3", (dm.cotan_laplacian(V, F) @ V)[4], [0, 0, -4 / np.sqrt(3)], 1e-12)
for pat in ("avoid", "max", "alt"):
    V, F = ddg40.unit_cube(pat)
    i7 = int(np.where((V == 1).all(1))[0][0])
    close(f"Example 40.3.7(c): unit cube ({pat}): (L x) at (1, 1, 1) = -(1, 1, 1)", (dm.cotan_laplacian(V, F) @ V)[i7], [-1, -1, -1], 1e-12)
wa = dm.cotan_weights(*ddg40.unit_cube("avoid"))
wm = dm.cotan_weights(*ddg40.unit_cube("max"))
check("Example 40.3.7(c): the individual cotan weights differ between the diagonal patterns", set(wa) != set(wm))
for n in (9, 17, 33):
    V, F = ddg40.saddle_grid(n, 0.5)
    o = int(np.argmin(np.linalg.norm(V, axis=1)))
    check(f"Example 40.3.7(d): saddle grid {n} x {n}: (L x) = 0 at the origin", np.abs((dm.cotan_laplacian(V, F) @ V)[o]).max() < 1e-12)
V, F = dm.torus_mesh(2, 0.8, 24, 48)
u = np.arctan2(V[:, 2], np.linalg.norm(V[:, :2], axis=1) - 2)
Hx = -(2 + 1.6 * np.cos(u)) / (1.6 * (2 + 0.8 * np.cos(u)))
close("Example 40.3.7(e): torus H_i = dgdec.mean_curvature within 0.0029 of H", np.abs(dd.mean_curvature(V, F) - Hx).max(), 0.0029, 1e-4)
close("Exercise 40.3.6: exact H from -0.8036 to -0.2083", [Hx.min(), Hx.max()], [-0.8036, -0.2083], 5e-5)

# ---------------------------------------------------------------- Non-example 40.3.8
s = np.linspace(0, 1, 7)
V, F = dm.grid_mesh(s, s)
inner = np.setdiff1d(np.arange(len(V)), dm.boundary_vertices(F))
W = V.copy()
W[inner, :2] += np.random.default_rng(2).uniform(-0.04, 0.04, (len(inner), 2))
LU = dm.uniform_laplacian(F, len(W)) @ W
Lc = dm.cotan_laplacian(W, F) @ W
close("Non-example 40.3.8: uniform Laplacian of length up to 0.057 in the plane; cotan < 1e-15; a cotan weight -0.36",
      [np.linalg.norm(LU[inner], axis=1).max(), np.abs(Lc[inner]).max() < 1e-15, min(dm.cotan_weights(W, F).values())], [0.057, 1, -0.36], 6e-4)
check("Non-example 40.3.8: the uniform Laplacian is tangent to the plane", np.allclose(LU[:, 2], 0))

# ---------------------------------------------------------------- Theorem 40.3.10 and Example 40.3.11
for name, (V, F) in tests + [("saddle star", ddg40.saddle_star(0.35)), ("example star", ddg40.example_star())]:
    KN = ddg40.curvature_normals(V, F)[2]
    idx = range(len(V)) if "star" not in name else [0]
    worst = max(np.abs(0.5 * ddg40.boundary_integral_N_cross_dN(V, F, i) - KN[i]).max() for i in idx)
    check(f"Theorem 40.3.10(a): {name}: (1/2) oint N x dN with great arcs = (K N)_i", worst < 1e-12, worst)
for name, (V, F), i in [("example star", ddg40.example_star(), 0), ("icosphere(1) vertex 0", dm.icosphere(1), 0),
                        ("cube corner", ddg40.unit_cube(), 7)]:
    KN = ddg40.curvature_normals(V, F)[2][i]
    Nk = dm.face_normals(V, F)[ddg40.faces_around(F, i)]
    c = Nk.sum(0) / np.linalg.norm(Nk.sum(0))
    va = sum(ddg40.spherical_triangle_vector_area(c, Nk[k], Nk[(k + 1) % len(Nk)], n=50) for k in range(len(Nk)))
    close(f"Theorem 40.3.10(b): {name}: (K N)_i = vector area of the Gauss image", va, KN, 2e-4)
V, F = ddg40.unit_cube()
KNc = ddg40.curvature_normals(V, F)[2][7]
close("Example 40.3.11(a): cube corner (K N) = (pi/4)(1, 1, 1), |.| = 1.360 < 1.571", list(KNc) + [np.linalg.norm(KNc), np.pi / 2],
      [np.pi / 4] * 3 + [1.360, 1.571], 5e-4)
er, ratio = [], []
for lev in (1, 2, 3, 4):
    V, F = dm.icosphere(lev)
    KN = ddg40.curvature_normals(V, F)[2]
    A = dm.mass_voronoi(V, F)
    er.append(np.abs(KN / A[:, None] - V).max())
    ratio.append((np.linalg.norm(KN, axis=1) / dm.angle_defect(V, F)).min())
close("Example 40.3.11(b): icosphere levels 1-4: max |(K N)_i / A_i - x_i| = 0.0527, 0.0174, 0.0064, 0.0028", er,
      [0.0527, 0.0174, 0.0064, 0.0028], 1e-4)
close("Theorem 40.3.10 (text): min |(K N)_i| / delta_i = 0.975 (level 1), 0.9995 (level 4)", [ratio[0], ratio[3]], [0.975, 0.9995], 5e-4)
check("Theorem 40.3.10 (text): |(K N)_i| <= delta_i on every icosphere vertex", all(r <= 1 for r in ratio))
V, F = ddg40.saddle_star(0.35)
th = np.array([t for t, ell in dm.edge_bending_angles(V, F).values()])
AN, Lx, KN = ddg40.curvature_normals(V, F)
close("Example 40.3.11(c), Figure 40.3.1(b): saddle star bending angles +-1.0394, (K N) = (0, 0, -1.0301), A = (0, 0, 0.866), Lx = 0",
      [np.abs(th).min(), np.abs(th).max(), np.sum(th > 0), *KN[0], AN[0, 2], np.abs(Lx[0]).max()],
      [1.0394, 1.0394, 3, 0, 0, -1.0301, 0.866, 0], 5e-5)
V, F = ddg40.example_star()
AN, Lx, KN = ddg40.curvature_normals(V, F)
close("Example 40.3.11(c), Figure 40.3.1(a): example star A_i, (L x)_i, (K N)_i", [*AN[0], *Lx[0], *KN[0]],
      [-0.026, 0.026, 0.770, 0.035, -0.022, -1.623, -0.011, -0.002, 0.778], 6e-4)
close("Example 40.3.11(c): (K N)_i = (-0.0109, -0.0020, 0.7777)", KN[0], [-0.0109, -0.0020, 0.7777], 6e-5)
for n, val in ((17, -3.949), (33, -3.987)):
    V, F = ddg40.saddle_grid(n, 0.5)
    o = int(np.argmin(np.linalg.norm(V, axis=1)))
    KN = ddg40.curvature_normals(V, F)[2]
    close(f"Example 40.3.11(d): saddle grid {n}: (K N)_i / A_i = {val} e3", KN[o] / dm.mass_voronoi(V, F)[o], [0, 0, val], 6e-4)

# ---------------------------------------------------------------- Figure 40.3.2 numbers
V, F = dm.torus_mesh(2, 0.8, 24, 48)
AN, Lx, KN = ddg40.curvature_normals(V, F)
A = dm.mass_voronoi(V, F)
ring = np.where((np.abs(V[:, 1]) < 1e-12) & (V[:, 0] > 0))[0]
uu_ = np.arctan2(V[ring, 2], V[ring, 0] - 2)
i0, ip = ring[np.argmin(np.abs(uu_))], ring[np.argmin(np.abs(np.abs(uu_) - np.pi))]
close("Figure 40.3.2: |(L x)_i| / A_i = 1.606 (outer), 0.422 (inner); 2|H| = 1.607, 0.417",
      [np.linalg.norm(Lx[i0]) / A[i0], np.linalg.norm(Lx[ip]) / A[ip], 2 * 0.80357, 2 * 0.20833], [1.606, 0.422, 1.607, 0.417], 6e-4)
close("Figure 40.3.2: (K N)_i / A_i = +0.448 (outer, outward), 1.030 toward the tube center (inner); K = 0.446, -1.042",
      [KN[i0, 0] / A[i0], KN[ip, 0] / A[ip], 1 / (0.8 * 2.8), 1 / (0.8 * 1.2)], [0.448, 1.030, 0.446, 1.042], 6e-4)
N = dm.vertex_normals(V, F, "area")
close("Smooth vs. Discrete: tangential part of (L x)_i on the torus at most 0.75 %",
      (np.linalg.norm(np.cross(Lx, N), axis=1) / np.linalg.norm(Lx, axis=1)).max(), 0.0075, 1e-4)

# ---------------------------------------------------------------- Conventions box: the slide formulas in our notation
V, F = dm.icosphere(3)
A = dm.mass_voronoi(V, F)
L = dm.cotan_laplacian(V, F) @ V
slide23 = np.zeros_like(V)
slide25 = np.zeros_like(V)
for (i, j), w in dm.cotan_weights(V, F).items():
    slide23[i] += w * (V[i] - V[j])
    slide23[j] += w * (V[j] - V[i])
for (i, j), (t, ell) in dm.edge_bending_angles(V, F).items():
    slide25[i] += 0.5 * t / ell * (V[j] - V[i])
    slide25[j] += 0.5 * t / ell * (V[i] - V[j])
check("Conventions: slide 23's (HN)_i = -(L x)_i and points outward on the unit sphere",
      np.allclose(slide23, -L) and np.all(np.sum(slide23 * V, 1) > 0))
check("Conventions: slide 25's (KN)_i with (f_j - f_i) points inward on the unit sphere (opposite to K N = +x A)",
      np.all(np.sum(slide25 * V, 1) < 0) and np.allclose(slide25, -ddg40.curvature_normals(V, F)[2]))
check("Conventions: on the unit sphere Delta x = -2 x, while Crane's H N (H = +1 outward) is +x: Delta f = -2 H_Crane N",
      np.all(np.sum(L / A[:, None] * V, 1) < -1.9))

# ---------------------------------------------------------------- Exercises 40.3.1, 40.3.2, 40.3.4, 40.3.5
V, F = ddg40.octahedron()
th = dm.edge_bending_angles(V, F)[(0, 4)][0]
KN = ddg40.curvature_normals(V, F)[2][4]
close("Exercise 40.3.1: octahedron: theta = arccos(1/3), sin(theta/2) = 1/sqrt 3, (K N) = sqrt 2 arccos(1/3) e3 = 1.7408 e3",
      [th, np.sin(th / 2), *KN, 2 * np.pi / 3], [np.arccos(1 / 3), 1 / np.sqrt(3), 0, 0, np.sqrt(2) * np.arccos(1 / 3), 2.0944], 1e-4)
close("Exercise 40.3.1: (K N) = 1.7408", KN[2], 1.7408, 1e-4)
ph, la = sp.symbols("phi lambda", positive=True)
U = sp.Matrix([sp.sin(ph) * sp.cos(la), sp.sin(ph) * sp.sin(la), sp.cos(ph)])
octant = [sp.integrate(sp.integrate(U[c] * sp.sin(ph), (ph, 0, sp.pi / 2)), (la, 0, sp.pi / 2)) for c in range(3)]
check("Exercise 40.3.2: the vector area of the octant triangle is (pi/4)(1, 1, 1)", all(sp.simplify(o - sp.pi / 4) == 0 for o in octant))
for name, (V, F) in [("icosphere(3)", dm.icosphere(3)), ("torus", dm.torus_mesh(2, 0.8, 24, 48)), ("genus2_mesh(0.16)", dm.genus2_mesh(0.16))]:
    AN, Lx, KN = ddg40.curvature_normals(V, F)
    check(f"Exercise 40.3.4: {name}: sum (L x)_i = 0, sum (K N)_i = 0, sum A_i = 0",
          np.abs(Lx.sum(0)).max() < 1e-10 and np.abs(KN.sum(0)).max() < 1e-10 and np.abs(AN.sum(0)).max() < 1e-10)
V, F = dm.torus_mesh(2, 0.8, 12, 20)
a1 = ddg40.curvature_normals(V, F)
a2 = ddg40.curvature_normals(V, F[:, ::-1])
check("Exercise 40.3.5(b), (c): reversing the orientation flips (K N)_i and keeps (L x)_i",
      np.allclose(a2[2], -a1[2], atol=1e-12) and np.allclose(a2[1], a1[1], atol=1e-12))

summary()

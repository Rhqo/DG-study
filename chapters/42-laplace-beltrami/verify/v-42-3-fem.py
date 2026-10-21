"""Verification for Section 42.3 (Discretization via FEM).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-3-fem.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch42  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "42-3-fem.html"
rng = np.random.default_rng(4203)


def angle(p, q, r):
    a, b = q - p, r - p
    return np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


# ---------------------------------------------------------------- Lemma 42.3.2 (random triangles in R^3, some obtuse)
ok, nobt = True, 0
for _ in range(200):
    P = rng.normal(size=(3, 3))
    G = dd.barycentric_gradients(P)
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    th = [angle(P[c], P[(c + 1) % 3], P[(c + 2) % 3]) for c in range(3)]
    nobt += max(th) > np.pi / 2
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        ok &= abs(A * G[i] @ G[j] + 0.5 / np.tan(th[k])) < 1e-9
        ok &= abs(A * G[i] @ G[i] - 0.5 * (1 / np.tan(th[j]) + 1 / np.tan(th[k]))) < 1e-9
check(f"Lemma 42.3.2: A <grad phi_i, grad phi_j> = -cot(theta_k)/2 and A |grad phi_i|^2 = (cot + cot)/2 "
      f"(200 random triangles, {nobt} obtuse)", ok and nobt > 50)

# ---------------------------------------------------------------- Exercise 42.3.1, the example after Definition 42.3.3
V1 = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
F1 = np.array([[0, 1, 2]])
close("Definition 42.3.3, Exercise 42.3.1: stiffness of the right isosceles triangle",
      ch42.stiffness(V1, F1), 0.5 * np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]), tol=1e-14)
close("Exercise 42.3.1: M^G = (1/24)[[2,1,1],[1,2,1],[1,1,2]], lumped M = I/6",
      [ch42.galerkin_mass(V1, F1), np.diag(ch42.galerkin_mass(V1, F1).sum(1))],
      [np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 24, np.eye(3) / 6], tol=1e-15)

# ---------------------------------------------------------------- Theorem 42.3.5 and Corollary 42.3.6 on several meshes
Vg, Fg = dm.grid_mesh(np.linspace(0, 1, 6), np.linspace(0, 1, 5))
Vd, Fd = ch42.disk_mesh(8)
Vn = np.array([[0.0, 0, 0], [2.0, 0, 0], [1.0, 0.3, 0], [1.0, -0.25, 0]])
Fn = np.array([[0, 1, 2], [1, 0, 3]])
meshes = {"icosphere(2)": dm.icosphere(2), "jittered sphere": ch42.jittered_sphere(2, s=0.15, seed=42),
          "torus_mesh(2, 0.8, 12, 24)": dm.torus_mesh(2.0, 0.8, 12, 24), "grid 5x4": (Vg, Fg),
          "disk mesh (boundary)": (Vd, Fd), "non-Delaunay pair": (Vn, Fn)}
for name, (V, F) in meshes.items():
    L = dm.cotan_laplacian(V, F)
    S = ch42.stiffness(V, F)
    D0 = dd.d0(F, len(V))
    lam = np.linalg.eigvalsh(-L)
    u = rng.normal(size=len(V))
    check(f"Theorem 42.3.5 on {name}: stiffness = -L = d0^T star1 d0",
          np.abs(S + L).max() < 1e-11 and np.abs(D0.T @ (dd.hodge1(V, F)[:, None] * D0) + L).max() < 1e-11)
    check(f"Corollary 42.3.6 on {name}: -u^T L u = int |grad W0 u|^2, -L >= 0, kernel = constants",
          abs(-u @ L @ u - dm.dirichlet_energy(V, F, u)) < 1e-9 * max(1, abs(u @ L @ u)) and lam[0] > -1e-10
          and lam[1] > 1e-8)

# ---------------------------------------------------------------- Example 42.3.7
V, F = ch42.jittered_sphere(3, s=0.15, seed=42)
L = dm.cotan_laplacian(V, F)
w = np.array(list(dm.cotan_weights(V, F).values()))
lam = np.linalg.eigvalsh(-L)
check("Example 42.3.7: 642 vertices, 1280 faces (168 obtuse), 1920 edges (31 negative weights, min -0.499)",
      len(V) == 642 and len(F) == 1280 and (dm.corner_angles(V, F).max(1) > np.pi / 2).sum() == 168 and len(w) == 1920
      and (w < 0).sum() == 31 and round(w.min(), 3) == -0.499)
close("Example 42.3.7: eigenvalues of -L start 0, 0.0388", [lam[0], lam[1]], [0.0, 0.0388], tol=5e-5)

# ---------------------------------------------------------------- Proposition 42.3.9, Non-example 42.3.10, Proposition 42.3.12
bary, wq = dd.triangle_quadrature(4)
ok = True
for _ in range(20):
    P = rng.normal(size=(3, 3))
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    Mq = A * np.einsum("q,qa,qb->ab", wq, bary, bary)
    ok &= np.allclose(Mq, A / 12 * (np.ones((3, 3)) + np.eye(3)), atol=1e-13)
check("Proposition 42.3.9: int_f phi_i phi_j = (A_f / 12)(1 + delta_ij) (quadrature on 20 random triangles)", ok)
MG = ch42.galerkin_mass(V, F)
check("Proposition 42.3.9: M^G is symmetric positive definite", np.allclose(MG, MG.T) and np.linalg.eigvalsh(MG)[0] > 0)
check("Proposition 42.3.12: row sums of M^G = barycentric dual areas = dgdec.hodge0(barycentric)",
      np.allclose(MG.sum(1), dm.mass_barycentric(V, F)) and np.allclose(MG.sum(1), dd.hodge0(V, F)))
tot = [ch42.galerkin_mass(*dm.icosphere(lv)).sum() for lv in (1, 2, 3)]
close("Non-example 42.3.10: 1^T M^G 1 = 11.67, 12.33, 12.51 on icosphere(1)-(3)", tot, [11.67, 12.33, 12.51], tol=6e-3)
check("Non-example 42.3.10: n0 = 42, 162, 642", [len(dm.icosphere(lv)[0]) for lv in (1, 2, 3)] == [42, 162, 642])

# ---------------------------------------------------------------- Proposition 42.3.13: (42.3.2) by quadrature equals L u = M^G f
V, F = dm.icosphere(1)
u, f = rng.normal(size=len(V)), rng.normal(size=len(V))
lhs, rhs = np.zeros(len(V)), np.zeros(len(V))
for tri in F:
    P = V[tri]
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    G = dd.barycentric_gradients(P)
    gu = u[tri] @ G
    fq = bary @ f[tri]
    for a in range(3):
        lhs[tri[a]] += A * gu @ G[a]
        rhs[tri[a]] -= A * np.sum(wq * fq * bary[:, a])
L = dm.cotan_laplacian(V, F)
close("Proposition 42.3.13: (42.3.2) assembled by quadrature: lhs = -(L u), rhs = -(M^G f)",
      [lhs, rhs], [-(L @ u), -(ch42.galerkin_mass(V, F) @ f)], tol=1e-12)

# ---------------------------------------------------------------- Figure 42.3.1
xi, xj, xk = np.array([0.0, 0, 0]), np.array([2.0, 0, 0]), np.array([0.9, 1.3, 0])
vals = {}
for key, xl in (("a", np.array([1.1, -1.1, 0])), ("b", np.array([1.0, -0.35, 0]))):
    Vf = np.array([xi, xj, xk, xl])
    Ff = np.array([[0, 1, 2], [1, 0, 3]])
    Sf = ch42.stiffness(Vf, Ff)
    vals[key] = (Sf[0, 1], np.degrees(angle(xk, xi, xj)), np.degrees(angle(xl, xi, xj)))
close("Figure 42.3.1: a_ij = -0.185 (a) and +0.492 (b)", [vals["a"][0], vals["b"][0]], [-0.185, 0.492], tol=6e-4)
close("Figure 42.3.1: opposite angles 75, 84 (a) and 75, 141 (b) degrees", [vals["a"][1], vals["a"][2], vals["b"][2]],
      [75, 84, 141], tol=0.6)
close("Figure 42.3.1: per-face contributions -0.135, -0.050, +0.627",
      [-0.5 / np.tan(np.radians(vals["a"][1])), -0.5 / np.tan(np.radians(vals["a"][2])), -0.5 / np.tan(np.radians(vals["b"][2]))],
      [-0.135, -0.050, 0.627], tol=6e-4)

# ---------------------------------------------------------------- Example 42.3.14, Figure 42.3.2
def poisson_error(V, F, kind):
    L = dm.cotan_laplacian(V, F)
    z = V[:, 2]
    if kind == "galerkin":
        M_ = ch42.galerkin_mass(V, F)
        b, m = M_ @ (-2 * z), M_.sum(1)
    else:
        m = ch42.mass(V, F, kind)
        b = m * (-2 * z)
    b = b - m * (b.sum() / m.sum())
    uu = np.linalg.solve(-L + np.outer(m, m), -b)
    e = uu - (z - (m @ z) / m.sum())
    return np.sqrt((e ** 2 * m).sum() / m.sum())


eg = [poisson_error(*dm.icosphere(lv), "galerkin") for lv in range(1, 5)]
eb = [poisson_error(*dm.icosphere(lv), "barycentric") for lv in range(1, 5)]
close("Example 42.3.14: Galerkin errors 4.9e-2, 1.3e-2, 3.3e-3, 8.3e-4 (relative)", np.array(eg) / [4.9e-2, 1.3e-2, 3.3e-3, 8.3e-4],
      np.ones(4), tol=0.02)
close("Example 42.3.14: lumped errors 3.6e-3, 7.7e-4, 1.7e-4, 4.1e-5 (relative)", np.array(eb) / [3.6e-3, 7.7e-4, 1.7e-4, 4.1e-5],
      np.ones(4), tol=0.03)
hs = [dd.primal_edge_lengths(*dm.icosphere(lv)).mean() for lv in range(1, 5)]
close("Example 42.3.14: mean edge length 0.58 ... 0.075", [hs[0], hs[-1]], [0.58, 0.075], tol=6e-3)
ratios = np.array(eg[:-1]) / np.array(eg[1:])
check("Example 42.3.14: errors drop by about 4 per level", np.all((3.4 < ratios) & (ratios < 4.6)))
ejg = [poisson_error(*ch42.jittered_sphere(lv, s=0.15, seed=42), "galerkin") for lv in range(1, 5)]
ejb = [poisson_error(*ch42.jittered_sphere(lv, s=0.15, seed=42), "barycentric") for lv in range(1, 5)]
slopes = [np.polyfit(np.log(hs[1:]), np.log(e[1:]), 1)[0] for e in (eg, eb, ejg, ejb)]
close("Figure 42.3.2: fitted slopes 2.00, 2.13, 1.97, 2.44 (jittered: own edge lengths)",
      [slopes[0], slopes[1]], [2.00, 2.13], tol=0.01)
hsj = [dd.primal_edge_lengths(*ch42.jittered_sphere(lv, s=0.15, seed=42)).mean() for lv in range(1, 5)]
sj = [np.polyfit(np.log(hsj[1:]), np.log(e[1:]), 1)[0] for e in (ejg, ejb)]
close("Figure 42.3.2: jittered slopes 1.97, 2.44", sj, [1.97, 2.44], tol=0.01)
rat = np.concatenate([np.array(eg) / np.array(eb), np.array(ejg) / np.array(ejb)])
check(f"Figure 42.3.2: lumped errors are 5-20 times smaller here (ratios {np.round(rat, 1)})", np.all((5 <= rat) & (rat <= 21)))

# ---------------------------------------------------------------- Implementation 42.3
ns, same, printed = ch42.run_printed(PAGE, "code-42-3-1", "out-42-3-1")
check("Implementation 42.3: code-42-3-1 prints exactly out-42-3-1", same, printed)
check("Implementation 42.3: code-42-3-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-42-3-1") <= 30)
St, MGt = ns["fem_matrices"](*dm.torus_mesh(2.0, 0.8, 8, 16))
check("Implementation 42.3: fem_matrices agrees with ch42.stiffness and ch42.galerkin_mass on a torus",
      np.allclose(St, ch42.stiffness(*dm.torus_mesh(2.0, 0.8, 8, 16)))
      and np.allclose(MGt, ch42.galerkin_mass(*dm.torus_mesh(2.0, 0.8, 8, 16))))

# ---------------------------------------------------------------- Exercise 42.3.4
V, F = ch42.jittered_sphere(2, s=0.15, seed=42)
MG, m = ch42.galerkin_mass(V, F), dm.mass_barycentric(V, F)
ok = True
for _ in range(50):
    v = rng.normal(size=len(V))
    q, ql = v @ MG @ v, v @ (m * v)
    ok &= (0.25 * ql - 1e-12 <= q <= ql + 1e-12)
check("Exercise 42.3.4: (1/4) v^T M v <= v^T M^G v <= v^T M v (50 random v)", ok)
one = np.ones(len(V))
close("Exercise 42.3.4: equality on the right for constants", one @ MG @ one, one @ (m * one), tol=1e-12)
n = 9
Vs, Fs = dm.grid_mesh(np.arange(n, dtype=float), np.arange(n, dtype=float), pattern="same")
ij = np.array([(i, j) for i in range(n) for j in range(n)])
col3 = np.array([1.0, -1.0, 0.0])[(ij[:, 0] + ij[:, 1]) % 3]
check("Exercise 42.3.4: the 3-coloring (1, -1, 0) sums to 0 on every face of grid_mesh(pattern='same')",
      np.abs(col3[Fs].sum(1)).max() == 0)
close("Exercise 42.3.4: equality on the left for the 3-coloring", col3 @ ch42.galerkin_mass(Vs, Fs) @ col3,
      0.25 * col3 @ (dm.mass_barycentric(Vs, Fs) * col3), tol=1e-12)

# ---------------------------------------------------------------- Exercise 42.3.5
ns5, same5, printed5 = ch42.run_printed(PAGE, "code-ex-42-3-5", "out-ex-42-3-5")
check("Exercise 42.3.5: code-ex-42-3-5 prints exactly out-ex-42-3-5", same5, printed5)
check("Exercise 42.3.5: code block is at most 30 lines", dgcode.code_lines(PAGE, "code-ex-42-3-5") <= 30)

# ---------------------------------------------------------------- Exercise 42.3.6 (torus)
R, r = 2.0, 0.8
res = []
for nu, nv in ((12, 24), (16, 32), (24, 48)):
    V, F = dm.torus_mesh(R, r, nu, nv)
    L = dm.cotan_laplacian(V, F)
    uang = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]) - R)
    f = -np.sin(uang) * (R + 2 * r * np.cos(uang)) / (r * (R + r * np.cos(uang)))
    row = [dd.primal_edge_lengths(V, F).mean()]
    for kind in ("galerkin", "barycentric"):
        if kind == "galerkin":
            M_ = ch42.galerkin_mass(V, F)
            b, mm = M_ @ f, M_.sum(1)
        else:
            mm = ch42.mass(V, F, kind)
            b = mm * f
        b = b - mm * (b.sum() / mm.sum())
        uu = np.linalg.solve(-L + np.outer(mm, mm), -b)
        e = uu - (V[:, 2] - (mm @ V[:, 2]) / mm.sum())
        row.append(np.sqrt((e ** 2 * mm).sum() / mm.sum()))
    res.append(row)
res = np.array(res)
close("Exercise 42.3.6: mean edge lengths 0.536, 0.403, 0.269", res[:, 0], [0.536, 0.403, 0.269], tol=6e-4)
close("Exercise 42.3.6: Galerkin errors 3.2e-2, 1.8e-2, 8.1e-3 (relative)", res[:, 1] / [3.2e-2, 1.8e-2, 8.1e-3], np.ones(3),
      tol=0.02)
close("Exercise 42.3.6: lumped errors 7.1e-3, 4.1e-3, 1.8e-3 (relative)", res[:, 2] / [7.1e-3, 4.1e-3, 1.8e-3], np.ones(3),
      tol=0.02)
check("Exercise 42.3.6: lumped errors are about 4 times smaller", np.all(np.abs(res[:, 1] / res[:, 2] - 4.4) < 0.3))
V, F = dm.torus_mesh(R, r, 24, 48)
w = dd.hodge1(V, F)
check("Exercise 42.3.6: torus_mesh(2, 0.8, 24, 48) has 2304 triangles, 1152 obtuse ones, 1152 zero weights (diagonals), "
      "no negative weight", len(F) == 2304 and (dm.corner_angles(V, F).max(1) > np.pi / 2).sum() == 1152
      and (np.abs(w) < 1e-12).sum() == 1152 and np.all(w > -1e-12))

summary()

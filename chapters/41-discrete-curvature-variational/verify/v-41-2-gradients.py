"""Verification for Section 41.2 (Volume, Area, and Mean Curvature Gradients).

Run from the project root: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/verify/v-41-2-gradients.py
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
import ddg41  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "41-2-gradients.html"
rng = np.random.default_rng(41202)

# ---------------------------------------------------------------- Implementation 41.2 and Exercise 41.2.6
check("Implementation 41.2: at most 30 lines", dgcode.code_lines(PAGE, "code-41-2-1") <= 30)
ns, out = ddg41.run_printed(PAGE, "code-41-2-1")
check("Implementation 41.2: printed output matches the page", out.strip() == ddg41.printed_block(PAGE, "out-41-2-1").strip(), out)
gV, gA, gM = ns["gradients"](ns["V"], ns["F"])
close("Implementation 41.2: gradients = lib (vector area, -L x, (K N))",
      np.array([gV, gA, gM]), np.array([ddg41.vector_area(ns["V"], ns["F"]), dd.area_gradient(ns["V"], ns["F"]),
                                        ddg41.gauss_curvature_normal(ns["V"], ns["F"])]), 1e-11)
ns2, out2 = ddg41.run_printed(PAGE, "code-ex-41-2-6", dict(ns))
check("Exercise 41.2.6: printed output matches the page", out2.strip() == ddg41.printed_block(PAGE, "out-ex-41-2-6").strip(), out2)

# ---------------------------------------------------------------- Proposition 41.2.1 and Example 41.2.2
r_ = sp.symbols("r", positive=True)
sym_equal("Example 41.2.2(a): dV/dr = int dA (sphere)", sp.diff(sp.Rational(4, 3) * sp.pi * r_ ** 3, r_), 4 * sp.pi * r_ ** 2)
sym_equal("Example 41.2.2(a): dA/dr = -int 2 H dA with H = -1/r", sp.diff(4 * sp.pi * r_ ** 2, r_), -2 * (-1 / r_) * 4 * sp.pi * r_ ** 2)
sym_equal("Example 41.2.2(a): d(int H dA)/dr = -int K dA", sp.diff(-1 / r_ * 4 * sp.pi * r_ ** 2, r_), -(1 / r_ ** 2) * 4 * sp.pi * r_ ** 2)
ex = dgsym.EXAMPLES["sphere"]
Ks, Hs = dgsym.K_H(ex["expr"], *ex["coords"], ex["positive"])
check("Example 41.2.2(a): dgsym sphere H = -1/r, K = 1/r^2 (outward normal)",
      sp.simplify(Hs + 1 / ex["params"][0]) == 0 and sp.simplify(Ks - 1 / ex["params"][0] ** 2) == 0)
# torus: exact formulas for H, K with the outward normal, checked against dgsym
ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
R, r = ex["params"]
Kd, Hin = dgsym.K_H(ex["expr"], u, v, ex["positive"])
H = -(R + 2 * r * sp.cos(u)) / (2 * r * (R + r * sp.cos(u)))
K = sp.cos(u) / (r * (R + r * sp.cos(u)))
dom = {u: (0, 6.2), v: (0, 6.2), R: (1.5, 3), r: (0.2, 0.9)}
sym_equal("Example 41.2.2(b): torus H (outward N) = -dgsym H (x_u x x_v inward)", H, -Hin, dom)
sym_equal("Example 41.2.2(b): torus K", K, Kd, dom)
Nout = -dgsym.unit_normal(ex["expr"], u, v, ex["positive"])
X = sp.Matrix(ex["expr"])
phR = sp.simplify(Nout.dot(X.diff(R)))
phr = sp.simplify(Nout.dot(X.diff(r)))
sym_equal("Example 41.2.2(b): phi_R = cos u", phR, sp.cos(u), dom)
sym_equal("Example 41.2.2(b): phi_r = 1", phr, 1, dom)
dA_ = r * (R + r * sp.cos(u))
I = lambda f: sp.integrate(sp.integrate(sp.expand(f * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi))  # noqa: E731
Vt, At = 2 * sp.pi ** 2 * R * r ** 2, 4 * sp.pi ** 2 * R * r
sym_equal("Example 41.2.2(b): int dA = 4 pi^2 R r", I(1), At)
sym_equal("Example 41.2.2(b): int cos u dA = 2 pi^2 r^2 = dV/dR", I(sp.cos(u)), sp.diff(Vt, R))
sym_equal("Example 41.2.2(b): -int 2 H cos u dA = 4 pi^2 r = dA/dR",
          sp.integrate(sp.integrate(sp.simplify(-2 * H * sp.cos(u) * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), sp.diff(At, R))
TH = sp.integrate(sp.integrate(sp.simplify(H * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi))
sym_equal("Example 41.2.2(b): int H dA = -2 pi^2 R", TH, -2 * sp.pi ** 2 * R)
sym_equal("Example 41.2.2(b): -int K cos u dA = -2 pi^2 = d(int H)/dR",
          sp.integrate(sp.integrate(sp.simplify(-K * sp.cos(u) * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), sp.diff(TH, R))
sym_equal("Example 41.2.2(b): dV/dr = int dA", sp.diff(Vt, r), At)
sym_equal("Example 41.2.2(b): dA/dr = -int 2 H dA = 4 pi^2 R",
          sp.integrate(sp.integrate(sp.simplify(-2 * H * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), sp.diff(At, r))
sym_equal("Example 41.2.2(b): d(int H)/dr = 0 = -int K dA",
          sp.integrate(sp.integrate(sp.simplify(-K * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), sp.diff(TH, r))
vals = {R: 2, r: sp.Rational(4, 5)}
close("Example 41.2.2(b): numbers at (R, r) = (2, 0.8): 12.633, 31.583, -19.739",
      [float(sp.diff(Vt, R).subs(vals)), float(sp.diff(At, R).subs(vals)), float(-2 * sp.pi ** 2)], [12.633, 31.583, -19.739], 6e-4)
close("Figure 41.2.1: A = 63.17", float(At.subs(vals)), 63.17, 6e-3)
# Exercise 41.2.5: Minkowski formulas on the torus and the sphere
sym_equal("Exercise 41.2.5: torus int H <x, N> dA = -A",
          sp.integrate(sp.integrate(sp.simplify(H * X.dot(Nout) * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), -At)
sym_equal("Exercise 41.2.5: torus int K <x, N> dA = -int H dA",
          sp.integrate(sp.integrate(sp.simplify(K * X.dot(Nout) * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), -TH)
sym_equal("Exercise 41.2.5: torus int <x, N> dA = 3 V",
          sp.integrate(sp.integrate(sp.simplify(X.dot(Nout) * dA_), (u, 0, 2 * sp.pi)), (v, 0, 2 * sp.pi)), 3 * Vt)

# ---------------------------------------------------------------- Proposition 41.2.3, Theorem 41.2.4, Example 41.2.5
def Vq(V, F, q):
    W = V - q
    return np.einsum("ij,ij->i", W[F[:, 0]], np.cross(W[F[:, 1]], W[F[:, 2]])).sum() / 6


meshes = {"noisy icosphere(1)": (dm.icosphere(1)[0] * (1 + 0.2 * rng.uniform(-1, 1, (42, 1))), dm.icosphere(1)[1]),
          "torus_mesh(2, 0.8, 8, 16)": dm.torus_mesh(2, 0.8, 8, 16), "cube": ddg41.cube_mesh(1, center=(0.5, 0.5, 0.5))}
for name, (V, F) in meshes.items():
    vals_q = [Vq(V, F, rng.normal(size=3) * 3) for _ in range(3)]
    close(f"Proposition 41.2.3: V_q independent of q ({name})", np.array(vals_q), np.full(3, dm.signed_volume(V, F)), 1e-11)
    g = ddg41.central_gradient(lambda Y: dm.signed_volume(Y, F), V)
    close(f"Theorem 41.2.4: grad V = (1/3) sum A_f N_f by finite differences ({name})", g, ddg41.vector_area(V, F), 1e-7)
    close(f"Theorem 41.2.4: (1/6) sum x_j x x_k form ({name})", dd.signed_volume_gradient(V, F), ddg41.vector_area(V, F), 1e-13)
V, F = meshes["noisy icosphere(1)"]
W = V.copy()
W[5] += np.array([0.1, -0.05, 0.07])
check("Theorem 41.2.4: grad_{x_5} V does not change when x_5 moves", np.abs(ddg41.vector_area(W, F)[5] - ddg41.vector_area(V, F)[5]).max() < 1e-14)
check("Proposition 41.2.3: unit sphere icosphere(4) volume close to 4 pi / 3 from below",
      0.98 * 4 * np.pi / 3 < dm.signed_volume(*dm.icosphere(4)) < 4 * np.pi / 3)
for lev, want in zip((1, 2, 3, 4), (0.060, 0.028, 0.012, 0.0055)):
    V, F = dm.icosphere(lev)
    e = np.abs(ddg41.vector_area(V, F) / dm.mass_barycentric(V, F)[:, None] - V).max()
    close(f"Example 41.2.5(a): icosphere({lev}) max |grad V / A_i - x_i| = {want}", e, want, 0.6e-3 if lev < 4 else 1e-4)
V, F = ddg41.cube_mesh(1, center=(0.5, 0.5, 0.5))
gVc = ddg41.vector_area(V, F)
nf = np.array([np.sum(np.any(F == i, axis=1)) for i in range(len(V))])
c111 = int(np.argmin(np.linalg.norm(V - 1, axis=1)))
close("Example 41.2.5(b): corner (1, 1, 1) has six triangles and grad V = (1/3)(1, 1, 1)", np.r_[nf[c111], gVc[c111]], [6, 1 / 3, 1 / 3, 1 / 3], 1e-14)
c100 = int(np.argmin(np.linalg.norm(V - np.array([1.0, 0, 0]), axis=1)))
close("Example 41.2.5(b): corner (1, 0, 0): grad V = (1/3, -1/6, -1/6)", gVc[c100], [1 / 3, -1 / 6, -1 / 6], 1e-14)
for i in range(len(V)):
    for axis in range(3):
        sq = [f for f in range(len(F)) if i in F[f] and np.allclose(V[F[f], axis], V[i, axis])]
        if abs(abs(gVc[i, axis]) - dm.face_areas(V, F)[sq].sum() / 3) > 1e-14:
            check(f"Example 41.2.5(b): component rule at corner {V[i]}", False)
check("Example 41.2.5(b): every component = (1/3) area of the triangles of that square at the corner", True)
check("Example 41.2.5(b): grad A = (1, 1, 1) at (1, 1, 1) whatever the diagonals",
      np.allclose(dd.area_gradient(V, F)[c111], 1, atol=1e-14))
close("Example 41.2.5(c): cube V_q = 1 for q = 0 and q = (3, -1, 2)", [Vq(V, F, np.zeros(3)), Vq(V, F, np.array([3, -1, 2.0]))], [1, 1], 1e-13)

# ---------------------------------------------------------------- Lemma 41.2.6 and Theorem 41.2.7
F1 = np.array([[0, 1, 2]])
for t in range(5):
    P = rng.normal(size=(3, 3))
    th = dm.corner_angles(P, F1)[0]
    N = dm.face_normals(P, F1)[0]
    close(f"Lemma 41.2.6: N x (x_k - x_j) = cot th_k (x_i - x_j) + cot th_j (x_i - x_k) (random triangle {t})",
          np.cross(N, P[2] - P[1]), (P[0] - P[1]) / np.tan(th[2]) + (P[0] - P[2]) / np.tan(th[1]), 1e-11)
for name, (V, F) in list(meshes.items()) + [("saddle grid 9 x 9 (boundary)", ddg41.saddle_grid(9, 0.5))]:
    g = ddg41.central_gradient(lambda Y: dm.face_areas(Y, F).sum(), V)
    close(f"Theorem 41.2.7: grad A = -(L x) by finite differences, boundary vertices included ({name})",
          g, -dm.cotan_laplacian(V, F) @ V, 1e-8)
s = np.linspace(0, 1, 6)
V, F = dm.grid_mesh(s, s)
inner = (V[:, 0] > 0) & (V[:, 0] < 1) & (V[:, 1] > 0) & (V[:, 1] < 1)
V[inner, :2] += 0.06 * rng.uniform(-1, 1, (inner.sum(), 2))
check("Example 41.2.8(a): planar mesh, interior grad A = 0", np.abs(dd.area_gradient(V, F)[inner]).max() < 1e-14)
V, F = dm.icosphere(3)
close("Example 41.2.8(b): icosphere(3) max |grad A / A_i - 2 x_i| = 0.0150",
      np.abs(dd.area_gradient(V, F) / dm.mass_voronoi(V, F)[:, None] - 2 * V).max(), 0.0150, 5e-5)
for name, (V, F) in meshes.items():
    L = dm.cotan_laplacian(V, F)
    close(f"Example 41.2.8(c): -1/2 sum_c (x^c)^T L x^c = A ({name})", -0.5 * np.trace(V.T @ L @ V), dm.face_areas(V, F).sum(), 1e-11)


# ---------------------------------------------------------------- Theorem 41.2.9 (Schlaefli), 41.2.11, Corollary 41.2.12
def schlafli_local(V, F):
    """sum over edges of l_e grad_{x_i} theta_e from Proposition 41.1.8, for every vertex i"""
    out = np.zeros_like(V)
    for (i, j), fs in dm.edge_faces(F).items():
        if len(fs) != 2:
            continue
        hinge = {}
        for f in fs:                                              # f traverses i -> j (role f) or j -> i (role g)
            t = list(F[f])
            a = t.index(i)
            hinge["f" if t[(a + 1) % 3] == j else "g"] = [x for x in t if x not in (i, j)][0]
        grads = ddg41.bending_angle_gradients(V[i], V[j], V[hinge["f"]], V[hinge["g"]])
        ell = np.linalg.norm(V[j] - V[i])
        for vtx, gv in zip((i, j, hinge["f"], hinge["g"]), grads):
            out[vtx] += ell * gv
    return out


for name, (V, F) in meshes.items():
    check(f"Theorem 41.2.9: sum_e l_e grad_i theta_e = 0 at every vertex (analytic, {name})", np.abs(schlafli_local(V, F)).max() < 1e-12)
    bend = lambda Y: np.array([t for t, _ in dm.edge_bending_angles(Y, F).values()])  # noqa: E731
    ells = np.array([ell for _, ell in dm.edge_bending_angles(V, F).values()])
    Wd = rng.normal(size=V.shape)
    h = 1e-6
    close(f"Theorem 41.2.9: sum l dtheta = 0 along a random motion (finite differences, {name})",
          np.dot(ells, (bend(V + h * Wd) - bend(V - h * Wd)) / (2 * h)), 0.0, 1e-7)
    g = ddg41.central_gradient(lambda Y: ddg41.half_sum_ltheta(Y, F), V)
    close(f"Theorem 41.2.11: grad (1/2) sum l theta = (K N) ({name})", g, ddg41.gauss_curvature_normal(V, F), 1e-7)
    close(f"Theorem 41.2.11: grad H_total = -(K N) ({name})", ddg41.central_gradient(lambda Y: ddg41.total_mean_curvature(Y, F), V),
          -ddg41.gauss_curvature_normal(V, F), 1e-7)
    close(f"Corollary 41.2.12: grad sum delta = 0 ({name})", ddg41.central_gradient(lambda Y: dm.angle_defect(Y, F).sum(), V), np.zeros_like(V), 1e-8)
V, F = ddg41.saddle_grid(9, 0.5)
close("Corollary 41.2.12: grad sum delta = 0 also with boundary (saddle grid)",
      ddg41.central_gradient(lambda Y: dm.angle_defect(Y, F).sum(), V), np.zeros_like(V), 1e-8)
check("Theorem 41.2.9: fails near the boundary (saddle grid)", np.abs(schlafli_local(V, F)).max() > 0.1)
# Example 41.2.10
T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]])
FT = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
if dm.signed_volume(T, FT) < 0:
    FT = FT[:, ::-1]
d = T[0] / np.linalg.norm(T[0])


def thetas(t):
    X_ = T.copy()
    X_[0] = X_[0] + t * d
    b = dm.edge_bending_angles(X_, FT)
    return {k: b[k][0] for k in b}


b0 = dm.edge_bending_angles(T, FT)
tp, tm = thetas(1e-6), thetas(-1e-6)
rates = {k: (tp[k] - tm[k]) / 2e-6 for k in b0}
apex = [rates[k] for k in rates if 0 in k]
base = [rates[k] for k in rates if 0 not in k]
close("Example 41.2.10: bending angle 1.9106 = pi - arccos(1/3)", list(b0.values())[0][0], np.pi - np.arccos(1 / 3), 1e-12)
close("Example 41.2.10: apex edges dtheta/dt = 0.13608, base edges -0.13608", np.r_[apex, base], np.r_[[0.13608] * 3, [-0.13608] * 3], 1e-5)
close("Example 41.2.10: dtheta_apex = sqrt(6)/18 (closed form)", apex[0], np.sqrt(6) / 18, 1e-8)
# Figure 41.2.2 numbers
V0, F0 = dm.icosphere(2)
gg = V0[:, 0] * V0[:, 1] + 0.6 * V0[:, 2] ** 2 - 0.3 * V0[:, 0]
Xt = lambda t: V0 * (1 + 0.35 * t * gg)[:, None]  # noqa: E731
dT = [(ddg41.half_sum_ltheta(Xt(t + 1e-5), F0) - ddg41.half_sum_ltheta(Xt(t - 1e-5), F0)) / 2e-5 for t in (0.0, 1.0)]
close("Figure 41.2.2: dT/dt from 0.87 to 1.60", dT, [0.874, 1.600], 1e-3)
# Exercise 41.2.3 (regular tetrahedron, edge a = 2 sqrt 2)
a = 2 * np.sqrt(2)
th0 = np.pi - np.arccos(1 / 3)
close("Exercise 41.2.3: |grad V| = a^2 / (4 sqrt 3)", np.linalg.norm(ddg41.vector_area(T, FT)[0]), a * a / (4 * np.sqrt(3)), 1e-12)
close("Exercise 41.2.3: |grad A| = sqrt 2 a", np.linalg.norm(dd.area_gradient(T, FT)[0]), np.sqrt(2) * a, 1e-12)
close("Exercise 41.2.3: |grad (1/2) sum l theta| = (sqrt 6 / 2) theta_0", np.linalg.norm(ddg41.gauss_curvature_normal(T, FT)[0]), np.sqrt(6) / 2 * th0, 1e-12)
check("Exercise 41.2.3: the three gradients are parallel to x_i",
      all(np.linalg.norm(np.cross(g_[0], T[0])) < 1e-12 for g_ in (ddg41.vector_area(T, FT), dd.area_gradient(T, FT), ddg41.gauss_curvature_normal(T, FT))))
close("Exercise 41.2.3: V = a^3 / (6 sqrt 2), A = sqrt 3 a^2", [dm.signed_volume(T, FT), dm.face_areas(T, FT).sum()], [a ** 3 / (6 * np.sqrt(2)), np.sqrt(3) * a * a], 1e-12)
summary()

"""Verification for Section 40.4 (Steiner's Formula and Mollified Polyhedra).

Run from the project root: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/verify/v-40-4-steiner-mollified.py
(the two grid volumes with h = 0.01 take about ten seconds; numpy threads are limited by check_all.sh)
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgsym  # noqa: E402
import ddg40  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "40-4-steiner-mollified.html"
pi = np.pi

# ---------------------------------------------------------------- Implementation 40.4 and Exercise 40.4.6
check("Implementation 40.4: at most 30 lines", dgcode.code_lines(PAGE, "code-40-4-1") <= 30)
ns, out = ddg40.run_printed(PAGE, "code-40-4-1")
check("Implementation 40.4: printed output matches the page", out.strip() == ddg40.printed_block(PAGE, "out-40-4-1").strip(), out)
ns2, out2 = ddg40.run_printed(PAGE, "code-ex-40-4-6", dict(ns))
check("Exercise 40.4.6: printed output matches the page", out2.strip() == ddg40.printed_block(PAGE, "out-ex-40-4-6").strip(), out2)
steiner = ns["steiner"]
for name, (V, F) in [("tetrahedron", ddg40.tetrahedron()), ("octahedron", ddg40.octahedron()),
                     ("perturbed icosahedron", ddg40.perturbed_icosahedron()), ("torus_mesh(2, 0.8, 8, 12)", dm.torus_mesh(2, 0.8, 8, 12))]:
    close(f"Implementation 40.4: steiner = ddg40.steiner_coefficients on {name}", np.array(steiner(V, F)),
          np.array(ddg40.steiner_coefficients(V, F)), 1e-12)

# ---------------------------------------------------------------- Proposition 40.4.1: (x + eps N)_u x (x + eps N)_v = (1 - 2 eps H + eps^2 K) x_u x x_v
rng = np.random.default_rng(40)
e = sp.Symbol("varepsilon", real=True)
for key in ("sphere", "cylinder", "torus", "saddle"):
    ex = dgsym.EXAMPLES[key]
    X = sp.Matrix(ex["expr"])
    u, v = ex["coords"]
    pos = ex.get("positive", ())
    Nn = dgsym.unit_normal(X, u, v, pos)
    K, H = dgsym.K_H(X, u, v, pos)
    Xe = X + e * Nn
    lhs = Xe.diff(u).cross(Xe.diff(v))
    rhs = (1 - 2 * e * H + e ** 2 * K) * X.diff(u).cross(X.diff(v))
    syms = [u, v, e] + list(ex.get("params", ()))
    f = sp.lambdify(syms, list(lhs - rhs), modules="numpy")
    dom = ex["domain"]
    err = 0.0
    for _ in range(40):
        vals = [rng.uniform(*[float(t) for t in dom[u]]), rng.uniform(*[float(t) for t in dom[v]]), rng.uniform(-0.3, 0.3)]
        vals += [rng.uniform(*[float(t) for t in dom[p]]) for p in ex.get("params", ())]
        if key == "sphere":
            vals[0] = rng.uniform(0.2, pi - 0.2)
        err = max(err, float(np.max(np.abs(np.array(f(*vals), float)))))
    check(f"Proposition 40.4.1: {key}: (x_eps)_u x (x_eps)_v = (1 - 2 eps H + eps^2 K)(x_u x x_v) at 40 random points "
          f"(max error {err:.1e})", err < 1e-10)
check("Proposition 40.4.1: 1 - 2 eps H + eps^2 K = (1 - eps k1)(1 - eps k2)",
      sp.expand((1 - e * sp.Symbol("k1")) * (1 - e * sp.Symbol("k2")))
      == sp.expand(1 - 2 * e * (sp.Symbol("k1") + sp.Symbol("k2")) / 2 + e ** 2 * sp.Symbol("k1") * sp.Symbol("k2")))

# ---------------------------------------------------------------- Example 40.4.3: sphere and torus (outward normals)
r, R, t = sp.symbols("r R t", positive=True)
th, ph, uu, vv = sp.symbols("theta phi u v", real=True)
Asph = sp.integrate(sp.integrate((1 - 2 * e * (-1 / r) + e ** 2 / r ** 2) * r ** 2 * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
check("Example 40.4.3(a): sphere area(x_eps) = 4 pi (r + eps)^2", sp.simplify(Asph - 4 * sp.pi * (r + e) ** 2) == 0)
Vsph = sp.Rational(4, 3) * sp.pi * r ** 3 + sp.integrate(Asph.subs(e, t), (t, 0, e))
check("Example 40.4.3(a): sphere volume (40.4.2) = (4/3) pi (r + eps)^3", sp.simplify(Vsph - sp.Rational(4, 3) * sp.pi * (r + e) ** 3) == 0)
Hout = -(R + 2 * r * sp.cos(uu)) / (2 * r * (R + r * sp.cos(uu)))
Ktor = sp.cos(uu) / (r * (R + r * sp.cos(uu)))
dA = r * (R + r * sp.cos(uu))
check("Example 40.4.3(b): torus, outward normal: int H dA = -2 pi^2 R",
      sp.simplify(sp.integrate(sp.simplify(Hout * dA), (uu, 0, 2 * sp.pi), (vv, 0, 2 * sp.pi)) + 2 * sp.pi ** 2 * R) == 0)
check("Example 40.4.3(b): torus int K dA = 0", sp.simplify(sp.integrate(sp.simplify(Ktor * dA), (uu, 0, 2 * sp.pi), (vv, 0, 2 * sp.pi))) == 0)
Ator = sp.integrate(sp.expand(sp.simplify((1 - 2 * e * Hout + e ** 2 * Ktor) * dA)), (uu, 0, 2 * sp.pi), (vv, 0, 2 * sp.pi))
check("Example 40.4.3(b): torus area(x_eps) = 4 pi^2 R (r + eps)", sp.simplify(Ator - 4 * sp.pi ** 2 * R * (r + e)) == 0)
Vtor = 2 * sp.pi ** 2 * R * r ** 2 + sp.integrate(Ator.subs(e, t), (t, 0, e))
check("Example 40.4.3(b): torus volume = 2 pi^2 R (r + eps)^2", sp.simplify(Vtor - 2 * sp.pi ** 2 * R * (r + e) ** 2) == 0)
# the inward torus normal of dgsym gives +2 pi^2 R (Exercise 39.3.5): the sign of H flips with N
ex = dgsym.EXAMPLES["torus"]
Kt, Ht = dgsym.K_H(sp.Matrix(ex["expr"]), *ex["coords"], ex["positive"])
sub = {ex["coords"][0]: uu, ex["params"][0]: R, ex["params"][1]: r}
check("Example 40.4.3(b): dgsym torus (inward normal) has H = -H_out and the same K",
      sp.simplify(Ht.subs(sub) + Hout) == 0 and sp.simplify(Kt.subs(sub) - Ktor) == 0)
# plane curves: convex region pushed out by eps: perimeter L + 2 pi eps, area A + L eps + pi eps^2 (disk of radius a)
a = sp.Symbol("a", positive=True)
check("Plane remark: disk of radius a: pi (a + eps)^2 = A + L eps + pi eps^2", sp.expand(sp.pi * (a + e) ** 2 - (sp.pi * a ** 2 + 2 * sp.pi * a * e + sp.pi * e ** 2)) == 0)

# ---------------------------------------------------------------- Definition 40.4.4: translation; Example 40.4.5; Non-example 40.4.6
def polygon_A_L(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y), np.sum(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1))


Q = np.array([[0, 0], [2, 0], [2.4, 1.2], [0.6, 1.6]], float)
Lp = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
AQ, LQ = polygon_A_L(Q)
close("Example 40.4.5(b), Figure 40.4.1(a): quadrilateral A = 2.76", AQ, 2.76, 1e-12)
close("Example 40.4.5(b), Figure 40.4.1(a): quadrilateral L = 6.818", LQ, 6.818, 5e-4)
fQ = AQ + LQ * 0.3 + pi * 0.09
gQ = ddg40.polygon_minkowski_area_grid(Q, 0.3, 0.002)
close("Example 40.4.5(b): A + L eps + pi eps^2 = 5.088 at eps = 0.3", fQ, 5.088, 5e-4)
check(f"Example 40.4.5(b): grid count (h = 0.002) {gQ:.5f} within 1e-3 of the formula", abs(gQ - fQ) < 1e-3)
gQt = ddg40.polygon_minkowski_area_grid(Q + np.array([0.37, -1.21]), 0.3, 0.002)
check(f"Definition 40.4.4: translating P does not change area(P + eps B) (grid {gQt:.5f} vs {gQ:.5f})", abs(gQt - gQ) < 1e-3)
# stadium: segment + disk = 2 eps l + pi eps^2 (thin rectangle as a polygon of width 1e-9 is avoided: Monte Carlo)
ell, eps = 1.7, 0.4
Z = rng.uniform([-eps, -eps], [ell + eps, eps], size=(400000, 2))
dseg = np.linalg.norm(Z - np.c_[np.clip(Z[:, 0], 0, ell), np.zeros(len(Z))], axis=1)
mc = np.mean(dseg <= eps) * (ell + 2 * eps) * 2 * eps
check(f"Example 40.4.5(a): stadium area 2 eps l + pi eps^2 = {2 * eps * ell + pi * eps ** 2:.4f} (Monte Carlo {mc:.4f})",
      abs(mc - (2 * eps * ell + pi * eps ** 2)) < 0.01)
check("Example 40.4.5(a): capsule volume pi eps^2 l + (4/3) pi eps^3 (cylinder + two half balls)",
      sp.simplify(sp.pi * e ** 2 * sp.Symbol("l") + 2 * sp.Rational(2, 3) * sp.pi * e ** 3 - (sp.pi * e ** 2 * sp.Symbol("l") + sp.Rational(4, 3) * sp.pi * e ** 3)) == 0)
AL, LL = polygon_A_L(Lp)
check("Non-example 40.4.6: L-shape A = 3, L = 8", abs(AL - 3) < 1e-12 and abs(LL - 8) < 1e-12)
for eps in (0.2, 0.3):
    true = 3 + 8 * eps + (5 * pi / 4 - 1) * eps ** 2
    g = ddg40.polygon_minkowski_area_grid(Lp, eps, 0.002)
    check(f"Non-example 40.4.6: eps = {eps}: grid {g:.5f} = 3 + 8 eps + (5 pi/4 - 1) eps^2 = {true:.5f} (within 1e-3)", abs(g - true) < 1e-3)
    check(f"Non-example 40.4.6: eps = {eps}: grid differs from 3 + 8 eps + pi eps^2 = {3 + 8 * eps + pi * eps ** 2:.5f}",
          abs(g - (3 + 8 * eps + pi * eps ** 2)) > 0.005)
close("Non-example 40.4.6: eps = 0.3 values 5.663 and 5.683", np.array([3 + 2.4 + (5 * pi / 4 - 1) * 0.09, 3 + 2.4 + pi * 0.09]),
      np.array([5.663, 5.683]), 5e-4)


# ---------------------------------------------------------------- Remark 40.4.16: signed offset curve of a plane polygon
def signed_offset_area(P, eps, n=4000):
    """Shoelace area of the closed curve: offset edges joined by arcs of the signed turning angle at every vertex."""
    m = len(P)
    pts = []
    for k in range(m):
        a, b, c = P[k - 1], P[k], P[(k + 1) % m]
        t0, t1 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        n0 = np.array([t0[1], -t0[0]])
        tau = np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1)
        phi0 = np.arctan2(n0[1], n0[0])
        for s in np.linspace(0, tau, n + 1):
            pts.append(b + eps * np.array([np.cos(phi0 + s), np.sin(phi0 + s)]))
    pts = np.array(pts)
    return 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]), pts


def winding(pts, q):
    d = pts - q
    ang = np.arctan2(d[:, 1], d[:, 0])
    return int(round(np.sum(np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))) / (2 * pi)))


for name, P in (("L-shape", Lp), ("quadrilateral", Q)):
    A0, L0 = polygon_A_L(P)
    for eps in (0.2, 0.3, 0.7):
        s, _ = signed_offset_area(P, eps)
        check(f"Remark 40.4.16: {name}, eps = {eps}: signed area of the signed offset curve {s:.6f} = A + L eps + pi eps^2",
              abs(s - (A0 + L0 * eps + pi * eps ** 2)) < 1e-5)

eps = 0.3
_, pts = signed_offset_area(Lp, eps)
wn = [winding(pts, np.array(q)) for q in ((1 + 0.9 * eps, 1 + 0.9 * eps), (1 + 0.3 * eps, 1 + 0.3 * eps), (0.5, 0.5),
                                         (2 + 0.5 * eps, 0.5), (3.0, 3.0))]
check(f"Remark 40.4.16: L-shape, eps = 0.3: winding numbers {wn} = [2 (square minus quarter disk), 1 (quarter disk), 1 (in P), "
      f"1 (band), 0 (outside)]", wn == [2, 1, 1, 1, 0])
close("Remark 40.4.16: signed area - true area = (1 - pi/4) eps^2 = area of the winding-2 region",
      (3 + 8 * eps + pi * eps ** 2) - (3 + 8 * eps + (5 * pi / 4 - 1) * eps ** 2), (1 - pi / 4) * eps ** 2, 1e-12)

# ---------------------------------------------------------------- Lemma 40.4.8 (b): the nearest-point criterion (Monte Carlo on a convex polygon)
Z = rng.uniform(-1, 3, size=(2000, 2))
ok = True
for y in Z:
    best = None
    for k in range(len(Q)):
        a_, b_ = Q[k], Q[(k + 1) % len(Q)]
        tt = np.clip((y - a_) @ (b_ - a_) / ((b_ - a_) @ (b_ - a_)), 0, 1)
        z = a_ + tt * (b_ - a_)
        if best is None or np.linalg.norm(y - z) < np.linalg.norm(y - best):
            best = z
    if MPath(Q).contains_point(y):
        continue
    ok &= bool(np.all((Q - best) @ (y - best) <= 1e-9))
check("Lemma 40.4.8(b): for 2000 random points outside a convex quadrilateral, y - pi(y) lies in the normal cone at pi(y)", ok)

# ---------------------------------------------------------------- Lemma 40.4.9 on convex polyhedra
convex = [("tetrahedron", ddg40.tetrahedron()), ("octahedron", ddg40.octahedron()), ("unit cube", ddg40.unit_cube()),
          ("perturbed icosahedron", ddg40.perturbed_icosahedron()), ("icosphere(1)", dm.icosphere(1))]
for name, (V, F) in convex:
    N = dm.face_normals(V, F)
    side = np.einsum("fk,fnk->fn", N, V[None, :, :] - V[F[:, 0]][:, None, :])
    check(f"Lemma 40.4.9(a): {name}: every vertex lies in every H_f", np.all(side <= 1e-12))
    th = dm.edge_bending_angles(V, F)
    check(f"Lemma 40.4.9(c): {name}: every bending angle theta_ij >= 0", min(x for x, _ in th.values()) >= -1e-12)
    ef = dm.edge_faces(F)
    okc, okarc = True, True
    for (i, j), (theta, _) in th.items():
        f, g = ef[(i, j)]
        okarc &= abs(np.arccos(np.clip(N[f] @ N[g], -1, 1)) - abs(theta)) < 1e-9
        p = 0.4 * V[i] + 0.6 * V[j]
        for _ in range(3):
            aa, bb = rng.uniform(0, 1, 2)
            uvec = aa * N[f] + bb * N[g]
            okc &= np.all((V - p) @ uvec <= 1e-12)
            if theta > 1e-9:
                okc &= not np.all((V - p) @ (-aa * N[f] + bb * N[g]) <= 1e-12) or aa < 1e-9
    check(f"Lemma 40.4.9(c): {name}: arc length between N_f and N_g = |theta_ij|", okarc)
    check(f"Lemma 40.4.9(c): {name}: a N_f + b N_g (a, b >= 0) is a normal at an edge point; a < 0 is not", okc)

# ---------------------------------------------------------------- Theorem 40.4.10 and Corollary 40.4.11
for name, (V, F) in convex:
    _, _, _, G = ddg40.steiner_coefficients(V, F)
    close(f"Theorem 40.4.10: {name}: sum delta = 4 pi", G, 4 * pi, 1e-10)
Vp, Fp = ddg40.perturbed_icosahedron()
check("Corollary 40.4.11: perturbed icosahedron: every vertex is a convex vertex", all(ddg40.is_convex_vertex(Vp, Fp, i) for i in range(len(Vp))))
U = rng.normal(size=(400000, 3))
U /= np.linalg.norm(U, axis=1, keepdims=True)
arg = np.argmax(U @ Vp.T, axis=1)
frac = np.bincount(arg, minlength=len(Vp)) / len(U)
dlt = dm.angle_defect(Vp, Fp)
sig = np.sqrt(dlt / (4 * pi) * (1 - dlt / (4 * pi)) / len(U))
zmax = np.max(np.abs(frac - dlt / (4 * pi)) / sig)
check(f"Corollary 40.4.11: area(Sigma_i) / 4 pi (Monte Carlo, 400000 directions) = delta_i / 4 pi at all 12 vertices "
      f"(max {zmax:.2f} sigma)", zmax < 4.5)
# Sigma_i is contained in P_i for every direction (u maximizes <u, .> at x_i => <u, x_j - x_i> <= 0 for the link)
okin = True
for i in range(len(Vp)):
    link = ddg40.link_cycle(Fp, i)
    sel = U[arg == i][:2000]
    okin &= bool(np.all(sel @ (Vp[link] - Vp[i]).T <= 1e-12))
check("Lemma 40.4.9(d): directions maximized at x_i satisfy <u, x_j - x_i> <= 0 for the link (Sigma_i in P_i)", okin)

# volumes of P + eps B on a grid (h = 0.01)
for name, (V, F), eps in (("unit cube", ddg40.unit_cube(), 0.3), ("tetrahedron", ddg40.tetrahedron(), 0.3)):
    V0, A0, M, G = ddg40.steiner_coefficients(V, F)
    poly = V0 + A0 * eps + M * eps ** 2 + G * eps ** 3 / 3
    g = ddg40.mollified_volume_grid(V, F, eps, 0.01, rng=np.random.default_rng(1))
    check(f"Theorem 40.4.10, Example 40.4.12: {name}, eps = {eps}: stratified grid volume (h = 0.01) {g:.5f} vs Steiner "
          f"polynomial {poly:.5f} (rel. error {abs(g / poly - 1):.1e} < 1e-4)", abs(g / poly - 1) < 1e-4)
# areas of the boundary pieces (mesh of the pieces, refined)
for name, (V, F) in (("tetrahedron", ddg40.tetrahedron()), ("unit cube", ddg40.unit_cube()), ("perturbed icosahedron", (Vp, Fp))):
    V0, A0, M, G = ddg40.steiner_coefficients(V, F)
    eps = 0.35
    tot = {"face": 0.0, "edge": 0.0, "vertex": 0.0}
    for P, Fq, k in ddg40.mollified_pieces(V, F, eps, n_arc=64, n_sph=40):
        tot[k] += dm.face_areas(P, Fq).sum()
    check(f"Theorem 40.4.10: {name}: piece areas face {tot['face']:.4f} = A, edge {tot['edge']:.4f} = eps sum l theta "
          f"{2 * eps * M:.4f}, vertex {tot['vertex']:.4f} = eps^2 sum delta {eps ** 2 * G:.4f} (1e-3 rel.)",
          abs(tot["face"] - A0) < 1e-9 and abs(tot["edge"] / (2 * eps * M) - 1) < 1e-3 and abs(tot["vertex"] / (eps ** 2 * G) - 1) < 1e-3)

# ---------------------------------------------------------------- Example 40.4.12 and Exercises 40.4.1, 40.4.2
Vc, Fc = ddg40.unit_cube()
V0, A0, M, G = ddg40.steiner_coefficients(Vc, Fc)
close("Example 40.4.12(a): cube V = 1, A = 6, (1/2) sum l theta = 3 pi, sum delta = 4 pi", np.array([V0, A0, M, G]),
      np.array([1, 6, 3 * pi, 4 * pi]), 1e-12)
for diag in ("avoid", "max", "alt"):
    check(f"Example 40.4.12(a): cube with diagonals '{diag}': same Steiner coefficients",
          np.allclose(ddg40.steiner_coefficients(*ddg40.unit_cube(diag)), [1, 6, 3 * pi, 4 * pi], atol=1e-12))
check("Example 40.4.12(a): theta = pi/2 on the 12 cube edges, 0 on the 6 diagonals",
      sorted(round(x, 12) for x, _ in dm.edge_bending_angles(Vc, Fc).values()) == [0.0] * 6 + [round(pi / 2, 12)] * 12)
check("Example 40.4.12(a): 1 + 6 eps + 3 pi eps^2 + (4/3) pi eps^3 = slabs + quarter cylinders + eighth balls",
      sp.simplify(1 + 6 * e + 12 * sp.pi / 4 * e ** 2 + 8 * sp.pi / 6 * e ** 3 - (1 + 6 * e + 3 * sp.pi * e ** 2 + sp.Rational(4, 3) * sp.pi * e ** 3)) == 0)
check("Exercise 40.4.1: area 6 + 6 pi eps + 4 pi eps^2 is d/d eps of the volume",
      sp.simplify(sp.diff(1 + 6 * e + 3 * sp.pi * e ** 2 + sp.Rational(4, 3) * sp.pi * e ** 3, e) - (6 + 6 * sp.pi * e + 4 * sp.pi * e ** 2)) == 0)
V2 = ddg40.steiner_coefficients(2 * Vc, Fc)
check("Example 40.4.12(a): side-2 cube: A = 24, sum l theta = 12 pi, sum delta = 4 pi, so area 24 + 12 pi eps + 4 pi eps^2 "
      "(Example 31.2.7)", abs(V2[1] - 24) < 1e-12 and abs(2 * V2[2] - 12 * pi) < 1e-12 and abs(V2[3] - 4 * pi) < 1e-12)
Vt, Ft = ddg40.tetrahedron()
V0, A0, M, G = ddg40.steiner_coefficients(Vt, Ft)
close("Example 40.4.12(b): tetrahedron V = 8/3, A = 8 sqrt 3 = 13.856", np.array([V0, A0]), np.array([8 / 3, 8 * np.sqrt(3)]), 1e-12)
close("Example 40.4.12(b): theta = pi - arccos(1/3) = 1.9106", min(x for x, _ in dm.edge_bending_angles(Vt, Ft).values()), 1.9106, 5e-5)
close("Example 40.4.12(b): (1/2) sum l theta = 6 sqrt 2 * 1.9106 = 16.212", M, 6 * np.sqrt(2) * (pi - np.arccos(1 / 3)), 1e-12)
close("Example 40.4.12(b): (1/2) sum l theta = 16.212", M, 16.212, 5e-4)
close("Example 40.4.12(b): vol(P_0.3) = 8.3958", V0 + A0 * 0.3 + M * 0.09 + G * 0.009, 8.3958, 5e-5)
Vo, Fo = ddg40.octahedron()
V0, A0, M, G = ddg40.steiner_coefficients(Vo, Fo)
close("Exercise 40.4.2: octahedron V = 4/3, A = 4 sqrt 3 = 6.928", np.array([V0, A0]), np.array([4 / 3, 4 * np.sqrt(3)]), 1e-12)
close("Exercise 40.4.2: theta = arccos(1/3) = 1.2310, (1/2) sum l theta = 10.445", np.array([min(x for x, _ in dm.edge_bending_angles(Vo, Fo).values()), M]),
      np.array([1.2310, 10.445]), 5e-4)
check("Exercise 40.4.2: octahedron sum delta = 6 * 2 pi / 3", abs(G - 4 * pi) < 1e-12 and np.allclose(dm.angle_defect(Vo, Fo), 2 * pi / 3))

# ---------------------------------------------------------------- Figure 40.4.2 numbers
pieces = ddg40.mollified_pieces(Vt, Ft, 0.35, n_arc=16, n_sph=10)
tot = {"face": 0.0, "edge": 0.0, "vertex": 0.0}
for P, Fq, k in pieces:
    tot[k] += dm.face_areas(P, Fq).sum()
V0, A0, M, G = ddg40.steiner_coefficients(Vt, Ft)
close("Figure 40.4.2(a): drawn piece areas 13.856, 11.342, 1.530", np.array([tot["face"], tot["edge"], tot["vertex"]]),
      np.array([13.856, 11.342, 1.530]), 5e-4)
close("Figure 40.4.2(a): exact eps sum l theta = 11.349, eps^2 4 pi = 1.539", np.array([0.7 * M, 0.35 ** 2 * G]), np.array([11.349, 1.539]), 5e-4)
terms = np.array([V0, 0.6 * A0, 0.36 * M, 0.072 * G])
close("Figure 40.4.2(b): eps = 0.6 terms 2.67, 8.31, 5.84, 0.90", terms, np.array([2.67, 8.31, 5.84, 0.90]), 5e-3)
close("Figure 40.4.2(b): eps = 0.6 sum 17.72", terms.sum(), 17.72, 5e-3)
g06 = ddg40.mollified_volume_grid(Vt, Ft, 0.6, 0.02)
close("Figure 40.4.2(b): grid (h = 0.02) at eps = 0.6 is 17.69", g06, 17.69, 5e-3)
check("Figure 40.4.2(b): within 0.2%", abs(g06 / terms.sum() - 1) < 2e-3)

# ---------------------------------------------------------------- Proposition 40.4.13: curvature of the mollified surface
Ssum, Gs, s = sp.symbols("S G s", positive=True)        # S = sum l theta, G = sum area(Sigma_i) = 4 pi
Aeps = sp.Symbol("A") + e * Ssum + e ** 2 * Gs
intH = -sp.Rational(1, 2) / e * (e * Ssum) - 1 / e * (e ** 2 * Gs)
intK = 1 / e ** 2 * (e ** 2 * Gs)
check("Proposition 40.4.13: int H = -(1/2) sum l theta - eps * sum area(Sigma_i), int K = sum area(Sigma_i)",
      sp.simplify(intH + Ssum / 2 + e * Gs) == 0 and sp.simplify(intK - Gs) == 0)
check("Proposition 40.4.13: area(dP_eps) - 2 s int H + s^2 int K = area(dP_{eps+s}) (Proposition 40.4.1 piece by piece)",
      sp.expand(Aeps - 2 * s * intH + s ** 2 * intK - Aeps.subs(e, e + s)) == 0)
ex = dgsym.EXAMPLES["cylinder"]
Kc, Hc = dgsym.K_H(ex["expr"], *ex["coords"])
check("Proposition 40.4.13: cylinder piece (outward) K = 0, H = -1/(2 eps); sphere piece K = 1/eps^2, H = -1/eps",
      sp.simplify(Hc + 1 / (2 * ex["params"][0])) == 0 and sp.simplify(Kc) == 0
      and dgsym.EXAMPLES["sphere"]["expected"]["H"] == -1 / dgsym.EXAMPLES["sphere"]["params"][0])

# ---------------------------------------------------------------- Definition 40.4.14 and Example 40.4.15
Hij = {k: -0.5 * l * x for k, (x, l) in dm.edge_bending_angles(Vc, Fc).items()}
check("Example 40.4.15(a): cube H_ij = -pi/4 on the 12 edges, 0 on the diagonals, sum -3 pi",
      sorted(round(x, 12) for x in Hij.values()) == [round(-pi / 4, 12)] * 12 + [0.0] * 6 and abs(sum(Hij.values()) + 3 * pi) < 1e-12)
Vs, Fs = dm.icosphere(2)
Hd = dm.mean_curvature_dihedral(Vs, Fs, mass=np.ones(len(Vs)))
Hsum = sum(-0.5 * l * x for x, l in dm.edge_bending_angles(Vs, Fs).values())
close("Definition 40.4.14: H_i = (1/2) sum_j H_ij (dgmesh.mean_curvature_dihedral with A_i = 1), and sum_i H_i = sum_ij H_ij",
      Hd.sum(), Hsum, 1e-10)
Vr, Fr = Vs, Fs[:, ::-1]
check("Definition 40.4.14: reversing the orientation flips every theta_ij and H_ij",
      np.allclose(dm.mean_curvature_dihedral(Vr, Fr, mass=np.ones(len(Vr))), -Hd, atol=1e-12))
V4, F4 = dm.icosphere(4)
Hb = dm.mean_curvature_dihedral(V4, F4, mass=dm.mass_barycentric(V4, F4))
check(f"Example 40.4.15(b): barycentric A_i, level 4: H_i in [{Hb.min():.3f}, {Hb.max():.3f}] = [-1.146, -0.998]",
      abs(Hb.min() + 1.146) < 5e-4 and abs(Hb.max() + 0.998) < 5e-4)
Vtor, Ftor = dm.torus_mesh(2, 0.8, 24, 48)
thv = np.array([x for x, _ in dm.edge_bending_angles(Vtor, Ftor).values()])
check("Example 40.4.15(c): torus_mesh is not convex (some theta < 0)", thv.min() < -0.01 and thv.max() > 0.01)
errs = []
for nu, nv in ((12, 24), (24, 48), (48, 96), (96, 192)):
    _, A_, M_, _ = ddg40.steiner_coefficients(*dm.torus_mesh(2, 0.8, nu, nv))
    errs.append((-M_ + 2 * pi ** 2 * 2, 4 * pi ** 2 * 1.6 - A_))
errs = np.array(errs)
ratios = errs[1:] / errs[:-1]
check(f"Example 40.4.15(c), Exercise 40.4.6: errors shrink by about 1/4 per refinement (ratios {np.round(ratios.ravel(), 3)})",
      np.all(np.abs(ratios - 0.25) < 0.02))
close("Exercise 40.4.6: errors 0.113, 0.028, 0.0070, 0.0017 and 1.163, 0.293, 0.073, 0.018", errs.T.ravel(),
      np.array([0.113, 0.028, 0.0070, 0.0017, 1.163, 0.293, 0.073, 0.018]), 6e-4)

# ---------------------------------------------------------------- Exercise 40.4.4: face split and edge split
V0s = np.array(ddg40.steiner_coefficients(Vt, Ft))
m = Vt[Ft[0]].mean(0)
Vf = np.vstack([Vt, m])
a_, b_, c_ = Ft[0]
Ff = np.vstack([Ft[1:], [[a_, b_, 4], [b_, c_, 4], [c_, a_, 4]]])
check("Exercise 40.4.4(a): face split keeps V, A, sum l theta, sum delta", np.allclose(ddg40.steiner_coefficients(Vf, Ff), V0s, atol=1e-12))
check("Exercise 40.4.4(a): new vertex delta = 0, new edges theta = 0",
      abs(dm.angle_defect(Vf, Ff)[4]) < 1e-12 and all(abs(x) < 1e-12 for (i, j), (x, _) in dm.edge_bending_angles(Vf, Ff).items() if 4 in (i, j)))
i0, j0 = Ft[0][0], Ft[0][1]
fs = [f for f in Ft if i0 in f and j0 in f]
Fe = [f for f in Ft.tolist() if not (i0 in f and j0 in f)]
for f in fs:
    f = list(f)
    k = [x for x in f if x not in (i0, j0)][0]
    rot = f[f.index(k):] + f[:f.index(k)]          # (k, p, q) with p -> q along the orientation
    Fe += [[k, rot[1], 4], [k, 4, rot[2]]]
Ve = np.vstack([Vt, 0.5 * (Vt[i0] + Vt[j0])])
Fe = np.array(Fe)
check("Exercise 40.4.4(b): edge split keeps V, A, sum l theta, sum delta", np.allclose(ddg40.steiner_coefficients(Ve, Fe), V0s, atol=1e-12))
the = dm.edge_bending_angles(Ve, Fe)
tht = pi - np.arccos(1 / 3)
check("Exercise 40.4.4(b): the two halves keep theta, the two new edges have theta = 0, new vertex delta = 0",
      abs(the[tuple(sorted((i0, 4)))][0] - tht) < 1e-12 and abs(the[tuple(sorted((j0, 4)))][0] - tht) < 1e-12
      and sum(abs(x) < 1e-12 for (i, j), (x, _) in the.items() if 4 in (i, j)) == 2 and abs(dm.angle_defect(Ve, Fe)[4]) < 1e-12)

# ---------------------------------------------------------------- Exercise 40.4.3: convex polygons, random
for seed in range(3):
    rr = np.random.default_rng(100 + seed)
    angs = np.sort(rr.uniform(0, 2 * pi, 7))
    P = np.c_[np.cos(angs), np.sin(angs)] * rr.uniform(0.8, 1.2)
    A0, L0 = polygon_A_L(P)
    g = ddg40.polygon_minkowski_area_grid(P, 0.25, 0.002)
    check(f"Exercise 40.4.3: random convex 7-gon (seed {seed}): grid {g:.4f} = A + L eps + pi eps^2 = {A0 + L0 * 0.25 + pi * 0.0625:.4f}",
          abs(g - (A0 + L0 * 0.25 + pi * 0.0625)) < 1e-3)

# ---------------------------------------------------------------- Smooth vs. Discrete box: (L x)_i edge share l sin(theta/2) vs (1/2) l theta
_, Lx, _ = ddg40.curvature_normals(Vc, Fc)
ef = ddg40.mean_normal_edge_form(Vc, Fc)
close("Smooth vs. Discrete box: the edge form (40.3.5) = (L x)_i on the cube", ef, Lx, 1e-12)
i3 = [i for i in range(len(Vc)) if len(ddg40.faces_around(Fc, i)) == 3][0]
Hi3 = 0.5 * sum(Hij[k] for k in Hij if i3 in k)
check(f"Smooth vs. Discrete box: at a 3-valent cube corner (1/2)|(L x)_i| = sqrt(3)/2 = {np.linalg.norm(Lx[i3]) / 2:.4f} "
      f"but |H_i| = 3 pi / 8 = {abs(Hi3):.4f}: the two vertex values differ",
      abs(np.linalg.norm(Lx[i3]) - np.sqrt(3)) < 1e-12 and abs(Hi3 + 3 * pi / 8) < 1e-12)

summary()

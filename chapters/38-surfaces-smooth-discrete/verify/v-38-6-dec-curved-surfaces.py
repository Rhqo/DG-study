"""Verification for Section 38.6 (DEC on Curved Surfaces and Shape Recovery).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-6-dec-curved-surfaces.py
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "38-6-dec-curved-surfaces.html"
rng = np.random.default_rng(386)


def circumcenter(a, b, c):
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    return a + (ac @ ac * np.cross(n, ab) + ab @ ab * np.cross(ac, n)) / (2 * n @ n)


def nullity(V, F, tol=1e-9):
    A = ddg38.normal_constraint_matrix(F, dm.face_normals(V, F), len(V))
    return A.shape[1] - np.linalg.matrix_rank(A, tol=tol * np.linalg.norm(A, 2))


def flat_edges(V, F):
    N = dm.face_normals(V, F)
    out = 0
    for (i, j) in ddg38.edge_lengths(V, F):
        fs = [f for f in range(len(F)) if i in F[f] and j in F[f]]
        out += len(fs) == 2 and np.allclose(N[fs[0]], N[fs[1]], atol=1e-12)
    return out


# ---------------------------------------------------------------- Implementations and Exercise 38.6.6
for blk in ("code-38-6-1", "code-38-6-2", "code-ex-38-6-6"):
    check(f"{blk}: at most 30 lines", dgcode.code_lines(PAGE, blk) <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-6-1")
check("Implementation 38.6.1: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-38-6-1").strip(), out)
ns, out = ddg38.run_printed(PAGE, "code-38-6-2", ns)
check("Implementation 38.6.2: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-38-6-2").strip(), out)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-6-6", ns)
check("Exercise 38.6.6: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-ex-38-6-6").strip(), out)
lap, lengths, sfn = ns["laplacian_from_lengths"], ns["lengths"], ns["shape_from_normals"]

# ---------------------------------------------------------------- Proposition 38.6.1
for _ in range(5):
    P = rng.normal(size=(3, 3))
    a, b, c = (np.linalg.norm(P[(k + 1) % 3] - P[(k + 2) % 3]) for k in range(3))
    A = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    close("Prop 38.6.1(a): Heron's formula", 0.25 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)), A, 1e-12)
    u, w = P[1] - P[0], P[2] - P[0]
    alpha = np.arctan2(np.linalg.norm(np.cross(u, w)), u @ w)              # angle at P[0], opposite side a
    close("Prop 38.6.1(a): cot alpha = (b^2 + c^2 - a^2) / (4A)", 1 / np.tan(alpha), (b * b + c * c - a * a) / (4 * A), 1e-10)
Vj = dm.icosphere(2)[0] * (1 + 0.1 * rng.uniform(-1, 1, (162, 1)))
Fj = dm.icosphere(2)[1]
Vt, Ft = dm.torus_mesh(2.0, 0.8, 12, 8)
Vt2 = Vt + 0.05 * rng.normal(size=Vt.shape)
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("jittered icosphere", (Vj, Fj)), ("jittered torus", (Vt2, Ft))):
    L, A0 = lap(F, lengths(V, F), len(V))
    close(f"Prop 38.6.1(b) on {name}: L from lengths = dgmesh.cotan_laplacian", L, dm.cotan_laplacian(V, F), 1e-12)
    close(f"Prop 38.6.1(b) on {name}: circumcentric star0 from lengths = dgdec", A0, dd.hodge0(V, F, "circumcentric"), 1e-12)
    s0, s1, s2, L2 = ddg38.stars_from_lengths(F, ddg38.edge_lengths(V, F), len(V))
    E, _ = dd.edge_list(F)
    close(f"Prop 38.6.1(b) on {name}: star1 from lengths = dgdec.hodge1", np.array([s1[(int(i), int(j))] for i, j in E]),
          dd.hodge1(V, F), 1e-12)
    close(f"Prop 38.6.1(b) on {name}: barycentric star0, star2 from lengths", np.concatenate([s0, s2]),
          np.concatenate([dd.hodge0(V, F, "barycentric"), dd.hodge2(V, F)]), 1e-12)
    close(f"Prop 38.6.1(b) on {name}: L = -d0^T star1 d0", L, -dd.laplacian_psd(V, F), 1e-12)
# isometric copies: rigid motion + reflection of a mesh have the same operators
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
close("Cor 38.6.3: an isometric copy (orthogonal map) has the same L", dm.cotan_laplacian(Vj @ Q.T + 1.0, Fj),
      dm.cotan_laplacian(Vj, Fj), 1e-11)

# ---------------------------------------------------------------- Example 38.6.4, Non-example 38.6.5, Figure 38.6.1
L0 = dm.cotan_laplacian(*ddg38.hinge(0.0))
hf, hg = 0.5 * 0.4875, 0.5 * 2 / 3
for psi in np.linspace(0, 0.95 * np.pi, 8):
    V, F = ddg38.hinge(psi)
    close(f"Example 38.6.4: hinge psi = {psi:.2f}: L unchanged", dm.cotan_laplacian(V, F), L0, 1e-12)
    cf, cg = circumcenter(*V[F[0]]), circumcenter(*V[F[1]])
    m = (V[0] + V[1]) / 2
    check(f"Section 38.6: hinge psi = {psi:.2f}: both dual pieces perpendicular to the edge",
          abs((cf - m) @ (V[1] - V[0])) < 1e-12 and abs((cg - m) @ (V[1] - V[0])) < 1e-12)
    close(f"Non-example 38.6.5: |c_f - c_g|^2 = h_f^2 + h_g^2 + 2 h_f h_g cos psi (psi = {psi:.2f})",
          np.sum((cf - cg) ** 2), hf ** 2 + hg ** 2 + 2 * hf * hg * np.cos(psi), 1e-12)
    close(f"Section 38.6: |e*| = h_f + h_g along the surface (psi = {psi:.2f})",
          np.linalg.norm(cf - m) + np.linalg.norm(cg - m), hf + hg, 1e-12)
V, F = ddg38.hinge(0.0)
ang = dm.corner_angles(V, F)
close("Example 38.6.4: cot theta_2 = 0.4875, cot theta_3 = 2/3", [1 / np.tan(ang[0, 2]), 1 / np.tan(ang[1, 2])],
      [0.4875, 2 / 3], 1e-12)
close("Example 38.6.4: L01 = star1 = 0.577083", [L0[0, 1], dd.hodge1(V, F)[0]], [0.5770833333, 0.5770833333], 1e-9)
n_angle = [ddg38.vertex_normal(*ddg38.hinge(p), "angle")[0] for p in (0.0, 1.0, 2.0)]
check("Example 38.6.4: the angle-weighted normal at x_0 is e_3 at psi = 0 and turns with psi",
      np.allclose(n_angle[0], [0, 0, 1]) and ddg38.angle_deg(n_angle[0], n_angle[1]) > 10
      and ddg38.angle_deg(n_angle[1], n_angle[2]) > 10)
chords = []
for psi in np.deg2rad([0, 60, 120]):
    V, F = ddg38.hinge(psi)
    chords.append(np.linalg.norm(circumcenter(*V[F[0]]) - circumcenter(*V[F[1]])))
close("Non-example 38.6.5 / Figure 38.6.1: chords 0.5771, 0.5018, 0.2988", np.round(chords, 4), [0.5771, 0.5018, 0.2988],
      1e-12)

# ---------------------------------------------------------------- Example 38.6.6
sums, errs_c, errs_b = [], [], []
for level in range(5):
    V, F = dm.icosphere(level)
    L = dm.cotan_laplacian(V, F)
    Ac, Ab = dd.hodge0(V, F, "circumcentric"), dd.hodge0(V, F, "barycentric")
    check(f"Example 38.6.6: icosphere({level}) has no obtuse triangle; circumcentric = mixed > 0",
          dm.corner_angles(V, F).max() < np.pi / 2 and np.allclose(Ac, dd.hodge0(V, F, "mixed")) and Ac.min() > 0)
    sums.append(Ac.sum())
    errs_c.append(np.abs(L @ V[:, 2] / Ac + 2 * V[:, 2]).max())
    errs_b.append(np.abs(L @ V[:, 2] / Ab + 2 * V[:, 2]).max())
    rad = np.einsum("ij,ij->i", (L @ V) / Ac[:, None], V)
    close(f"Example 38.6.6 / Example 31.3.6: radial part of Delta_h x is exactly -2 (level {level})", rad,
          np.full(len(V), -2.0), 1e-11)
    close(f"Example 38.6.6(a): sum of star0 = mesh area (level {level})", Ac.sum(), dm.face_areas(V, F).sum(), 1e-12)
    if level == 4:
        val = np.array([len(n) for n in dm.neighbors(F)])
        eb = np.abs(L @ V[:, 2] / Ab + 2 * V[:, 2])
        check("Example 38.6.6(c): the largest barycentric error sits at a valence-5 vertex", val[eb.argmax()] == 5)
        close("Example 38.6.6(c): circumcentric/barycentric dual area ratio there is about 1.145",
              (Ac / Ab)[eb.argmax()], 1.145, 5e-4)
close("Example 38.6.6(a): sums 9.5745, 11.6659, 12.3298, 12.5065, 12.5514", np.round(sums, 4),
      [9.5745, 11.6659, 12.3298, 12.5065, 12.5514], 1e-12)
deficit = 4 * np.pi - np.array(sums)
close("Example 38.6.6(a): deficits 2.99, 0.900, 0.237, 0.060, 0.015", deficit, [2.99, 0.900, 0.237, 0.060, 0.015], 6e-3)
check("Example 38.6.6(a): deficit ratio about 1/4 per level", np.all(np.abs(deficit[2:] / deficit[1:-1] - 0.25) < 0.02),
      deficit[1:] / deficit[:-1])
check("Example 38.6.6(b): levels 0, 1 exact to 1e-13", max(errs_c[:2]) < 1e-13, errs_c)
close("Example 38.6.6(b): levels 2-4: 0.0284, 0.0150, 0.0076", np.round(errs_c[2:], 4), [0.0284, 0.0150, 0.0076], 1e-12)
close("Example 38.6.6(c): barycentric levels 1-4: 0.174, 0.229, 0.243, 0.247", np.round(errs_b[1:], 3),
      [0.174, 0.229, 0.243, 0.247], 1e-12)

# ---------------------------------------------------------------- Lemma 38.6.7, Theorem 38.6.8, Example 38.6.9
for name, (V, F) in (("jittered icosphere", (Vj, Fj)), ("jittered torus", (Vt2, Ft))):
    N = dm.face_normals(V, F)
    ok = True
    for (i, j) in ddg38.edge_lengths(V, F):
        fs = [f for f in range(len(F)) if i in F[f] and j in F[f]]
        ok &= np.linalg.norm(np.cross(np.cross(N[fs[0]], N[fs[1]]), V[j] - V[i])) < 1e-10
    check(f"Lemma 38.6.7 on {name}: every edge is parallel to N_f x N_g", ok)
Vtet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
Ftet = ddg38.orient_outward(Vtet, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
Vi, Wi, Fi = ddg38.dimpled_icosahedron()
Vjt = Vt + 0.02 * np.random.default_rng(1).normal(size=Vt.shape)
for name, (V, F) in (("icosphere(1)", dm.icosphere(1)), ("icosphere(2)", dm.icosphere(2)), ("octahedron", ddg38.octahedron()),
                     ("tetrahedron", (Vtet, Ftet)), ("icosahedron", (Vi, Fi)), ("dimpled icosahedron", (Wi, Fi)),
                     ("jittered torus 12x8", (Vjt, Ft))):
    check(f"Example 38.6.9: {name} has no flat edge and nullity 4", flat_edges(V, F) == 0 and nullity(V, F) == 4)
    d, Y = sfn(dm.face_normals(V, F), F, len(V))
    X = V - V.mean(0)
    c = np.sum(X * Y) / np.sum(X * X)
    check(f"Theorem 38.6.8: {name}: the recovered solution is c x (relative error < 1e-9)",
          d == 4 and np.abs(Y - c * X).max() < 1e-9 * np.abs(X).max())
for V, F in (dm.icosphere(2), (Vjt, Ft)):
    close("Theorem 38.6.8: -x (point reflection) has the same face normals", dm.face_normals(-V, F), dm.face_normals(V, F), 1e-14)
    close("Theorem 38.6.8: 2.5 x + t has the same face normals", dm.face_normals(2.5 * V + 1.0, F), dm.face_normals(V, F),
          1e-12)

# ---------------------------------------------------------------- Non-example 38.6.10, Exercise 38.6.4
Vb, Fb = ddg38.cube({})
for s in ([1.5, 0.7, 0.45], [0.3, 2.0, 1.0]):
    close(f"Non-example 38.6.10: box diag{tuple(s)} has the same face normals", dm.face_normals(Vb * s, Fb),
          dm.face_normals(Vb, Fb), 1e-14)
for bits in [(0, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 1), (1, 1, 1, 1, 1, 1)]:
    pat = dict(zip(["x+", "x-", "y+", "y-", "z+", "z-"], [bool(b) for b in bits]))
    Vc, Fc = ddg38.cube(pat)
    check(f"Non-example 38.6.10 / Ex 38.6.4: cube pattern {bits}: 6 flat edges and nullity 6",
          flat_edges(Vc, Fc) == 6 and nullity(Vc, Fc) == 6)
    # the solution space is spanned by the six "face coordinate" vectors
    basis = []
    for k in range(3):
        for sgn in (1, -1):
            y = np.zeros((8, 3))
            y[np.sign(Vc[:, k]) == sgn, k] = 1.0
            basis.append(y.ravel())
    A = ddg38.normal_constraint_matrix(Fc, dm.face_normals(Vc, Fc), 8)
    check(f"Exercise 38.6.4(a): the six indicator vectors solve (38.6.3) (pattern {bits})",
          np.abs(A @ np.array(basis).T).max() < 1e-14 and np.linalg.matrix_rank(np.array(basis)) == 6)
for D, same in (([1.5, 0.7, 0.45], True), ([-1.5, -0.7, -0.45], True), ([-1.0, 1.0, 1.0], False), ([-1.0, -1.0, 1.0], False)):
    eq = np.allclose(dm.face_normals(Vb * D, Fb), dm.face_normals(Vb, Fb))
    check(f"Exercise 38.6.4(b): D = {D}: same normals = {same}", eq == same)

# ---------------------------------------------------------------- Non-example 38.6.11, Exercise 38.6.3
T1, T2 = dm.torus_mesh(2.0, 0.8, 12, 8), dm.torus_mesh(1.2, 0.6, 12, 8)
close("Non-example 38.6.11: torus_mesh(2, 0.8) and (1.2, 0.6) have the same face normals", dm.face_normals(*T1),
      dm.face_normals(*T2), 1e-14)
for R, r in ((3.0, 0.5), (1.0, 0.9), (5.0, 2.0)):
    close(f"Non-example 38.6.11: torus_mesh({R}, {r}) has the same face normals", dm.face_normals(*dm.torus_mesh(R, r, 12, 8)),
          dm.face_normals(*T1), 1e-13)
check("Non-example 38.6.11: every quad diagonal is a flat edge (12 x 8)", flat_edges(*T1) == 96)
check("Non-example 38.6.11: nullity 19 on 12 x 8", nullity(*T1) == 19)
check("Non-example 38.6.11: nullity 39 on 24 x 16", nullity(*dm.torus_mesh(2.0, 0.8, 24, 16)) == 39)
Vq = T1[0]
for i, j in ((1, 7), (4, 2), (9, 5)):
    p = [Vq[(i % 12) * 8 + j % 8], Vq[(i % 12) * 8 + (j + 1) % 8], Vq[((i + 1) % 12) * 8 + j % 8],
         Vq[((i + 1) % 12) * 8 + (j + 1) % 8]]
    s1, s2 = p[1] - p[0], p[3] - p[2]
    planar = abs(np.linalg.det(np.array([p[1] - p[0], p[2] - p[0], p[3] - p[0]]))) < 1e-14
    check(f"Exercise 38.6.3: quad ({i}, {j}) is planar with parallel horizontal sides and equal legs",
          planar and np.linalg.norm(np.cross(s1, s2)) < 1e-14 and abs(s1[2]) < 1e-15
          and abs(np.linalg.norm(p[2] - p[0]) - np.linalg.norm(p[3] - p[1])) < 1e-14)

# ---------------------------------------------------------------- Proposition 38.6.12 (smooth parallel tori)
for (R, r) in ((2.0, 0.8), (1.2, 0.6), (3.0, 1.5)):
    for (u, v) in ((0.3, 1.1), (2.0, -0.7), (4.0, 2.5)):
        xu, xv = ddg38.partials(ddg38.torus, u, v, R=R, r=r)
        n = np.cross(xu, xv)
        n /= np.linalg.norm(n)
        N = -np.array([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)])
        close(f"Prop 38.6.12: torus ({R}, {r}): N = (38.3.2), independent of R, r", n, N, 1e-6)
        close(f"Prop 38.6.12: x_(R,r) + s N = x_(R,r-s) (s = 0.3 r)", ddg38.torus(u, v, R=R, r=r) + 0.3 * r * N,
              ddg38.torus(u, v, R=R, r=0.7 * r), 1e-14)
for V, F, chi in (dm.icosphere(2) + (2,), T1 + (0,)):
    check("Section 38.6 counting: n2 = 2 (n0 - chi)", len(F) == 2 * (len(V) - chi))

# ---------------------------------------------------------------- Theorem 38.6.14, Non-example 38.6.15, Figure 38.6.3
e1, e2 = ddg38.edge_lengths(Vi, Fi), ddg38.edge_lengths(Wi, Fi)
check("Non-example 38.6.15: 30 edges, all of the same length 1.0515 in both",
      len(e1) == 30 and np.allclose(list(e1.values()), 1.0514622) and max(abs(e1[k] - e2[k]) for k in e1) < 1e-14)
close("Non-example 38.6.15: same L", dm.cotan_laplacian(Wi, Fi), dm.cotan_laplacian(Vi, Fi), 1e-12)
close("Non-example 38.6.15: volumes 2.5362 and 1.8352", [dm.signed_volume(Vi, Fi), dm.signed_volume(Wi, Fi)],
      [2.5362, 1.8352], 5e-5)
turn = ddg38.angle_deg(dm.face_normals(Vi, Fi), dm.face_normals(Wi, Fi))
check("Non-example 38.6.15: exactly the five faces at x_0 turn, by 74.75 degrees",
      sorted(np.where(turn > 1e-9)[0].tolist()) == sorted(f for f in range(20) if 0 in Fi[f])
      and np.allclose(turn[turn > 1e-9], 74.75, atol=5e-3))
nb = sorted({int(j) for f in Fi if 0 in f for j in f if j != 0})
n0 = Vi[0] / np.linalg.norm(Vi[0])
close("Non-example 38.6.15: the link lies in the plane at height 1/sqrt 5", Vi[nb] @ n0, np.full(5, 1 / np.sqrt(5)), 1e-14)


def convex(V, F):
    N = dm.face_normals(V, F)
    return max(((V - V[F[f, 0]]) @ N[f]).max() for f in range(len(F))) < 1e-12


check("Theorem 38.6.14 / Non-example 38.6.15: the icosahedron is convex, the dimpled one is not",
      convex(Vi, Fi) and not convex(Wi, Fi))
check("Non-example 38.6.15: the dimpled icosahedron is still a discrete immersion (projection test at every vertex)",
      all(ddg38.star_passes(Wi, Fi, k, ddg38.vertex_normal(Wi, Fi, "angle")[k]) for k in range(12)))

# ---------------------------------------------------------------- Exercises 38.6.1, 38.6.2, 38.6.5
close("Exercise 38.6.1(a): 3-4-5 triangle: A = 6, cots 0, 4/3, 3/4",
      [0.25 * np.sqrt(12 * 6 * 4 * 2), (9 + 16 - 25) / 24, (16 + 25 - 9) / 24, (9 + 25 - 16) / 24], [6, 0, 4 / 3, 3 / 4], 1e-15)
E, _ = dd.edge_list(Fb)
diag_edges = [k for k, (i, j) in enumerate(E) if np.sum(Vb[i] != Vb[j]) == 2]
check("Exercise 38.6.1(c): the six square diagonals of the cube have star1 = 0",
      len(diag_edges) == 6 and np.abs(dd.hodge1(Vb, Fb)[diag_edges]).max() < 1e-14)
th, ph = sp.symbols("theta phi", positive=True)
f = sp.cos(th)
lap_f = sp.diff(sp.sin(th) * sp.diff(f, th), th) / sp.sin(th) + sp.diff(f, ph, 2) / sp.sin(th) ** 2
check("Exercise 38.6.2: Delta cos(theta) = -2 cos(theta) on the unit sphere", sp.simplify(lap_f + 2 * f) == 0)
for xf in (sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph)):
    lap_x = sp.diff(sp.sin(th) * sp.diff(xf, th), th) / sp.sin(th) + sp.diff(xf, ph, 2) / sp.sin(th) ** 2
    check("Exercise 38.6.2: Delta x = -2 x and Delta y = -2 y", sp.simplify(lap_x + 2 * xf) == 0)
for V, F in (dm.icosphere(2), (Vjt, Ft), (Wi, Fi)):
    t = rng.normal(size=3)
    close("Exercise 38.6.5(a): Vol(x + t) = Vol(x)", dm.signed_volume(V + t, F), dm.signed_volume(V, F), 1e-12)
    close("Exercise 38.6.5(a): Vol(c x) = c^3 Vol(x)", dm.signed_volume(-1.7 * V, F), (-1.7) ** 3 * dm.signed_volume(V, F),
          1e-12)
    check("Exercise 38.6.5(b): Vol(x) > 0 and Vol(-x) < 0", dm.signed_volume(V, F) > 0 > dm.signed_volume(-V, F))

summary()

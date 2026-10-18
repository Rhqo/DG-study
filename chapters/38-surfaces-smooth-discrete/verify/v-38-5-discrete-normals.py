"""Verification for Section 38.5 (Discrete Normals and the Discrete Gauss Map).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-5-discrete-normals.py
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "38-5-discrete-normals.html"
rng = np.random.default_rng(385)
KINDS = ("uniform", "area", "angle", "vector-mixed")
diag = np.ones(3) / np.sqrt(3)


def face_normals_area(V, F):
    cr = dm.face_cross(V, F)
    A = 0.5 * np.linalg.norm(cr, axis=1)
    return cr / (2 * A[:, None]), A


def random_rotation():
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.linalg.det(Q))


def random_star(m=6, jitter=0.25):
    phi = np.linspace(0, 2 * np.pi, m, endpoint=False) + jitter * rng.uniform(-1, 1, m)
    link = np.c_[(1 + 0.3 * rng.uniform(-1, 1, m)) * np.cos(phi), (1 + 0.3 * rng.uniform(-1, 1, m)) * np.sin(phi),
                 0.3 * rng.uniform(-1, 1, m)]
    V = np.vstack([[0.05, -0.03, 0.2], link])
    F = np.array([[0, 1 + k, 1 + (k + 1) % m] for k in range(m)])
    return V, F


def ordered_star(F, i):
    """faces at interior vertex i in orientation order: (i, j_k, j_{k+1})."""
    star = [list(f) for f in F if i in f]
    rot = [f[f.index(i):] + f[:f.index(i)] for f in star]
    order = [rot[0]]
    while len(order) < len(rot):
        nxt = [f for f in rot if f[1] == order[-1][2]]
        order.append(nxt[0])
    return order


# ---------------------------------------------------------------- Implementation 38.5 and Exercise 38.5.5
check("Implementation 38.5: at most 30 lines", dgcode.code_lines(PAGE, "code-38-5-1") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-5-1")
check("Implementation 38.5: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-38-5-1").strip(),
      out)
four = ns["vertex_normals_four"]
V, F = dm.icosphere(2)
Vt, Ft = dm.torus_mesh(2.0, 0.8, 24, 16)
Vj = dm.icosphere(2)[0] * (1 + 0.05 * rng.uniform(-1, 1, (len(V), 1)))
for name, (W, G) in (("icosphere(2)", (V, F)), ("torus (24, 16)", (Vt, Ft)), ("jittered icosphere", (Vj, F))):
    nn = four(W, G)
    for k, kind in zip(("uniform", "area", "angle", "vector"), KINDS):
        close(f"Implementation 38.5: '{k}' agrees with ddg38.vertex_normal on {name}", nn[k],
              ddg38.vertex_normal(W, G, kind), 1e-12)
    for k in ("uniform", "area", "angle"):
        close(f"Implementation 38.5: '{k}' agrees with dgmesh.vertex_normals on {name}", nn[k],
              dm.vertex_normals(W, G, kind=k), 1e-12)


def errors(nu, nv):
    W, G = dm.torus_mesh(2.0, 0.8, nu, nv)
    core = 2.0 * np.c_[W[:, :2] / np.linalg.norm(W[:, :2], axis=1, keepdims=True), np.zeros(len(W))]
    ex = (W - core) / 0.8
    return np.array([ddg38.angle_deg(ddg38.vertex_normal(W, G, k), ex).max() for k in KINDS])


e1, e2 = errors(24, 16), errors(48, 32)
check("Example 38.5.10: doubling the torus grid divides each error by about 4", np.all((e1 / e2 > 3.5) & (e1 / e2 < 4.6)),
      e1 / e2)
check("Example 38.5.10: on the torus (24, 16) uniform is the most accurate, vector area the least",
      e1.argmin() == 0 and e1.argmax() == 3, e1)
ei = np.array([ddg38.angle_deg(ddg38.vertex_normal(V, F, k), V).max() for k in KINDS])
check("Example 38.5.10: on icosphere(2) angle-weighted is the most accurate", ei.argmin() == 2, ei)
check("Example 38.5.10: the icosphere vertices lie on the unit sphere", np.allclose(np.linalg.norm(V, axis=1), 1))

check("Exercise 38.5.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-38-5-5") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-5-5", ns)
check("Exercise 38.5.5: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-ex-38-5-5").strip(),
      out)
Av = ddg38.vector_area(V, F, "barycentric")
close("Exercise 38.5.5: the solution's vector area equals ddg38.vector_area (barycentric)", ns["vector_area_bary"](V, F),
      Av, 1e-15)
W = V.copy()
W[17] += np.array([0.03, -0.05, 0.08])
Aw = ddg38.vector_area(W, F, "barycentric")
changed = sorted(np.where(np.linalg.norm(Aw - Av, axis=1) > 1e-14)[0].tolist())
check("Exercise 38.5.5(b): moving vertex 17 changes exactly the vector areas of its neighbors",
      changed == dm.neighbors(F)[17] and changed == [60, 61, 62, 63, 105, 108], changed)
ratio = np.linalg.norm(Av, axis=1) / dm.mass_barycentric(V, F)
check("Exercise 38.5.5(c): |A_i| < A_i everywhere, ratio in [0.9842, 0.9856]",
      ratio.max() < 1 and 0.9842 < ratio.min() < 0.98435 and 0.98545 < ratio.max() < 0.9856, (ratio.min(), ratio.max()))

# ---------------------------------------------------------------- Definition 38.5.1, Equation (38.5.2)
for _ in range(5):
    xi, xj, xk = rng.normal(size=(3, 3))
    cr = np.cross(xj - xi, xk - xi)
    close("Eq (38.5.2): A_f N_f = (x_i x x_j + x_j x x_k + x_k x x_i) / 2",
          0.5 * cr, 0.5 * (np.cross(xi, xj) + np.cross(xj, xk) + np.cross(xk, xi)), 1e-13)
    close("Def 38.5.1: N_f is invariant under cyclic relabeling", np.cross(xk - xj, xi - xj), cr, 1e-13)
    t = rng.normal(size=3)
    close("Eq (38.5.2): A_f N_f is translation invariant", np.cross(xj - xi, xk - xi),
          np.cross(xj + t - xi - t, xk + t - xi - t), 1e-13)

# ---------------------------------------------------------------- Proposition 38.5.2
for name, (W, G) in (("jittered icosphere", (Vj, F)), ("torus (24, 16)", (Vt, Ft))):
    N, _ = face_normals_area(W, G)
    fidx = {tuple(f): n for n, f in enumerate(G.tolist())}
    ok_perp, ok_anti, ok_arc, ok_closed = True, True, True, True
    for i in range(0, len(W), 7):
        star = ordered_star(G, i)
        rows = []
        for f in star:
            key = [tuple(f), tuple(f[1:] + f[:1]), tuple(f[2:] + f[:2])]
            rows.append([fidx[k] for k in key if k in fidx][0])
        m = len(star)
        ok_closed &= star[-1][2] == star[0][1]
        for k in range(m):
            a, b = N[rows[k]], N[rows[(k + 1) % m]]
            e = W[star[k][2]] - W[i]                      # shared edge (i, j_{k+1})
            ok_perp &= abs(a @ e) < 1e-12 and abs(b @ e) < 1e-12
            ok_anti &= a @ b > -1 + 1e-9
            for s in np.linspace(0, 1, 5):
                g = (1 - s) * a + s * b
                ok_arc &= abs((g / np.linalg.norm(g)) @ e) < 1e-12
    check(f"Prop 38.5.2(a) on {name}: consecutive normals are perpendicular to the shared edge", ok_perp)
    check(f"Prop 38.5.2(a) on {name}: consecutive normals are not antipodal", ok_anti)
    check(f"Prop 38.5.2(a) on {name}: the great arc between them stays perpendicular to the edge", ok_arc)
    check(f"Prop 38.5.2(b) on {name}: the faces around a vertex close up", ok_closed)

# ---------------------------------------------------------------- Example 38.5.3, Figure 38.5.1
Vo, Fo = ddg38.octahedron()
No, Ao = face_normals_area(Vo, Fo)
star = ordered_star(Fo, 4)
fidx = {tuple(f): n for n, f in enumerate(Fo.tolist())}
Ns = []
for f in star:
    key = [tuple(f), tuple(f[1:] + f[:1]), tuple(f[2:] + f[:2])]
    Ns.append(No[[fidx[k] for k in key if k in fidx][0]])
Ns = np.array(Ns)
check("Example 38.5.3: the four normals are (+-1, +-1, 1)/sqrt 3",
      np.allclose(np.abs(Ns) * np.sqrt(3), 1) and np.allclose(Ns[:, 2] * np.sqrt(3), 1))
check("Example 38.5.3: neighboring normals differ in the sign of one component",
      all(np.sum(np.sign(Ns[k]) != np.sign(Ns[(k + 1) % 4])) == 1 for k in range(4)))
inner = []
for k in range(4):
    a, b, c = Ns[k - 1], Ns[k], Ns[(k + 1) % 4]
    ta, tc = a - (a @ b) * b, c - (c @ b) * b
    inner.append(np.arccos(ta @ tc / np.linalg.norm(ta) / np.linalg.norm(tc)))
close("Example 38.5.3: interior angles of the spherical quadrilateral are 2 pi / 3", np.array(inner),
      np.full(4, 2 * np.pi / 3), 1e-12)
close("Example 38.5.3: Girard area 4 (2 pi / 3) - 2 pi = 2 pi / 3", sum(inner) - 2 * np.pi, 2 * np.pi / 3, 1e-12)
close("Example 38.5.3: the angle defect of the vertex is 2 pi - 4 pi / 3", dm.angle_defect(Vo, Fo)[4], 2 * np.pi / 3,
      1e-12)
# Monte Carlo area of the spherical quadrilateral {|x| < z, |y| < z on S^2 within the four planes}: region of the
# sphere on the positive side of the four great circles through consecutive normals
P = rng.normal(size=(400000, 3))
P /= np.linalg.norm(P, axis=1, keepdims=True)
inside = np.ones(len(P), bool)
for k in range(4):
    n = np.cross(Ns[k], Ns[(k + 1) % 4])
    inside &= P @ n > 0
check("Example 38.5.3: Monte Carlo area of the spherical polygon is 2 pi / 3 (+- 0.02)",
      abs(4 * np.pi * inside.mean() - 2 * np.pi / 3) < 0.02, 4 * np.pi * inside.mean())

# ---------------------------------------------------------------- Proposition 38.5.5, Figure 38.5.2
for _ in range(4):
    Vs, Fs = random_star(int(rng.integers(5, 9)))
    m = len(Vs) - 1
    # (a) the barycentric quadrilateral has area A_f / 3, computed from its four corners
    ok = True
    for f in Fs:
        a, b, c = Vs[f]
        quad = [a, (a + b) / 2, (a + b + c) / 3, (a + c) / 2]
        area = 0.5 * np.linalg.norm(np.cross(quad[1] - quad[0], quad[2] - quad[0])) + \
            0.5 * np.linalg.norm(np.cross(quad[2] - quad[0], quad[3] - quad[0]))
        ok &= abs(area - 0.5 * np.linalg.norm(np.cross(b - a, c - a)) / 3) < 1e-14
    check("Prop 38.5.5(a): barycentric piece has area A_f / 3", ok)
    link_sum = sum(np.cross(Vs[1 + k], Vs[1 + (k + 1) % m]) for k in range(m)) / 2
    close("Prop 38.5.5(b): sum A_f N_f = (1/2) sum x_jk x x_jk+1", dm.face_cross(Vs, Fs).sum(0) / 2, link_sum, 1e-13)
    W = Vs.copy()
    W[0] += rng.normal(size=3)
    close("Prop 38.5.5(b): A_i does not depend on x_i", ddg38.vector_area(W, Fs)[0], ddg38.vector_area(Vs, Fs)[0], 1e-13)
for name, (W, G) in (("icosphere(2)", (V, F)), ("torus (24, 16)", (Vt, Ft)), ("jittered icosphere", (Vj, F)),
                     ("octahedron", (Vo, Fo))):
    check(f"Prop 38.5.5(c) on {name}: sum_f A_f N_f = 0 and sum_i A_i = 0",
          np.abs(dm.face_cross(W, G).sum(0)).max() < 1e-12 and np.abs(ddg38.vector_area(W, G).sum(0)).max() < 1e-12)
Vt2 = Vt + 0.05 * rng.normal(size=Vt.shape)
check("Prop 38.5.5(c) on a jittered torus: sum_i A_i = 0", np.abs(ddg38.vector_area(Vt2, Ft).sum(0)).max() < 1e-12)

phi = np.deg2rad([0, 70, 150, 200, 290])
link = np.c_[np.array([1.0, 0.8, 1.1, 0.9, 1.0]) * np.cos(phi), np.array([1.0, 0.8, 1.1, 0.9, 1.0]) * np.sin(phi),
             [0.1, 0.3, -0.1, 0.2, -0.2]]
Fs = np.array([[0, 1 + k, 1 + (k + 1) % 5] for k in range(5)])
res = []
for c in ([0.05, 0, 0.5], [-0.25, 0.2, 0]):
    Vs = np.vstack([c, link])
    res.append((ddg38.vector_area(Vs, Fs)[0], dm.mass_barycentric(Vs, Fs)[0],
                ddg38.star_passes(Vs, Fs, 0, np.array([0, 0, 1.0]))))
close("Figure 38.5.2: the vector areas of the two stars agree", res[0][0], res[1][0], 1e-14)
close("Figure 38.5.2: A_i = (1/6) sum x_jk x x_jk+1", res[0][0], sum(np.cross(link[k], link[(k + 1) % 5])
                                                                     for k in range(5)) / 6, 1e-14)
check("Figure 38.5.2: the dual areas differ", abs(res[0][1] - res[1][1]) > 0.05, (res[0][1], res[1][1]))
check("Figure 38.5.2: both stars pass the projection test with n = e_3", res[0][2] and res[1][2])
check("Def 38.5.4: |A_i| <= A_i with equality only for a planar star",
      all(np.linalg.norm(r[0]) < r[1] for r in res))
Vp = np.vstack([[0.1, 0.1, 0], np.c_[link[:, :2], np.zeros(5)]])
close("Def 38.5.4: equality |A_i| = A_i for a planar star", np.linalg.norm(ddg38.vector_area(Vp, Fs)[0]),
      dm.mass_barycentric(Vp, Fs)[0], 1e-14)

# ---------------------------------------------------------------- Definition 38.5.6, Non-example 38.5.7
close("Def 38.5.6: barycentric vector-area normal = area-weighted normal", ddg38.vertex_normal(Vj, F, "vector-barycentric"),
      ddg38.vertex_normal(Vj, F, "area"), 1e-12)
Vn = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.1, 0]])
Fn = np.array([[0, 1, 2]])
close("Non-example 38.5.7: circumcentric pieces -0.1375, -0.1375, 0.325",
      ddg38.dual_cell_pieces(Vn, Fn, "circumcentric")[0], [-0.1375, -0.1375, 0.325], 1e-12)
close("Non-example 38.5.7: mixed pieces 0.0125, 0.0125, 0.025", ddg38.dual_cell_pieces(Vn, Fn, "mixed")[0],
      [0.0125, 0.0125, 0.025], 1e-12)
close("Non-example 38.5.7: area 0.05 and the pieces sum to it", ddg38.dual_cell_pieces(Vn, Fn, "circumcentric")[0].sum(),
      0.05, 1e-12)
close("Non-example 38.5.7: the angle at x_k is about 157 degrees", np.degrees(dm.corner_angles(Vn, Fn)[0, 2]), 157.38,
      0.01)
# the circumcentric piece agrees with the area of [x_i, m_ij, c_f] + [x_i, c_f, m_ik] (signed) from the circumcenter
a, b, c = Vn
d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
cc = np.array([ux, uy, 0])
sarea = lambda p, q, r: 0.5 * np.cross(q - p, r - p)[2]
close("Non-example 38.5.7: piece at x_i = signed area of [x_i, m_ij, c] + [x_i, c, m_ik]",
      sarea(a, (a + b) / 2, cc) + sarea(a, cc, (a + c) / 2), -0.1375, 1e-12)
check("Non-example 38.5.7: the circumcenter lies outside the triangle", cc[1] < 0, cc)

# ---------------------------------------------------------------- Proposition 38.5.8
Vs, Fs = random_star(6)
Vs[1:, 2] = 0.0
Vs[0, 2] = 0.0
nn = [ddg38.vertex_normal(Vs, Fs, k)[0] for k in KINDS]
check("Prop 38.5.8(a): planar star: all four normals are e_3", all(np.allclose(n, [0, 0, 1]) for n in nn))
Q, tvec = random_rotation(), rng.normal(size=3)
for k in KINDS:
    close(f"Prop 38.5.8(b): '{k}' is rigid-motion equivariant", ddg38.vertex_normal(Vj @ Q.T + tvec, F, k),
          ddg38.vertex_normal(Vj, F, k) @ Q.T, 1e-12)
    close(f"Prop 38.5.8(b): '{k}' is scale invariant", ddg38.vertex_normal(2.7 * Vj, F, k), ddg38.vertex_normal(Vj, F, k),
          1e-12)
for trial in range(4):
    Vs, Fs = random_star(int(rng.integers(5, 8)))
    f = Fs[1]
    lam = rng.dirichlet(np.ones(3))
    p = lam @ Vs[f]
    W = np.vstack([Vs, p])
    q = len(Vs)
    G1 = np.vstack([np.delete(Fs, 1, axis=0), [[f[0], f[1], q], [f[0], q, f[2]], [f[1], f[2], q]]])
    # edge split of the edge (j, k) opposite i
    s = rng.uniform(0.2, 0.8)
    W2 = np.vstack([Vs, (1 - s) * Vs[f[1]] + s * Vs[f[2]]])
    G2 = np.vstack([np.delete(Fs, 1, axis=0), [[f[0], f[1], q], [f[0], q, f[2]]]])
    for nm, (WW, GG) in (("1-to-3 split", (W, G1)), ("edge split", (W2, G2))):
        close(f"Prop 38.5.8(c): angle-weighted normal is invariant under a {nm} (trial {trial})",
              ddg38.vertex_normal(WW, GG, "angle")[0], ddg38.vertex_normal(Vs, Fs, "angle")[0], 1e-12)
        d_u = ddg38.angle_deg(ddg38.vertex_normal(WW, GG, "uniform")[0], ddg38.vertex_normal(Vs, Fs, "uniform")[0])
        d_a = ddg38.angle_deg(ddg38.vertex_normal(WW, GG, "area")[0], ddg38.vertex_normal(Vs, Fs, "area")[0])
        if nm == "1-to-3 split":
            check(f"Prop 38.5.8(d): uniform and area-weighted normals move under a {nm} (trial {trial})",
                  d_u > 1e-3 and d_a > 1e-3, (d_u, d_a))
        else:
            check(f"Prop 38.5.8(d): the uniform normal moves under an {nm} (trial {trial})", d_u > 1e-3, d_u)

# ---------------------------------------------------------------- Example 38.5.9, Figure 38.5.3, Exercises 38.5.3, 38.5.4
w211 = np.array([2.0, 1.0, 1.0]) / np.sqrt(6)
VA, FA = ddg38.cube({"x+": True})
check("Example 38.5.9: corner (1, 1, 1) is vertex 7; all 12 triangles have area 2",
      np.allclose(VA[7], 1) and np.allclose(dm.face_areas(VA, FA), 2))
close("Example 38.5.9(a): uniform = (2, 1, 1)/sqrt 6", ddg38.vertex_normal(VA, FA, "uniform")[7], w211, 1e-12)
close("Example 38.5.9(a): area-weighted = (2, 1, 1)/sqrt 6", ddg38.vertex_normal(VA, FA, "area")[7], w211, 1e-12)
close("Example 38.5.9(a): angle-weighted = (1, 1, 1)/sqrt 3", ddg38.vertex_normal(VA, FA, "angle")[7], diag, 1e-12)
close("Example 38.5.9(a): mixed vector-area = (1, 1, 1)/sqrt 3", ddg38.vertex_normal(VA, FA, "vector-mixed")[7], diag, 1e-12)
close("Example 38.5.9(a): deviation 19.47 degrees", ddg38.angle_deg(w211, diag), 19.4712, 1e-4)
VB, FB = ddg38.cube({})
check("Example 38.5.9(b): all four normals are (1, 1, 1)/sqrt 3",
      all(np.allclose(ddg38.vertex_normal(VB, FB, k)[7], diag) for k in KINDS))
P = ddg38.dual_cell_pieces(VA, FA, "mixed")
ang = dm.corner_angles(VA, FA)
check("Example 38.5.9 / Ex 38.5.3: Voronoi piece 1 at the right corner, 1/2 at the acute corners",
      np.allclose(P[np.isclose(ang, np.pi / 2)], 1) and np.allclose(P[np.isclose(ang, np.pi / 4)], 0.5))
dirs = set()
ok_angle = True
for bits in product([False, True], repeat=6):
    pat = dict(zip(["x+", "x-", "y+", "y-", "z+", "z-"], bits))
    Vc, Fc = ddg38.cube(pat)
    ok_angle &= np.allclose(ddg38.vertex_normal(Vc, Fc, "angle")[7], diag)
    ok_angle &= np.allclose(ddg38.vertex_normal(Vc, Fc, "vector-mixed")[7], diag)
    ok_angle &= np.allclose(ddg38.vertex_normal(Vc, Fc, "area")[7], ddg38.vertex_normal(Vc, Fc, "uniform")[7])
    cnt = np.zeros(3)
    for f in Fc:
        if 7 in f:
            cnt += np.abs(np.round(dm.face_normals(Vc, f[None])[0]))
    dirs.add(tuple(int(c) for c in cnt))
    close("Exercise 38.5.3: uniform numerator = (c_1, c_2, c_3)", ddg38.vertex_normal(Vc, Fc, "uniform")[7],
          cnt / np.linalg.norm(cnt), 1e-12)
check("Exercise 38.5.3: angle-weighted and mixed vector-area are (1,1,1)/sqrt 3 for all 64 patterns; area = uniform",
      ok_angle)
check("Exercise 38.5.3: c_k in {1, 2}, all 8 combinations occur", dirs == set(product([1, 2], repeat=3)), dirs)
# Exercise 38.5.4: pattern B, split the corner triangle of the square x = 1 at its barycenter
fx = [n for n, f in enumerate(FB) if 7 in f and np.allclose(dm.face_normals(VB, f[None])[0], [1, 0, 0])]
check("Exercise 38.5.4: pattern B has exactly one corner triangle in the square x = 1", len(fx) == 1)
f = FB[fx[0]]
W = np.vstack([VB, VB[f].mean(0)])
G = np.vstack([np.delete(FB, fx[0], axis=0), [[f[0], f[1], 8], [f[1], f[2], 8], [f[2], f[0], 8]]])
check("Exercise 38.5.4: the split mesh is a closed oriented surface",
      dm.euler_characteristic(G) == 2 and dm.oriented_consistently(G) and dm.is_manifold(G))
close("Exercise 38.5.4: uniform moves by 19.47 degrees", ddg38.angle_deg(ddg38.vertex_normal(W, G, "uniform")[7], diag),
      np.degrees(np.arccos(4 / np.sqrt(18))), 1e-9)
close("Exercise 38.5.4: arccos(4/sqrt 18) = 19.47", np.degrees(np.arccos(4 / np.sqrt(18))), 19.47, 0.005)
close("Exercise 38.5.4: area-weighted moves by 10.02 degrees", ddg38.angle_deg(ddg38.vertex_normal(W, G, "area")[7], diag),
      np.degrees(np.arccos(8 / np.sqrt(66))), 1e-9)
close("Exercise 38.5.4: arccos(8/sqrt 66) = 10.02", np.degrees(np.arccos(8 / np.sqrt(66))), 10.02, 0.005)
close("Exercise 38.5.4: angle-weighted does not move", ddg38.vertex_normal(W, G, "angle")[7], diag, 1e-12)
at7 = [g for g in G if 7 in g and np.allclose(dm.face_normals(W, g[None])[0], [1, 0, 0])]
close("Exercise 38.5.4: two faces of area 2/3 in x = 1 at the corner", dm.face_areas(W, np.array(at7)), [2 / 3, 2 / 3], 1e-12)
# R2 review: the mixed Voronoi vector-area normal is not subdivision invariant either (Prop 38.5.8(d))
ang7 = dm.corner_angles(W, np.array(at7))
check("Exercise 38.5.4: the two corner faces in x = 1 are obtuse at the barycenter (108.43 deg), acute at the corner",
      np.allclose(np.sort(np.degrees(ang7.max(axis=1))), np.degrees(np.arccos(-1 / np.sqrt(10))), atol=1e-9)
      and all(ang7[k, list(at7[k]).index(7)] < np.pi / 2 for k in range(2)), np.degrees(ang7))
close("Exercise 38.5.4: mixed vector-area numerator at the corner is (1/3, 1, 1)",
      ddg38.vector_area(W, G, "mixed")[7], np.array([1 / 3, 1, 1]), 1e-12)
close("Exercise 38.5.4: mixed vector-area moves by arccos(7/sqrt 57) = 22.00 degrees",
      ddg38.angle_deg(ddg38.vertex_normal(W, G, "vector-mixed")[7], diag), np.degrees(np.arccos(7 / np.sqrt(57))), 1e-9)
close("Exercise 38.5.4: arccos(7/sqrt 57) = 22.00", np.degrees(np.arccos(7 / np.sqrt(57))), 22.00, 0.005)
# Prop 38.5.8(c), Eq. (38.5.6): angle-weighted numerator = (2/eps^2) * vector area of the star inside B_eps(x_i)
for (Vt_, Ft_, name) in ((VA, FA, "cube pattern A"), (W, G, "split cube")):
    Ft_ = np.asarray(Ft_)
    num = np.zeros(3)
    eps = 0.05
    for g in Ft_[(Ft_ == 7).any(axis=1)]:
        k = list(g).index(7)
        a, b = Vt_[g[(k + 1) % 3]] - Vt_[7], Vt_[g[(k + 2) % 3]] - Vt_[7]
        th = np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)
        sector = 0.5 * eps ** 2 * th                      # area of the face inside the ball
        num += (2 / eps ** 2) * sector * dm.face_normals(Vt_, g[None])[0]
    close(f"Eq (38.5.6): angle-weighted numerator = 2/eps^2 * sum of sector areas * N_f ({name})",
          num / np.linalg.norm(num), ddg38.vertex_normal(Vt_, Ft_, "angle")[7], 1e-12)
check("Prop 38.5.8(c): a flip (pattern A -> B) keeps the angle-weighted normal and moves uniform/area",
      np.allclose(ddg38.vertex_normal(VA, FA, "angle")[7], ddg38.vertex_normal(VB, FB, "angle")[7])
      and not np.allclose(ddg38.vertex_normal(VA, FA, "uniform")[7], ddg38.vertex_normal(VB, FB, "uniform")[7]))

# ---------------------------------------------------------------- Exercise 38.5.1 (tetrahedron)
Vtet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
Ftet = ddg38.orient_outward(Vtet, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
cr = dm.face_cross(Vtet, Ftet) / 2
ok = True
for n, f in enumerate(Ftet):
    m = ({0, 1, 2, 3} - set(f.tolist())).pop()
    ok &= np.allclose(cr[n], -2 * Vtet[m])
check("Exercise 38.5.1: A_f N_f of the face opposite x_m is -2 x_m", ok)
check("Exercise 38.5.1: face (1, 3, 2) is the outward orientation opposite x_0",
      any(list(f) in ([1, 3, 2], [3, 2, 1], [2, 1, 3]) for f in Ftet.tolist()))
close("Exercise 38.5.1: |A_f N_f| = 2 sqrt 3 = area of an equilateral triangle of side 2 sqrt 2",
      np.linalg.norm(cr[0]), np.sqrt(3) / 4 * 8, 1e-12)
close("Exercise 38.5.1: the sum vanishes", cr.sum(0), np.zeros(3), 1e-14)
for k in KINDS:
    close(f"Exercise 38.5.1: '{k}' vertex normals are x_i / |x_i|", ddg38.vertex_normal(Vtet, Ftet, k),
          Vtet / np.sqrt(3), 1e-12)

# ---------------------------------------------------------------- Exercise 38.5.2 (saddle star)
h = 0.5
Vsd = np.array([[0, 0, 0], [1, 0, h], [0, 1, -h], [-1, 0, h], [0, -1, -h]], float)
Fsd = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
Nsd = dm.face_normals(Vsd, Fsd)
close("Exercise 38.5.2(a): normals (-h, h, 1), (h, h, 1), (h, -h, 1), (-h, -h, 1) normalized", Nsd,
      np.array([[-h, h, 1], [h, h, 1], [h, -h, 1], [-h, -h, 1]]) / np.sqrt(1 + 2 * h * h), 1e-14)
az = np.degrees(np.arctan2(Nsd[:, 1], Nsd[:, 0]))
close("Exercise 38.5.2(b): xy directions 135, 45, -45, -135 (clockwise)", az, [135, 45, -45, -135], 1e-12)
signed = sum(np.cross(Nsd[k], Nsd[(k + 1) % 4]) @ [0, 0, 1] for k in range(4))
check("Exercise 38.5.2(b): the polygon of normals turns clockwise about e_3", signed < 0, signed)
close("Exercise 38.5.2(c): cos theta = -h^2/(1 + h^2) = -0.2", (Vsd[1] @ Vsd[2]) / (Vsd[1] @ Vsd[1]), -0.2, 1e-14)
close("Exercise 38.5.2(c): each angle 101.54 degrees", np.degrees(dm.corner_angles(Vsd, Fsd)[:, 0]), np.full(4, 101.537), 1e-3)
close("Exercise 38.5.2(c): angle sum 406.15 degrees", np.degrees(dm.corner_angles(Vsd, Fsd)[:, 0].sum()), 406.15, 0.005)

summary()

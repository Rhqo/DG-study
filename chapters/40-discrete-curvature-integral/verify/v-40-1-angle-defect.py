"""Verification for Section 40.1 (Angle Defect and Discrete Gaussian Curvature).

Run from the project root: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/verify/v-40-1-angle-defect.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "40-1-angle-defect.html"

# ---------------------------------------------------------------- Implementation 40.1 and Exercise 40.1.5
check("Implementation 40.1: at most 30 lines", dgcode.code_lines(PAGE, "code-40-1-1") <= 30)
ns, out = ddg40.run_printed(PAGE, "code-40-1-1")
check("Implementation 40.1: printed output matches the page", out.strip() == ddg40.printed_block(PAGE, "out-40-1-1").strip(), out)
ad = ns["angle_defects"]
meshes = {"icosphere(3)": dm.icosphere(3), "torus_mesh(2, 0.8, 12, 24)": dm.torus_mesh(2, 0.8, 12, 24),
          "grid 6x5": dm.grid_mesh(np.linspace(0, 1, 6), np.linspace(0, 2, 5)), "saddle grid": ddg40.saddle_grid(9, 0.5),
          "example star": ddg40.example_star(), "cap": ddg40.icosphere_cap(2, 1.2)}
for name, (V, F) in meshes.items():
    close(f"Implementation 40.1: angle_defects = dgmesh.angle_defect on {name}", ad(V, F), dm.angle_defect(V, F), 1e-12)
ns2, out2 = ddg40.run_printed(PAGE, "code-ex-40-1-5", dict(ns))
check("Exercise 40.1.5: printed output matches the page", out2.strip() == ddg40.printed_block(PAGE, "out-ex-40-1-5").strip(), out2)

# ---------------------------------------------------------------- Example 40.1.2
for name, (V, F), val in [("tetrahedron", ddg40.tetrahedron(), np.pi), ("octahedron", ddg40.octahedron(), 2 * np.pi / 3),
                          ("icosahedron", dm.icosahedron(), np.pi / 3)]:
    close(f"Example 40.1.2(a): {name} delta", dm.angle_defect(V, F), np.full(len(V), val), 1e-12)
for pat in ("avoid", "max", "alt"):
    V, F = ddg40.unit_cube(pat)
    close(f"Example 40.1.2(b): unit cube ({pat}) delta = pi/2 at every corner", dm.angle_defect(V, F), np.full(8, np.pi / 2), 1e-12)
V, F = dm.grid_mesh(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
d = dm.angle_defect(V, F)
close("Example 40.1.2(c): grid corners pi/2, everything else 0", d, np.where(np.isin(np.arange(25), [0, 4, 20, 24]), np.pi / 2, 0), 1e-12)
close("Example 40.1.2(c): the four corners add to 2 pi", d.sum(), 2 * np.pi, 1e-12)

# ---------------------------------------------------------------- Non-example 40.1.3: two tetrahedra sharing a vertex
Vt, Ft = ddg40.tetrahedron()
V2 = np.vstack([Vt, 2 * Vt[0] - Vt[1:]])                       # point reflection of the tetrahedron through x_0
F2 = np.vstack([Ft, np.where(Ft == 0, 0, Ft + 3)])
th = dm.corner_angles(V2, F2)
Theta0 = sum(th[f, list(F2[f]).index(0)] for f in range(len(F2)) if 0 in F2[f])
close("Non-example 40.1.3: the pinched vertex has angle sum 2 pi", Theta0, 2 * np.pi, 1e-12)
lk = dm.vertex_link(F2, 0)
check("Non-example 40.1.3: its link is not a cycle", dm.link_type(lk) != "cycle")

# ---------------------------------------------------------------- Proposition 40.1.4 and Exercise 40.1.3
def counts(F):
    return dm.counts(F)


def chi(F):
    n0, n1, n2 = counts(F)
    return n0 - n1 + n2


V, F = dm.icosphere(1)
c0 = counts(F)
# (a) edge flip of the edge shared by faces 0 and its neighbour
ef = dm.edge_faces(F)
(i, j), (f, g) = next((e, fs) for e, fs in ef.items() if len(fs) == 2)
k = [v for v in F[f] if v not in (i, j)][0]
l = [v for v in F[g] if v not in (i, j)][0]
Fa = [t for n, t in enumerate(F.tolist()) if n not in (f, g)]
# keep the orientation: in the face that traverses i -> j write (i, j, k); the flipped faces are (k, l, j), (l, k, i)
tf = list(F[f])
a = tf.index(i)
if tf[(a + 1) % 3] != j:
    i, j = j, i
Fa += [[k, l, j], [l, k, i]]
Fa = np.array(Fa)
check("Proposition 40.1.4(a): an edge flip keeps (n0, n1, n2)", counts(Fa) == c0 and dm.oriented_consistently(Fa))
# (b) face split
p = len(V)
Fb = np.vstack([np.delete(F, 0, axis=0), [[F[0, 0], F[0, 1], p], [F[0, 1], F[0, 2], p], [F[0, 2], F[0, 0], p]]])
check("Proposition 40.1.4(b): a face split changes the counts by (1, 3, 2)",
      np.subtract(counts(Fb), c0).tolist() == [1, 3, 2] and chi(Fb) == 2)
# (c) edge split
Fc = [t for n, t in enumerate(F.tolist()) if n not in (f, g)]
Fc += [[i, p, k], [p, j, k], [j, p, l], [p, i, l]]
Fc = np.array(Fc)
check("Proposition 40.1.4(c): an edge split changes the counts by (1, 3, 2)",
      np.subtract(counts(Fc), c0).tolist() == [1, 3, 2] and chi(Fc) == 2 and dm.oriented_consistently(Fc))
# (d) 1-to-4 subdivision and Exercise 40.1.3
Vt, Ftor = dm.torus_mesh(2, 0.8, 6, 8)
Vs, Fs = dm.subdivide(Vt, Ftor)
n0, n1, n2 = counts(Ftor)
check("Proposition 40.1.4(d): subdivision counts (n0 + n1, 2 n1 + 3 n2, 4 n2), chi unchanged (torus)",
      counts(Fs) == (n0 + n1, 2 * n1 + 3 * n2, 4 * n2) and chi(Fs) == chi(Ftor) == 0)
for L in range(5):
    V, F = dm.icosphere(L)
    deg = np.array([len(n) for n in dm.neighbors(F, len(V))])
    check(f"Exercise 40.1.3: icosphere({L}) has (10 4^L + 2, 30 4^L, 20 4^L), chi = 2, twelve vertices of valence 5",
          counts(F) == (10 * 4 ** L + 2, 30 * 4 ** L, 20 * 4 ** L) and chi(F) == 2 and np.sum(deg != 6) == 12 and np.all(deg[deg != 6] == 5))

# ---------------------------------------------------------------- Lemma 40.1.6 (Girard)
rng = np.random.default_rng(5)
worst = 0.0
for _ in range(200):
    c = rng.normal(size=3)
    c /= np.linalg.norm(c)
    P = c + 0.6 * rng.normal(size=(3, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    if np.dot(P[0], np.cross(P[1], P[2])) < 0:
        P = P[[0, 2, 1]]
    if np.dot(P[0], np.cross(P[1], P[2])) < 1e-3:
        continue
    worst = max(worst, abs(ddg40.solid_angle(*P) - ddg40.girard_area(ddg40.spherical_interior_angles(P))))
check("Lemma 40.1.6(a): Girard's formula = solid angle on 200 random spherical triangles", worst < 1e-10, worst)
# a lune of angle alpha has area 2 alpha (Monte Carlo with 2e5 points, 3 sigma)
U = rng.normal(size=(200000, 3))
U /= np.linalg.norm(U, axis=1, keepdims=True)
alpha = 1.1
in_lune = (U[:, 1] > 0) & (U @ np.array([np.sin(alpha), -np.cos(alpha), 0.0]) > 0)
close("Lemma 40.1.6: lune of angle alpha has area 2 alpha (Monte Carlo)", in_lune.mean() * 4 * np.pi, 2 * alpha, 0.03)
t = np.sort(rng.uniform(0, 2 * np.pi, 7))
P = np.c_[0.5 * np.cos(t), 0.5 * np.sin(t), np.ones(7)]
P /= np.linalg.norm(P, axis=1, keepdims=True)
close("Lemma 40.1.6(b): convex heptagon area = sum of angles - 5 pi", ddg40.spherical_polygon_area(P),
      ddg40.girard_area(ddg40.spherical_interior_angles(P)), 1e-10)

# ---------------------------------------------------------------- Proposition 40.1.7, Example 40.1.8, Figure 40.1.1
V, F = ddg40.example_star()
fs = ddg40.faces_around(F, 0)
N = dm.face_normals(V, F)[fs]
theta = dm.corner_angles(V, F)[fs, 0]
check("Figure 40.1.1: the example star is a convex vertex", ddg40.is_convex_vertex(V, F, 0))
close("Proposition 40.1.7(b): interior angles of P_i are pi - theta_k", ddg40.spherical_interior_angles(N), np.pi - theta, 1e-12)
close("Figure 40.1.1: interior angles 2.298, 2.172, 2.300, 2.175, 2.316, 2.141",
      ddg40.spherical_interior_angles(N), [2.298, 2.172, 2.300, 2.175, 2.316, 2.141], 6e-4)
close("Proposition 40.1.7(c): area(P_i) = delta_i = 0.8363 (Theta_i = 5.4469)",
      [ddg40.spherical_polygon_area(N), 2 * np.pi - theta.sum(), theta.sum()], [0.8363, 0.8363, 5.4469], 5e-5)
c = [V[j] - V[0] for j in ddg40.link_cycle(F, 0)]
pts = []
for k in range(6):
    for s in np.linspace(0, 1, 7):
        for w in np.linspace(0, 1, 7):
            u = (1 - s) * N.sum(0) / 6 + s * ((1 - w) * N[k] + w * N[(k + 1) % 6])
            pts.append(u / np.linalg.norm(u))
pts = np.array(pts)
check("Proposition 40.1.7(a): the fan of P_i lies in the intersection of hemispheres <u, x_j - x_i> <= 0",
      np.all(pts @ np.array(c).T <= 1e-12))
check("Proposition 40.1.7(a): N_1 -> N_2 -> ... is counterclockwise around P_i (positive signed area)",
      ddg40.spherical_polygon_area(N) > 0)
V0, F0 = dm.icosahedron()
Vpert = V0 * (1 + 0.18 * np.random.default_rng(3).uniform(-1, 1, len(V0)))[:, None]
for name, (V, F) in [("icosphere(2)", dm.icosphere(2)), ("perturbed icosahedron", (Vpert, F0))]:
    d = dm.angle_defect(V, F)
    Nf = dm.face_normals(V, F)
    conv = all(ddg40.is_convex_vertex(V, F, i) for i in range(len(V)))
    areas = np.array([ddg40.spherical_polygon_area(Nf[ddg40.faces_around(F, i)]) for i in range(len(V))])
    girard = np.array([ddg40.girard_area(ddg40.spherical_interior_angles(Nf[ddg40.faces_around(F, i)])) for i in range(len(V))])
    check(f"Proposition 40.1.7: {name}: every vertex convex, Gauss image area = Girard = delta", conv
          and np.allclose(areas, d, atol=1e-12) and np.allclose(girard, d, atol=1e-12))
V, F = ddg40.unit_cube()
i7 = int(np.where((V == 1).all(1))[0][0])
N7 = dm.face_normals(V, F)[ddg40.faces_around(F, i7)]
check("Example 40.1.8: the corner (1, 1, 1) has three triangles with normals e1, e2, e3 and is convex",
      len(N7) == 3 and np.allclose(N7[np.argsort(np.argmax(N7, axis=1))], np.eye(3), atol=1e-12) and ddg40.is_convex_vertex(V, F, i7))
close("Example 40.1.8: octant triangle, angles pi/2, area pi/2", [ddg40.spherical_polygon_area(N7)] + list(ddg40.spherical_interior_angles(N7)),
      [np.pi / 2] * 4, 1e-12)

# ---------------------------------------------------------------- Remark 40.1.9
V, F = ddg40.star_mesh(np.array([[1, 0, .5], [0, 1, -.5], [-1, 0, .5], [0, -1, -.5]], float))
d0 = 2 * np.pi - dm.corner_angles(V, F)[:, 0].sum()
Ns = dm.face_normals(V, F)[ddg40.faces_around(F, 0)]
close("Remark 40.1.9: Exercise 38.5.2 star: Theta = 406.15 deg", np.degrees(2 * np.pi - d0), 406.15, 5e-3)
close("Remark 40.1.9: Exercise 38.5.2 star: delta = signed area = -0.8054", [d0, ddg40.spherical_polygon_area(Ns, [0, 0, 1.0])],
      [-0.8054, -0.8054], 6e-5)
V, F = ddg40.saddle_star(0.35)
Ns = dm.face_normals(V, F)[ddg40.faces_around(F, 0)]
close("Remark 40.1.9: saddle_star(0.35): signed area of the Gauss image = delta",
      ddg40.spherical_polygon_area(Ns, [0, 0, 1.0]), 2 * np.pi - dm.corner_angles(V, F)[:, 0].sum(), 1e-12)
V0, F0 = dm.icosahedron()
nb = sorted({v for f in F0.tolist() if 0 in f for v in f} - {0})
W = V0.copy()
W[0] -= 2 * (V0[0] @ V0[0] - (V0[nb] @ V0[0]).mean()) * V0[0]
check("Remark 40.1.9: the dimpled vertex is not convex but has delta = pi/3 (same edge lengths)",
      not ddg40.is_convex_vertex(W, F0, 0) and abs(dm.angle_defect(W, F0)[0] - np.pi / 3) < 1e-12
      and dm.signed_volume(W, F0) < dm.signed_volume(V0, F0))

# ---------------------------------------------------------------- Proposition 40.1.10, (40.1.4), Example 40.1.11, Figure 40.1.2
V, F = ddg40.example_star()
seq = ddg40.link_cycle(F, 0)
h = min(ddg40.point_triangle_distance(V[[0]], V[a], V[b], V[b])[0] for a, b in zip(seq, seq[1:] + seq[:1]))
Theta = dm.corner_angles(V, F)[:, 0].sum()
close("Proposition 40.1.10: h_i = 0.872 for the example star", h, 0.872, 5e-4)
eps = 0.6
area = 0.0
for f, (a, j, k) in enumerate(F):
    u = (V[j] - V[a]) / np.linalg.norm(V[j] - V[a])
    w = V[k] - V[a] - ((V[k] - V[a]) @ u) * u
    w /= np.linalg.norm(w)
    s = np.linspace(0, dm.corner_angles(V, F)[f, 0], 4001)
    P = V[a] + eps * (np.cos(s)[:, None] * u + np.sin(s)[:, None] * w)
    check(f"Proposition 40.1.10: sector {f} lies in its face (distance to the opposite side >= eps)",
          ddg40.point_triangle_distance(V[[a]], V[j], V[k], V[k])[0] >= eps and np.allclose(np.linalg.norm(P - V[a], axis=1), eps))
    area += 0.5 * np.linalg.norm(np.cross(P[1:-1] - P[0], P[2:] - P[0]), axis=1).sum() + 0.5 * np.linalg.norm(np.cross(P[0] - V[a], P[-1] - V[a]))
close("Figure 40.1.2: disk area 0.9804 = Theta eps^2 / 2 (vs pi eps^2 = 1.1310)", [area, 0.5 * Theta * eps ** 2, np.pi * eps ** 2],
      [0.9804, 0.9804, 1.1310], 1e-4)
e = sp.symbols("varepsilon", positive=True)
A_sph = 2 * sp.pi * (1 - sp.cos(e))
sym_equal("(40.1.4): unit sphere: 2 pi - A''(eps) = area of the cap = int_{D_eps} K dA", 2 * sp.pi - sp.diff(A_sph, e, 2), A_sph)
Kp = sp.symbols("K", real=True)
A_ser = sp.pi * e ** 2 - sp.pi / 12 * Kp * e ** 4
sym_equal("(40.1.4): from (39.3.2): 2 pi - A'' = pi K eps^2 (leading term)", 2 * sp.pi - sp.diff(A_ser, e, 2), sp.pi * Kp * e ** 2)
sa = Theta / (2 * np.pi)
close("Example 40.1.11: sin alpha = Theta_i / 2 pi = 0.867", sa, 0.867, 5e-4)
for a, s0 in ((1.2, 0.63), (0.6, 0.31), (0.2, 0.10)):
    s0x = a * (np.pi / 2 - np.arcsin(sa))
    close(f"Figure 40.1.2: a = {a}: the cap ends at s0 = {s0}", s0x, s0, 5e-3)
    # independent: integrate K dA = -rho'' ds dv numerically along the profile (second differences of rho)
    rho, z, s, K = ddg40.rounded_cone_profile(Theta, a, 2.0, n=200001)
    ds = s[1] - s[0]
    check(f"Example 40.1.11: a = {a}: the profile has unit speed", np.allclose(np.hypot(np.diff(rho), np.diff(z)) / ds, 1, atol=1e-6))
    KdA = 2 * np.pi * np.sum(K * rho) * ds                                   # dA = rho ds dv
    close(f"Example 40.1.11: a = {a}: total curvature of the rounded tip = delta = 2 pi - Theta", KdA, 2 * np.pi - Theta, 2e-4)
    close(f"Example 40.1.11: a = {a}: formula 2 pi (1 - rho'(eps)) after the cap", ddg40.rounded_cone_curvature_in_disk(1.5, Theta, a),
          2 * np.pi - Theta, 1e-12)
    eps_in = 0.5 * s0x
    close(f"Example 40.1.11: a = {a}: inside the cap 2 pi (1 - cos(eps / a)) = sphere cap area / a^2",
          ddg40.rounded_cone_curvature_in_disk(eps_in, Theta, a), 2 * np.pi * (1 - np.cos(eps_in / a)), 1e-12)

# ---------------------------------------------------------------- Example 40.1.13, Non-example 40.1.14, Figure 40.1.3
V, F = dm.icosphere(4)
d = dm.angle_defect(V, F)
Am = dm.mass_voronoi(V, F)
close("Example 40.1.13(a): icosphere(4) delta in [0.00435, 0.00586], K_i in [1.0011, 1.0014], sum A_i = 12.5514",
      [d.min(), d.max(), (d / Am).min(), (d / Am).max(), Am.sum()], [0.00435, 0.00586, 1.0011, 1.0014, 12.5514], 5e-5)
S = np.zeros(len(V))
for cc in range(3):
    np.add.at(S, F[:, cc], dm.face_areas(V, F))
close("Non-example 40.1.14: delta / (area of the star) in [0.333, 0.382] on icosphere(4)", [(d / S).min(), (d / S).max()],
      [0.333, 0.382], 5e-4)
close("Non-example 40.1.14: the stars cover the surface three times", S.sum(), 3 * dm.face_areas(V, F).sum(), 1e-10)
V, F = dm.torus_mesh(2, 0.8, 24, 48)
d = dm.angle_defect(V, F)
u = np.arctan2(V[:, 2], np.linalg.norm(V[:, :2], axis=1) - 2)
close("Example 40.1.13(b): torus: positive part 12.4587, negative part -12.4587, max error 0.0075",
      [d[d > 0].sum(), d[d < 0].sum(), np.abs(d / dm.mass_voronoi(V, F) - np.cos(u) / (0.8 * (2 + 0.8 * np.cos(u)))).max()],
      [12.4587, -12.4587, 0.0075], 5e-5)
Kb, Km, ratio = [], [], []
for n in (9, 17, 33, 65, 129):
    V, F = ddg40.saddle_grid(n, 0.5)
    o = int(np.argmin(np.linalg.norm(V, axis=1)))
    dd = dm.angle_defect(V, F)[o]
    Kb.append(dd / dm.mass_barycentric(V, F)[o])
    Km.append(dd / dm.mass_voronoi(V, F)[o])
    ratio.append(dm.mass_barycentric(V, F)[o] / dm.mass_voronoi(V, F)[o])
close("Example 40.1.13(c), Figure 40.1.3(b): saddle origin mixed -3.82, -3.9536, ..., -3.9993", [Km[0], Km[1], Km[-1]],
      [-3.82, -3.9536, -3.9993], 6e-3)
close("Figure 40.1.3(b): saddle origin barycentric -2.91 -> -2.9996", [Kb[0], Kb[-1]], [-2.91, -2.9996], 6e-3)
close("Exercise 40.1.5: barycentric / Voronoi area at the valence-8 origin -> 4/3", ratio[-1], 4 / 3, 2e-3)
s = np.linspace(-0.5, 0.5, 17)
Vg, Fg = dm.grid_mesh(s, s)
deg = np.array([len(x) for x in dm.neighbors(Fg, len(Vg))])
interior = np.setdiff1d(np.arange(len(Vg)), dm.boundary_vertices(Fg))
r4 = (dm.mass_barycentric(Vg, Fg) / dm.mass_voronoi(Vg, Fg))[interior][deg[interior] == 4]
close("Exercise 40.1.5: at valence-4 vertices of the flat grid the barycentric area is 2/3 of the Voronoi area", r4, np.full(len(r4), 2 / 3), 1e-12)
err = {}
for lev in range(1, 6):
    V, F = dm.icosphere(lev)
    d = dm.angle_defect(V, F)
    deg = np.array([len(x) for x in dm.neighbors(F, len(V))])
    for kname, A in (("bary", dm.mass_barycentric(V, F)), ("mixed", dm.mass_voronoi(V, F))):
        for v in (5, 6):
            err[(kname, v, lev)] = np.abs(d[deg == v] / A[deg == v] - 1).max()
close("Figure 40.1.3(a): level 5: mixed 2.6e-4 (valence 5), 3.6e-4 (valence 6); barycentric valence 5 0.146",
      [err[("mixed", 5, 5)], err[("mixed", 6, 5)], err[("bary", 5, 5)]], [2.6e-4, 3.6e-4, 0.146], 6e-5)
close("Figure 40.1.3(a): barycentric valence 6 slows near 0.0024 (levels 4, 5)", [err[("bary", 6, 4)], err[("bary", 6, 5)]],
      [0.0027, 0.0024], 1e-4)
check("Figure 40.1.3(a): mixed errors shrink by about 1/4 per level",
      all(0.2 < err[("mixed", v, L + 1)] / err[("mixed", v, L)] < 0.3 for v in (5, 6) for L in (2, 3, 4)))

# ---------------------------------------------------------------- Remark 40.1.15
V, F = dm.icosphere(3)
hbar = np.mean([ell for t, ell in dm.edge_bending_angles(V, F).values()])
W = ddg40.jitter(V, 0.2 * hbar, seed=1)
W /= np.linalg.norm(W, axis=1, keepdims=True)
d = dm.angle_defect(W, F)
A = dm.mass_voronoi(W, F)
Kq = ddg40.quadric_fit_K(W, F)
check("Remark 40.1.15: jittered icosphere(3): sum delta - 4 pi below 1e-12", abs(d.sum() - 4 * np.pi) < 1e-12)
close("Remark 40.1.15: sum K_fit A_i - 4 pi = -0.875, max |K_i - 1| = 0.138, max |K_fit - 1| = 0.282",
      [(Kq * A).sum() - 4 * np.pi, np.abs(d / A - 1).max(), np.abs(Kq - 1).max()], [-0.875, 0.138, 0.282], 6e-4)
Kq0 = ddg40.quadric_fit_K(V, F)
check("Remark 40.1.15: on the regular icosphere(3) the fit is accurate (|K_fit - 1| < 0.012)", np.abs(Kq0 - 1).max() < 0.012)

# ---------------------------------------------------------------- Exercises 40.1.1, 40.1.2, 40.1.4
apex = np.array([0, 0, 1 / np.sqrt(2)])
base = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]])
Vp = np.vstack([apex, base])
for diag in ((1, 3), (2, 4)):
    a, b = diag
    others = [v for v in (1, 2, 3, 4) if v not in diag]
    Fp = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [a, others[0], b], [a, b, others[1]]]
    Fp = ddg40.orient_outward(Vp, Fp, np.array([0, 0, 0.2]))
    dp = dm.angle_defect(Vp, Fp)
    close(f"Exercise 40.1.1: square pyramid (diagonal {diag}): apex 2 pi/3, base corners 5 pi/6, total 4 pi",
          list(dp) + [dp.sum()], [2 * np.pi / 3] + [5 * np.pi / 6] * 4 + [4 * np.pi], 1e-12)
check("Exercise 40.1.1: all edges of the pyramid have length 1",
      np.allclose([np.linalg.norm(Vp[0] - Vp[k]) for k in range(1, 5)] + [np.linalg.norm(base[k] - base[k - 1]) for k in range(4)], 1))
for name, (V, F) in [("octahedron", ddg40.octahedron()), ("icosahedron", dm.icosahedron())]:
    close(f"Exercise 40.1.2(c): {name}: barycentric area = mixed area (equilateral)", dm.mass_barycentric(V, F), dm.mass_voronoi(V, F), 1e-12)
# a genuinely folded flat star: rotate the half x > 0 about the y-axis
V, F = ddg40.star_mesh(np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float))
R = np.array([[np.cos(1.0), 0, -np.sin(1.0)], [0, 1, 0], [np.sin(1.0), 0, np.cos(1.0)]])
Vf = V.copy()
Vf[1] = R @ V[1]
check("Exercise 40.1.2(b): a star folded along the y-axis has delta = 0 but is not planar",
      abs(2 * np.pi - dm.corner_angles(Vf, F)[:, 0].sum()) < 1e-12 and abs(Vf[1, 2]) > 0.5)
V, F = ddg40.saddle_star(0.35)
th = dm.corner_angles(V, F)[:, 0]
close("Exercise 40.1.4: saddle_star(0.35): cos theta = 0.3363, theta = 1.2278, Theta = 7.3668, delta = -1.0837, Theta/2 = 3.6834",
      [np.cos(th[0]), th[0], th.sum(), 2 * np.pi - th.sum(), th.sum() / 2], [0.3363, 1.2278, 7.3668, -1.0837, 3.6834], 6e-5)
check("Exercise 40.1.4: all six corner angles are equal", np.allclose(th, th[0]))

summary()

"""Verification for Section 40.2 (The Discrete Gauss-Bonnet Theorem).

Run from the project root: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/verify/v-40-2-discrete-gauss-bonnet.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "40-2-discrete-gauss-bonnet.html"
TWO_PI = 2 * np.pi

# ---------------------------------------------------------------- Implementation 40.2 and Exercise 40.2.5
check("Implementation 40.2: at most 30 lines", dgcode.code_lines(PAGE, "code-40-2-1") <= 30)
ns, out = ddg40.run_printed(PAGE, "code-40-2-1")
check("Implementation 40.2: printed output matches the page", out.strip() == ddg40.printed_block(PAGE, "out-40-2-1").strip(), out)
ns2, out2 = ddg40.run_printed(PAGE, "code-ex-40-2-5", dict(ns))
check("Exercise 40.2.5: printed output matches the page", out2.strip() == ddg40.printed_block(PAGE, "out-ex-40-2-5").strip(), out2)


def gb(V, F):
    return dm.angle_defect(V, F).sum() / TWO_PI, dm.euler_characteristic(F)


# ---------------------------------------------------------------- Theorem 40.2.1 and its proof
for name, (V, F) in [("icosphere(4)", dm.icosphere(4)), ("torus_mesh(2, 0.8, 7, 11)", dm.torus_mesh(2, 0.8, 7, 11)),
                     ("torus_mesh(2, 0.8, 48, 96)", dm.torus_mesh(2, 0.8, 48, 96)), ("genus2_mesh(0.16)", dm.genus2_mesh(0.16)),
                     ("tetrahedron", ddg40.tetrahedron()), ("unit cube", ddg40.unit_cube())]:
    s, chi = gb(V, F)
    close(f"Theorem 40.2.1: {name}: sum delta = 2 pi chi (chi = {chi})", s, chi, 1e-10)
    close(f"Theorem 40.2.1 (proof): {name}: sum of all corner angles = pi n2", dm.corner_angles(V, F).sum(), np.pi * len(F), 1e-8)
    n0, n1, n2 = dm.counts(F)
    check(f"Theorem 40.2.1 (proof): {name}: 3 n2 = 2 n1", 3 * n2 == 2 * n1)

# ---------------------------------------------------------------- Remark 40.2.2: no orientation, edge lengths only
RP2 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 2, 4], [2, 3, 5], [3, 4, 1], [4, 5, 2], [5, 1, 3]])
check("Remark 40.2.2: the 6-vertex RP^2 is a closed simplicial surface with chi = 1 and no consistent orientation",
      dm.is_manifold(RP2) and not dm.boundary_edges(RP2) and dm.euler_characteristic(RP2) == 1 and dm.orient(RP2)[1] is False)
rng = np.random.default_rng(4)
for trial in range(3):
    X = rng.normal(size=(6, 3))
    close(f"Remark 40.2.2: RP^2 with random vertex positions (trial {trial}): sum delta = 2 pi", dm.angle_defect(X, RP2).sum(), TWO_PI, 1e-10)
_, Ft = dm.torus_mesh(2, 0.8, 3, 3)
check("Exercise 40.2.4(d): the 3 x 3 torus triangulation is a closed simplicial surface (chi = 0)",
      dm.is_manifold(Ft) and not dm.boundary_edges(Ft) and dm.euler_characteristic(Ft) == 0)
grid = np.array([[i, j] for i in range(3) for j in range(3)], float)          # vertex i*3 + j at (u, v) = (i, j)
flat_theta = np.zeros(Ft.shape)
for f, t in enumerate(Ft):
    P = grid[t]
    P = P[0] + ((P - P[0] + 1.5) % 3 - 1.5)                                    # unwrap the face in the flat torus R^2 / 3Z^2
    for c in range(3):
        a, b = P[(c + 1) % 3] - P[c], P[(c + 2) % 3] - P[c]
        flat_theta[f, c] = np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
Theta = np.zeros(9)
for c in range(3):
    np.add.at(Theta, Ft[:, c], flat_theta[:, c])
close("Exercise 40.2.4(d): with flat edge lengths every angle defect of the 3 x 3 torus is 0", TWO_PI - Theta, np.zeros(9), 1e-12)
check("Exercise 40.2.4(d): the flat faces are right isosceles triangles (angles pi/2, pi/4, pi/4)",
      np.allclose(np.sort(flat_theta, axis=1), np.tile([np.pi / 4, np.pi / 4, np.pi / 2], (len(Ft), 1))))

# ---------------------------------------------------------------- Example 40.2.3
V, F = dm.torus_mesh(2, 0.8, 24, 48)
d = dm.angle_defect(V, F)
close("Example 40.2.3(b): torus parts +12.4587 and -12.4587", [d[d > 0].sum(), d[d < 0].sum()], [12.4587, -12.4587], 5e-5)
V, F = dm.genus2_mesh(0.16)
check("Example 40.2.3(c): genus2_mesh(0.16) counts (4032, 12102, 8068)", dm.counts(F) == (4032, 12102, 8068))
close("Example 40.2.3(c): its smallest face area is 4e-11", dm.face_areas(V, F).min(), 4.04e-11, 1e-12)

# ---------------------------------------------------------------- Example 40.2.4 (equilateral meshes)
for name, (V, F) in [("octahedron", ddg40.octahedron()), ("icosahedron", dm.icosahedron())]:
    deg = np.array([len(n) for n in dm.neighbors(F, len(V))])
    close(f"Example 40.2.4: {name}: delta_i = (6 - deg i) pi / 3", dm.angle_defect(V, F), (6 - deg) * np.pi / 3, 1e-12)
    check(f"Example 40.2.4: {name}: sum (6 - deg i) = 6 chi", np.sum(6 - deg) == 6 * dm.euler_characteristic(F))

# ---------------------------------------------------------------- Corollary 40.2.5
V, F = dm.icosphere(3)
for trial in range(4):
    W = V + np.random.default_rng(10 + trial).normal(0, 0.05, V.shape)
    close(f"Corollary 40.2.5: icosphere(3) with random vertex motion {trial}: sum delta = 4 pi", dm.angle_defect(W, F).sum(), 2 * TWO_PI, 1e-9)

# ---------------------------------------------------------------- Theorem 40.2.6 and Examples 40.2.7, 40.2.8, Figure 40.2.1
P2 = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
VL = np.c_[P2, np.zeros(6)]
FL = np.array([[3, 0, 1], [3, 1, 2], [3, 4, 5], [3, 5, 0]])
tau = dm.angle_defect(VL, FL)
close("Exercise 40.2.2: L-shaped polygon: tau = (pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2), sum 2 pi",
      list(tau) + [tau.sum()], [np.pi / 2] * 3 + [-np.pi / 2] + [np.pi / 2] * 2 + [TWO_PI], 1e-12)
check("Exercise 40.2.2: the four triangles cover the L (total area 3)", abs(dm.face_areas(VL, FL).sum() - 3) < 1e-12)
Vh, Fh = ddg40.icosphere_cap(3, np.pi / 2)
d = dm.angle_defect(Vh, Fh)
b = dm.boundary_vertices(Fh)
inner = np.setdiff1d(np.arange(len(Vh)), b)
close("Example 40.2.7(b): hemisphere: 337 vertices, interior 5.6526, boundary 0.6305, total 2 pi",
      [len(Vh), d[inner].sum(), d[b].sum(), d.sum()], [337, 5.6526, 0.6305, TWO_PI], 5e-5)
Vc, Fc = ddg40.icosphere_cap(3, 2 * np.pi / 3)
d = dm.angle_defect(Vc, Fc)
b = dm.boundary_vertices(Fc)
inner = np.setdiff1d(np.arange(len(Vc)), b)
col = np.arccos(np.clip(Vc[:, 2], -1, 1))
close("Example 40.2.8: cap 2 pi / 3: 481 vertices, 42 on the boundary, interior 8.5710, boundary -2.2878",
      [len(Vc), len(b), d[inner].sum(), d[b].sum()], [481, 42, 8.5710, -2.2878], 5e-5)
close("Example 40.2.8: the boundary stays between colatitudes 1.952 and 2.094; tau > 0 at 28 and < 0 at 14 vertices",
      [col[b].min(), col[b].max(), np.sum(d[b] > 0), np.sum(d[b] < 0)], [1.952, 2.094, 28, 14], 5e-4)
close("Example 40.2.8: smooth cap terms 3 pi and -pi", [TWO_PI * (1 - np.cos(2 * np.pi / 3)), TWO_PI * np.cos(2 * np.pi / 3)],
      [9.4248, -np.pi], 5e-5)
worst, disks = 0.0, True
for t in np.linspace(0.25, np.pi - 0.15, 240):
    Vs, Fs = ddg40.icosphere_cap(3, t)
    disks &= dm.euler_characteristic(Fs) == 1 and dm.is_manifold(Fs)
    worst = max(worst, abs(dm.angle_defect(Vs, Fs).sum() - TWO_PI))
check("Figure 40.2.1: all 240 caps are disks and the totals are 2 pi within 1e-11", disks and worst < 1e-11, worst)
for name, (V, F), chi in [("grid 7 x 4", dm.grid_mesh(np.linspace(0, 3, 7), np.linspace(0, 1, 4)), 1),
                          ("mobius_mesh(20)", dm.mobius_mesh(20), 0)]:
    close(f"Theorem 40.2.6: {name}: sum = 2 pi chi (chi = {chi})", dm.angle_defect(V, F).sum(), TWO_PI * chi, 1e-10)
V, F = dm.icosphere(3)
S = [set(F[k]) for k in (0, 400, 900)]
check("Implementation 40.2: the three removed faces share no vertex (three boundary loops)",
      not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2]))

# ---------------------------------------------------------------- Corollary 40.2.9 (a disk patch of the torus)
V, F = dm.torus_mesh(2, 0.8, 24, 48)
u = np.arctan2(V[:, 2], np.linalg.norm(V[:, :2], axis=1) - 2)
w = np.arctan2(V[:, 1], V[:, 0])
keep = np.all((np.abs(u[F] - 0.3) < 1.0) & (np.abs(w[F]) < 0.6), axis=1)
Vd, Fd, old = dm.submesh(V, F, keep)
check("Corollary 40.2.9: the torus patch is a simplicial disk", dm.euler_characteristic(Fd) == 1 and dm.is_manifold(Fd))
dD = dm.angle_defect(Vd, Fd)
bD = dm.boundary_vertices(Fd)
iD = np.setdiff1d(np.arange(len(Vd)), bD)
close("Corollary 40.2.9: interior defects (original values) = 2 pi - sum tau^D",
      dm.angle_defect(V, F)[old[iD]].sum(), TWO_PI - dD[bD].sum(), 1e-11)

# ---------------------------------------------------------------- Corollary 40.2.10, Remark 40.2.11, Figure 40.2.2
V, F = ddg40.perturbed_icosahedron()
r = np.linalg.norm(V, axis=1)
d = dm.angle_defect(V, F)
th = np.array([t for t, ell in dm.edge_bending_angles(V, F).values()])
close("Figure 40.2.2: scale factors in [0.82, 1.18], min bending angle 0.466, defects in [0.648, 1.509], sum 4 pi",
      [r.min() >= 0.82, r.max() <= 1.18, th.min(), d.min(), d.max(), d.sum()], [1, 1, 0.466, 0.648, 1.509, 2 * TWO_PI], 5e-4)
Nf = dm.face_normals(V, F)
areas = np.array([ddg40.spherical_polygon_area(Nf[ddg40.faces_around(F, i)]) for i in range(12)])
close("Remark 40.2.11: Gauss image areas = angle defects", areas, d, 1e-12)
U = np.random.default_rng(8).normal(size=(400000, 3))
U /= np.linalg.norm(U, axis=1, keepdims=True)
owner = np.argmax(U @ V.T, axis=1)
frac = np.bincount(owner, minlength=12) / len(U) * 2 * TWO_PI
close("Remark 40.2.11: the maximizing vertex cells have the angle defects as areas (Monte Carlo, 4e5 points)", frac, d, 0.03)
inside = np.array([np.all(U[owner == i][:2000] @ (V - V[i]).T <= 1e-12) for i in range(12)])
check("Remark 40.2.11: a maximizing direction of x_i satisfies <u, x_j - x_i> <= 0 for every j (u in P_i)", inside.all())
Vs, Fs = dm.icosphere(5)
own = np.argmax(Vs[Fs].mean(1) @ V.T, axis=1)
cell = np.array([dm.face_areas(Vs, Fs)[own == i].sum() for i in range(12)]) / dm.face_areas(Vs, Fs).sum() * 2 * TWO_PI
check("Figure 40.2.2: the drawn cells match the defects within 0.011", np.abs(cell - d).max() < 0.011, np.abs(cell - d).max())

# ---------------------------------------------------------------- Proposition 40.2.12 and Exercise 40.2.3
V, F = ddg40.unit_cube()
check("Proposition 40.2.12: every corner of the cube has delta = pi/2 != 0", np.allclose(dm.angle_defect(V, F), np.pi / 2))
n0, n1, n2 = dm.counts(F)
k = 7
cut = (n0 - 8 + 2 * k, n1 + k, n2)
check("Exercise 40.2.3(a): cube (8, 18, 12) cut along 7 edges -> (14, 25, 12), chi = 1",
      (n0, n1, n2) == (8, 18, 12) and cut == (14, 25, 12) and cut[0] - cut[1] + cut[2] == 1)
close("Exercise 40.2.3(b): sum of tau over the 14 boundary copies = 14 pi - (all corner angles) = 2 pi",
      14 * np.pi - dm.corner_angles(V, F).sum(), TWO_PI, 1e-12)

# ---------------------------------------------------------------- Non-example 40.2.13 (spherical icosahedron)
V, F = dm.icosahedron()
ang = np.array([ddg40.spherical_interior_angles(V[t]) for t in F])
close("Non-example 40.2.13: spherical icosahedron: every angle is 2 pi / 5", ang, np.full(ang.shape, TWO_PI / 5), 1e-12)
Theta = np.zeros(12)
for c in range(3):
    np.add.at(Theta, F[:, c], ang[:, c])
close("Non-example 40.2.13: vertex angle sums 2 pi, face excess pi/5, total excess 4 pi",
      [np.abs(Theta - TWO_PI).max(), ang[0].sum() - np.pi, (ang.sum(1) - np.pi).sum()], [0, np.pi / 5, 2 * TWO_PI], 1e-12)
close("Non-example 40.2.13: face excess = spherical area (Girard = solid angle)",
      [ddg40.solid_angle(*V[t]) if np.dot(V[t[0]], np.cross(V[t[1]], V[t[2]])) > 0 else -ddg40.solid_angle(*V[t]) for t in F],
      ang.sum(1) - np.pi, 1e-12)

# ---------------------------------------------------------------- Exercise 40.2.1 (triangular prism)
tri = np.array([[1, 0], [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)], [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]])
tri = tri / np.linalg.norm(tri[0] - tri[1])                                    # unit edges
Vp = np.vstack([np.c_[tri, np.zeros(3)], np.c_[tri, np.ones(3)]])
Fp = [[0, 2, 1], [3, 4, 5]]
for a in range(3):
    bb = (a + 1) % 3
    Fp += [[a, bb, bb + 3], [a, bb + 3, a + 3]]
Fp = ddg40.orient_outward(Vp, Fp, np.array([0, 0, 0.5]))
check("Exercise 40.2.1: prism has unit edges and counts (6, 12, 8)",
      np.allclose([np.linalg.norm(Vp[i] - Vp[j]) for i, j in dm.edges(Fp) if (i % 3 == j % 3) or (i // 3 == j // 3)], 1)
      and dm.counts(Fp) == (6, 12, 8))
close("Exercise 40.2.1: every vertex of the prism has delta = 2 pi / 3, total 4 pi", list(dm.angle_defect(Vp, Fp)) + [dm.angle_defect(Vp, Fp).sum()],
      [TWO_PI / 3] * 6 + [2 * TWO_PI], 1e-12)

summary()

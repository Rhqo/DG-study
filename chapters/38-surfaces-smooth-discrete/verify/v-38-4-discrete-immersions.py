"""Verification for Section 38.4 (Simplicial Surfaces and Discrete Immersions).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-4-discrete-immersions.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "38-4-discrete-immersions.html"
ez = np.array([0.0, 0.0, 1.0])
rng = np.random.default_rng(384)

# ---------------------------------------------------------------- Implementation 38.4 and Exercise 38.4.5
check("Implementation 38.4: at most 30 lines", dgcode.code_lines(PAGE, "code-38-4-1") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-4-1")
check("Implementation 38.4: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-38-4-1").strip(),
      out)
star_test = ns["star_test"]
check("Exercise 38.4.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-38-4-5") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-4-5", ns)
check("Exercise 38.4.5: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-ex-38-4-5").strip(), out)
# independent check: on the jittered planar grids a star fails the test iff two of its faces overlap with positive area
V0, F0, inner, noise = ns["V0"], ns["F0"], ns["inner"], ns["noise"]
for amp in (0.2, 0.4, 0.6, 0.8):
    V = V0.copy()
    V[inner, :2] += amp * 0.1 * noise[inner]
    agree = all((not star_test(V, F0, i, ez)) == (ddg38.max_star_overlap(V, F0, i) > 1e-12) for i in inner)
    check(f"Exercise 38.4.5 / Prop 38.4.11: test fails iff faces overlap (jitter {amp:.1f} h)", agree)

# ---------------------------------------------------------------- Proposition 38.4.2, Example 38.4.3, Non-example 38.4.4
V, F = ddg38.octahedron()
for f in F[:3]:
    lam = rng.dirichlet(np.ones(3))
    p = np.zeros(6)
    p[f] = lam
    close("Prop 38.4.2: x(p) = sum p^m x_m = V^T p on a face", V.T @ p, lam @ V[f], 1e-15)
Vt, Ft = dm.torus_mesh(2.0, 0.8, 24, 16)
i, j = 5, 9
close("Example 38.4.3(b): torus_mesh vertex (i, j) = x(2 pi i/24, 2 pi j/16)", Vt[i * 16 + j],
      ddg38.torus(2 * np.pi * i / 24, 2 * np.pi * j / 16), 1e-12)
check("Example 38.4.3(b): torus_mesh has chi = 0 and is a closed oriented surface",
      dm.euler_characteristic(Ft) == 0 and dm.is_manifold(Ft) and dm.oriented_consistently(Ft))
ell0 = ddg38.edge_lengths(*ddg38.hinge(0.0))
for psi in np.linspace(0, np.pi, 7):
    Vh, Fh = ddg38.hinge(psi)
    ell = ddg38.edge_lengths(Vh, Fh)
    n = dm.face_normals(Vh, Fh)
    check(f"Example 38.4.3(c): hinge psi = {psi:.2f}: lengths constant, normal angle = psi",
          all(abs(ell[k] - ell0[k]) < 1e-12 for k in ell0) and abs(np.arccos(np.clip(n[0] @ n[1], -1, 1)) - psi) < 1e-9)
Vi, Fi = dm.icosahedron()
f = Fi[0]
mid = Vi[f].mean(0)
check("Non-example 38.4.4: radial projection is not affine on a face (barycenter maps off the flat triangle)",
      abs(np.linalg.norm(mid / np.linalg.norm(mid)) - 1) < 1e-12 and np.linalg.norm(mid / np.linalg.norm(mid) - mid) > 0.1)

# ---------------------------------------------------------------- Proposition 38.4.6
Vs, Fs = dm.icosphere(2)
D0 = dd.d0(Fs, len(Vs))
dX = D0 @ Vs
E, index = dd.edge_list(Fs)
close("Def 38.4.5: (dx)_e = x_j - x_i", dX, Vs[E[:, 1]] - Vs[E[:, 0]], 0)
check("Prop 38.4.6(b): d1 dx = 0 on the icosphere", np.abs(dd.d1(Fs) @ dX).max() < 1e-14)
# (a): integrate dL along the edge [e_i, e_j] of R^{n0} by quadrature of dL(c'(t)) with c(t) = e_i + t (e_j - e_i)
n0 = len(Vs)
a_, b_ = int(E[7, 0]), int(E[7, 1])
cprime = np.zeros(n0)
cprime[b_], cprime[a_] = 1.0, -1.0
tq = np.linspace(0, 1, 11)
vals = np.array([Vs.T @ cprime for _ in tq])
integral = ((vals[1:] + vals[:-1]) / 2 * np.diff(tq)[:, None]).sum(0)
close("Prop 38.4.6(a): integral of dL over [e_i, e_j] = x_j - x_i", integral, Vs[b_] - Vs[a_], 1e-14)
# (c): a path of edges
path = [0]
nb = dm.neighbors(Fs)
for _ in range(6):
    path.append(nb[path[-1]][-1])
tot = sum((Vs[q] - Vs[p_]) for p_, q in zip(path[:-1], path[1:]))
close("Prop 38.4.6(c): sum of (dx) along a path = x_end - x_start", tot, Vs[path[-1]] - Vs[path[0]], 1e-14)
check("Prop 38.4.6(d): ker d0 = constants on the connected icosphere (rank n0 - 1)",
      np.linalg.matrix_rank(D0) == n0 - 1 and np.allclose(D0 @ np.ones(n0), 0))

# ---------------------------------------------------------------- Proposition 38.4.9 and Corollary 38.4.10
Vh, Fh = ddg38.hinge(np.pi)
nh = dm.face_normals(Vh, Fh)
check("Cor 38.4.10(c): folded hinge (psi = pi) has N_g = -N_f", np.allclose(nh[1], -nh[0]))
Vh0, Fh0 = ddg38.hinge(0.0)
nh0 = dm.face_normals(Vh0, Fh0)
check("Exercise 38.4.3: flat hinge (psi = 0) has N_g = N_f", np.allclose(nh0[1], nh0[0]))
close("Figure 38.4.2: hinge overlap area at psi = pi is 36/101", ddg38.planar_overlap(Vh[Fh[0]], Vh[Fh[1]]), 36 / 101, 1e-12)
for psi in (0.0, 1.0, 2.5, 3.1):
    Vq, Fq = ddg38.hinge(psi)
    # two triangles sharing an edge meet only along it unless coplanar on the same side
    n = dm.face_normals(Vq, Fq)
    side = np.cross(Vq[1] - Vq[0], Vq[2] - Vq[0]) @ np.cross(Vq[1] - Vq[0], Vq[3] - Vq[0])
    check(f"Example 38.4.13(b): hinge psi = {psi}: not folded (N_g != -N_f, x_3 not on the side of x_2 in a common plane)",
          not np.allclose(n[1], -n[0]) and (abs(np.sin(psi)) > 1e-9 or side < 0))
# homothety (38.4.4): pentagram star, two points with the same image near the center
Vb, Fb = ddg38.pentagram_star()
d = np.array([np.cos(np.deg2rad(126)), np.sin(np.deg2rad(126)), 0.0])


def preimage_in_face(V, tri, y):
    """barycentric coordinates of y in the planar triangle tri (z = 0)."""
    A = np.array([[V[tri[0], 0], V[tri[1], 0], V[tri[2], 0]], [V[tri[0], 1], V[tri[1], 1], V[tri[2], 1]], [1, 1, 1]])
    return np.linalg.solve(A, np.array([y[0], y[1], 1.0]))


for eps in (0.2, 0.02, 0.002):
    y = eps * d
    l1 = preimage_in_face(Vb, Fb[0], y)          # face (0, 1, 2)
    l2 = preimage_in_face(Vb, Fb[2], y)          # face (0, 3, 4)
    check(f"Non-example 38.4.15: y = {eps} (cos 126, sin 126) lies in faces (0,1,2) and (0,3,4) (distinct preimages)",
          np.all(l1 >= 0) and np.all(l2 >= 0) and set(Fb[0][l1 > 1e-12]) != set(Fb[2][l2 > 1e-12]))
angb = np.rad2deg(ddg38.projected_star_angles(Vb, Fb, 0, ez))
check("Non-example 38.4.15: all angles 144 deg, sum 720 deg", np.allclose(angb, 144) and np.isclose(angb.sum(), 720))
link_angles = [np.rad2deg(np.arctan2(Vb[k, 1], Vb[k, 0])) % 360 for k in range(1, 6)]
check("Non-example 38.4.15: link vertices at 90, 234, 18, 162, 306 deg", np.allclose(link_angles, [90, 234, 18, 162, 306]))
Va, Fa = ddg38.folded_star()
anga = np.rad2deg(ddg38.projected_star_angles(Va, Fa, 0, ez))
check("Non-example 38.4.14: angles 150, -30, 60, 60, 60, 60 (sum 360)", np.allclose(anga, [150, -30, 60, 60, 60, 60]))
for name, (Vx, Fx) in (("folded star", (Va, Fa)), ("pentagram star", (Vb, Fb))):
    check(f"Non-examples 38.4.14-15: {name}: edges and triangles nondegenerate",
          np.all(dm.face_areas(Vx, Fx) > 1e-3) and min(np.linalg.norm(Vx[a] - Vx[b]) for a, b in dm.edges(Fx)) > 0.1)
    check(f"Non-examples 38.4.14-15: {name}: star test fails and two faces overlap",
          not star_test(Vx, Fx, 0, ez) and ddg38.max_star_overlap(Vx, Fx, 0) > 1e-3)

# ---------------------------------------------------------------- Proposition 38.4.11 on random planar stars
agree, fails, passes = True, 0, 0
for trial in range(300):
    m = rng.integers(3, 9)
    ang = np.sort(rng.uniform(0, 2 * np.pi, m)) if trial % 3 else rng.uniform(0, 4 * np.pi, m)
    Vr, Fr = ddg38.planar_star(ang, rng.uniform(0.4, 1.0, m))
    if np.any(dm.face_areas(Vr, Fr) < 1e-6):
        continue
    t_ok = star_test(Vr, Fr, 0, ez) or star_test(Vr, Fr, 0, -ez)     # n or -n (Proposition 38.4.11)
    o_ok = ddg38.max_star_overlap(Vr, Fr, 0) < 1e-12
    agree &= (t_ok == o_ok)
    fails += not t_ok
    passes += t_ok
check(f"Prop 38.4.11: planar test agrees with exact overlap on random stars ({passes} pass, {fails} fail)",
      agree and fails > 20 and passes > 20)
s = np.linspace(0, 1, 6)
Vg, Fg = dm.grid_mesh(s, s)
inner_g = [i for i in range(len(Vg)) if 0 < Vg[i, 0] < 1 and 0 < Vg[i, 1] < 1]
check("Example 38.4.13(c): every interior star of the grid passes", all(star_test(Vg, Fg, i, ez) for i in inner_g))

# ---------------------------------------------------------------- Exercises 38.4.1, 38.4.2, 38.4.4
T = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
dT = {(0, 1): (0, -2, -2), (0, 2): (-2, 0, -2), (0, 3): (-2, -2, 0), (1, 2): (-2, 2, 0), (1, 3): (-2, 0, 2), (2, 3): (0, -2, 2)}
check("Exercise 38.4.1: tetrahedron edge vectors", all(np.allclose(T[j] - T[i], v) for (i, j), v in dT.items()))
check("Exercise 38.4.1: face (0, 1, 2) sum = 0 and all lengths 2 sqrt 2",
      np.allclose((T[1] - T[0]) + (T[2] - T[1]) + (T[0] - T[2]), 0)
      and all(np.isclose(np.linalg.norm(np.array(v)), 2 * np.sqrt(2)) for v in dT.values()))
check("Exercise 38.4.2: valence m needs 4 pi < m pi, so m >= 5", min(m for m in range(3, 10) if 4 * np.pi < m * np.pi) == 5)
Vo, Fo = ddg38.octahedron()
check("Exercise 38.4.4(a): octahedron vertex 0 passes with n = e1 (four 90 deg angles)",
      star_test(Vo, Fo, 0, np.array([1.0, 0, 0])) and np.allclose(ddg38.projected_star_angles(Vo, Fo, 0, [1, 0, 0]), np.pi / 2))
check("Exercise 38.4.4(a): fails with n = e2 (angles 45, -45, 45, -45 deg)",
      not star_test(Vo, Fo, 0, np.array([0, 1.0, 0]))
      and np.allclose(np.rad2deg(ddg38.projected_star_angles(Vo, Fo, 0, [0, 1, 0])), [45, -45, 45, -45]))

summary()
